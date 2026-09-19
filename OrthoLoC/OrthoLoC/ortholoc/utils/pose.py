from __future__ import annotations

import torch
import numpy as np
import cv2
from packaging import version
from scipy.spatial.transform import Rotation
from loguru import logger
from typing import TYPE_CHECKING
from imcui.ui.utils import ransac_zoo, DEFAULT_RANSAC_METHOD, _filter_matches_poselib, _filter_matches_opencv

if TYPE_CHECKING:
    from ortholoc.correspondences import Correspondences2D2D
    from ortholoc.image_matching.Matcher import Matcher

try:
    import poselib  # noqa

    HAS_POSELIB = True
except ImportError:
    HAS_POSELIB = False

try:
    import pycolmap  # noqa

    version_number = pycolmap.__version__
    if version.parse(version_number) < version.parse("0.5.0"):
        HAS_PYCOLMAP = False
    else:
        HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False


def inv_pose(pose_matrix: np.ndarray) -> np.ndarray | torch.Tensor:
    """
    Invert a pose matrix.

    Args:
        pose_matrix: Pose matrix to invert.

    Returns:
        Inverted pose matrix.
    """
    is_numpy = isinstance(pose_matrix, np.ndarray)
    if is_numpy:
        pose_matrix = torch.from_numpy(pose_matrix)
    R, t = decompose_pose(pose_matrix)
    R_inv = R.transpose(-2, -1)
    t_inv = -R_inv @ t.unsqueeze(-1)
    if is_numpy:
        return compose_pose(R_inv, t_inv).numpy()
    return compose_pose(R_inv, t_inv)


def decompose_pose(pose_matrix: np.ndarray) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """
    Decompose a pose matrix into rotation and translation components.

    Args:
        pose_matrix: Pose matrix to decompose.

    Returns:
        A tuple containing the rotation matrix and translation vector.
    """
    is_numpy = isinstance(pose_matrix, np.ndarray)
    if is_numpy:
        pose_matrix = torch.from_numpy(pose_matrix)
    # should operate on batchified and non-batchified poses
    if pose_matrix.ndimension() == 3:
        t = pose_matrix[:, :3, 3]
        R = pose_matrix[:, :3, :3]
    else:
        t = pose_matrix[:3, 3]
        R = pose_matrix[:3, :3]
    if is_numpy:
        return R.numpy(), t.numpy()
    return R, t


def compose_pose(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Compose a pose matrix from rotation and translation components.

    Args:
        R: Rotation matrix.
        t: Translation vector.

    Returns:
        Composed pose matrix.
    """
    if R.ndimension() == 3:
        t = t.view(-1, 3, 1)
        return torch.cat([R, t], dim=2)
    return torch.cat([R, t.view(3, 1)], dim=1)


def compute_raster_intrinsics_extrinsics(scale: tuple[float, float] | np.ndarray,
                                         offset: tuple[float, float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the raster intrinsics and extrinsics from the scale and offset.
    """
    sx, sy = scale
    ox, oy = offset[:2]
    pose_world2dop = np.array([[1, 0, 0, -ox], [0, 1, 0, -oy], [0, 0, 0, 1]], dtype=np.float32)
    intrinsics_dop = np.array([[1 / sx, 0, 0], [0, 1 / sy, 0], [0, 0, 1]], dtype=np.float32)
    return pose_world2dop, intrinsics_dop


def opencv_to_colmap_intrinsics(K: np.ndarray) -> np.ndarray:
    """
    Convert OpenCV intrinsics to COLMAP format.
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K


def run_pnp(pts2D: np.ndarray, pts3D: np.ndarray, K: np.ndarray, distortion: np.ndarray | None = None,
            mode: str = 'cv2', reprojectionError: float = 5.0, img_size: tuple[int, int] | None = None,
            no_ransac: bool = False, gravity_camera_up: np.ndarray | None = None,
            gravity_world_up: np.ndarray | None = None, gravity_threshold_deg: float = 2.0,
            ransac_seed: int | None = None, pnp_stats: dict | None = None,
            depth_prior=None, depth_threshold_m: float = 10.0,
            prior_fusion: str = 'hard', gravity_scale_deg: float = 10.0, depth_scale_m: float = 10.0,
            gravity_weight: float = 0.1, depth_weight: float = 0.1
            ) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """
    Perform Perspective-n-Point (PnP) pose estimation.
    use OPENCV model for distortion (4 values)
    adapted from https://github.com/naver/dust3r/blob/main/dust3r_visloc/localization.py

    Args:
        pts2D: 2D points in the image.
        pts3D: Corresponding 3D points in the world.
        K: Camera intrinsic matrix.
        distortion: Distortion coefficients (optional).
        mode: PnP mode ('cv2', 'poselib', or 'pycolmap').
        reprojectionError: Maximum reprojection error for RANSAC.
        img_size: Image size (width, height) for certain modes.
        no_ransac: Whether to disable RANSAC.
        gravity_camera_up: Signed Up in OpenCV camera coordinates; None keeps original PnP.
        gravity_world_up: Signed local Up in the SAME world frame as pts3D (not ECEF Z).
        gravity_threshold_deg: Hard angular validity threshold for P3P and refined poses.
        ransac_seed: Optional seed for point subsampling and PoseLib RANSAC.
        pnp_stats: Optional output dictionary of solver diagnostics (excluding the inlier mask).
        depth_prior: Callable on local-ECEF world-to-camera Rt; returns signed DSM residual in metres.
        depth_threshold_m: Hard absolute height-error threshold, including optimized poses.
        prior_fusion: 'hard' gates hypotheses, 'soft' retains legacy score penalties,
            'balanced' keeps visual RANSAC and adds guarded priors only to final BA.
        gravity_scale_deg, depth_scale_m: Positive residual scales, NOT acceptance thresholds.
        gravity_weight, depth_weight: Nonnegative weights; balanced mode uses each as a
            gradient target parameter (default 0.1 means about 0.15 of visual gradient).

    Returns:
        A tuple containing:
        - Success flag (True if PnP succeeded).
        - Estimated pose matrix (camera-to-world).
        - Inlier mask (if applicable).
    """

    use_gravity = gravity_camera_up is not None
    if use_gravity != (gravity_world_up is not None):
        raise ValueError('Both camera and world Up vectors are required for gravity PnP')
    use_depth = depth_prior is not None
    if prior_fusion not in ('hard', 'soft', 'balanced'):
        raise ValueError('prior_fusion must be hard, soft or balanced')
    if (not np.isfinite([gravity_scale_deg, depth_scale_m]).all() or
            gravity_scale_deg <= 0 or depth_scale_m <= 0):
        raise ValueError('Soft prior scales must be finite and positive')
    if (not np.isfinite([gravity_weight, depth_weight]).all() or gravity_weight < 0 or depth_weight < 0):
        raise ValueError('Soft prior weights must be finite and nonnegative')
    if (use_gravity or use_depth or prior_fusion == 'balanced') and (mode != 'poselib' or no_ransac):
        raise ValueError('Gravity/depth prior requires PoseLib with RANSAC enabled')
    if use_depth and (not callable(depth_prior) or not np.isfinite(depth_threshold_m) or depth_threshold_m < 0):
        raise ValueError('Depth prior must be callable, with a finite nonnegative threshold')
    if use_depth and distortion is not None:
        raise ValueError('Depth backprojection currently requires the existing undistorted PINHOLE camera')
    if ransac_seed is not None and ransac_seed < 0:
        raise ValueError('RANSAC seed must be nonnegative')
    if use_gravity or use_depth or prior_fusion == 'balanced':
        from ortholoc.gravity import require_gravity_backend
        gravity_backend = require_gravity_backend()
        if prior_fusion == 'soft' and not getattr(gravity_backend, 'supports_soft_priors', False):
            raise RuntimeError('Rebuild the soft-scoring PnP backend: bash scripts/build_gravity_pnp.sh')
        if prior_fusion == 'balanced' and not getattr(gravity_backend, 'supports_balanced_priors', False):
            raise RuntimeError('Rebuild the balanced PnP backend: bash scripts/build_gravity_pnp.sh')
        if use_depth and not getattr(gravity_backend, 'supports_depth', False):
            raise RuntimeError('Rebuild the depth-enabled PnP backend: bash scripts/build_gravity_pnp.sh')
        if use_depth and getattr(depth_prior, 'diagnostic_dir', None) is not None and not getattr(
                gravity_backend, 'supports_depth_diagnostics', False):
            raise RuntimeError('Rebuild the depth diagnostics backend: bash scripts/build_gravity_pnp.sh')

    num_input_points = len(pts2D)
    sample_idxs = None
    if len(pts2D) > 10_000:
        logger.warning(f"Too many points for PnP, using only the random 10_000")
        # sample 10_000 points
        rng = np.random if ransac_seed is None else np.random.default_rng(ransac_seed)
        idxs = rng.choice(len(pts2D), 10_000, replace=False)
        sample_idxs = idxs
        pts2D = pts2D[idxs]
        pts3D = pts3D[idxs]

    assert mode in ['cv2', 'poselib', 'pycolmap']
    try:
        if len(pts2D) > 4 and mode == "cv2":
            confidence = 0.9999
            iterationsCount = 10_000
            if distortion is not None:
                cv2_pts2ds = np.copy(pts2D)
                cv2_pts2ds = cv2.undistortPoints(cv2_pts2ds, K, np.array(distortion), R=None, P=K)
                pts2D = cv2_pts2ds.reshape((-1, 2))

            if not no_ransac:
                success, r_pose, t_pose, inliers_idxs_unsqueezed = cv2.solvePnPRansac(
                    pts3D, pts2D, K, None, flags=cv2.SOLVEPNP_SQPNP, iterationsCount=iterationsCount,
                    reprojectionError=reprojectionError, confidence=confidence)
                inliers_mask = np.zeros(pts2D.shape[0], dtype=bool)
                inliers_mask[np.squeeze(inliers_idxs_unsqueezed)] = True
            else:
                success, r_pose, t_pose = cv2.solvePnP(pts3D, pts2D, K, None, flags=cv2.SOLVEPNP_SQPNP)
                inliers_mask = None
            if not success:
                return False, None, None
            r_pose = cv2.Rodrigues(r_pose)[0]  # world2cam == world2cam2
            RT = np.r_[np.c_[r_pose, t_pose], [(0, 0, 0, 1)]]  # world2cam2
            return True, np.linalg.inv(RT), inliers_mask  # cam2toworld
        elif len(pts2D) > 4 and mode == "poselib":
            assert HAS_POSELIB
            confidence = 0.9999
            iterationsCount = 10_000
            # NOTE: `Camera` struct currently contains `width`/`height` fields,
            # however these are not used anywhere in the code-base and are provided simply to be consistent with COLMAP.
            # so we put garbage in there
            colmap_intrinsics = opencv_to_colmap_intrinsics(K)
            fx = colmap_intrinsics[0, 0]
            fy = colmap_intrinsics[1, 1]
            cx = colmap_intrinsics[0, 2]
            cy = colmap_intrinsics[1, 2]
            width = img_size[0] if img_size is not None else int(cx * 2)
            height = img_size[1] if img_size is not None else int(cy * 2)

            if distortion is None:
                camera = {'model': 'PINHOLE', 'width': width, 'height': height, 'params': [fx, fy, cx, cy]}
            else:
                camera = {'model': 'OPENCV', 'width': width, 'height': height, 'params': [fx, fy, cx, cy] + distortion}

            pts2D = np.copy(pts2D)
            pts2D[:, 0] += 0.5
            pts2D[:, 1] += 0.5
            ransac_options = {
                'max_reproj_error': reprojectionError,
                'max_iterations': iterationsCount,
                'success_prob': confidence
            }
            if ransac_seed is not None:
                ransac_options['seed'] = ransac_seed
            soft_options = dict(prior_fusion=prior_fusion, gravity_scale_deg=gravity_scale_deg,
                                depth_scale_m=depth_scale_m, gravity_weight=gravity_weight,
                                depth_weight=depth_weight) if prior_fusion in ('soft', 'balanced') else {}
            if use_depth:
                up_cam = gravity_camera_up if use_gravity else np.array([0., 0., 1.])
                up_world = gravity_world_up if use_gravity else np.array([0., 0., 1.])
                RT, meta = gravity_backend.estimate_absolute_pose(
                    pts2D, pts3D, camera, ransac_options, up_cam, up_world, gravity_threshold_deg,
                    depth_evaluator=depth_prior, depth_threshold_m=depth_threshold_m, use_gravity=use_gravity,
                    **soft_options)
                closest = meta.get('closest_depth_candidate')
                if closest is not None:
                    closest['pose_w2c_local_ecef'] = np.asarray(closest['pose_w2c_local_ecef']).tolist()
                if RT is not None and hasattr(depth_prior, 'details') and (prior_fusion == 'hard' or depth_weight > 0):
                    meta.update(depth_prior.details(RT))
                elif RT is None and hasattr(depth_prior, 'save_failure'):
                    try:
                        meta.update(depth_prior.save_failure(
                            pts2D, pts3D, camera, ransac_options, meta, up_cam, up_world,
                            gravity_threshold_deg))
                    except Exception as exc:
                        # Diagnostic I/O must not replace the original solver failure.
                        meta['failure_diagnostic_error'] = str(exc)
                        logger.warning(f'Cannot save failed PnP inputs: {exc}')
            elif use_gravity:
                RT, meta = gravity_backend.estimate_absolute_pose(
                    pts2D, pts3D, camera, ransac_options,
                    gravity_camera_up, gravity_world_up, gravity_threshold_deg, **soft_options)
            elif prior_fusion == 'balanced':
                RT, meta = gravity_backend.estimate_absolute_pose(
                    pts2D, pts3D, camera, ransac_options,
                    [0., 0., 1.], [0., 0., 1.], gravity_threshold_deg,
                    use_gravity=False, **soft_options)
            else:
                pose, meta = poselib.estimate_absolute_pose(pts2D, pts3D, camera, ransac_options, {})
                RT = None if pose is None else pose.Rt
            if pnp_stats is not None:
                pnp_stats.update({key: value for key, value in meta.items() if key != 'inliers'})
                prior_mode = 'gravity_depth' if use_gravity and use_depth else (
                    'depth' if use_depth else 'gravity' if use_gravity else 'none')
                pnp_stats.update(prior_mode=prior_mode,
                                 prior_fusion=prior_fusion if use_gravity or use_depth else 'none',
                                 backend='gravity_pnp_2.0.5' if use_gravity or use_depth or prior_fusion == 'balanced' else 'poselib',
                                 num_input_points=num_input_points, num_used_points=len(pts2D),
                                 seed=0 if ransac_seed is None else ransac_seed,
                                 success=RT is not None)
            if RT is None:
                return False, None, None
            inlier_mask = np.array(meta['inliers'], dtype=bool)
            if sample_idxs is not None:
                full_mask = np.zeros(num_input_points, dtype=bool)
                full_mask[sample_idxs] = inlier_mask
                inlier_mask = full_mask
            RT = np.r_[RT, [(0, 0, 0, 1)]]  # world2cam
            return True, np.linalg.inv(RT), inlier_mask  # cam2toworld
        elif len(pts2D) > 4 and mode == "pycolmap":
            assert HAS_PYCOLMAP
            assert img_size is not None

            pts2D = np.copy(pts2D)
            pts2D[:, 0] += 0.5
            pts2D[:, 1] += 0.5
            colmap_intrinsics = opencv_to_colmap_intrinsics(K)
            fx = colmap_intrinsics[0, 0]
            fy = colmap_intrinsics[1, 1]
            cx = colmap_intrinsics[0, 2]
            cy = colmap_intrinsics[1, 2]
            width = img_size[0]
            height = img_size[1]
            if distortion is None:
                camera_dict = {'model': 'PINHOLE', 'width': width, 'height': height, 'params': [fx, fy, cx, cy]}
            else:
                camera_dict = {
                    'model': 'OPENCV',
                    'width': width,
                    'height': height,
                    'params': [fx, fy, cx, cy] + distortion
                }

            pycolmap_camera = pycolmap.Camera(model=camera_dict['model'], width=camera_dict['width'],
                                              height=camera_dict['height'], params=camera_dict['params'])

            pycolmap_estimation_options = dict(
                ransac=dict(max_error=reprojectionError, min_inlier_ratio=0.01, min_num_trials=1000,
                            max_num_trials=100000, confidence=0.9999))
            pycolmap_refinement_options = dict(refine_focal_length=False, refine_extra_params=False)
            ret = pycolmap.absolute_pose_estimation(pts2D, pts3D, pycolmap_camera,
                                                    estimation_options=pycolmap_estimation_options,
                                                    refinement_options=pycolmap_refinement_options)
            inlier_mask = None
            if ret is None:
                ret = {'success': False}
            else:
                ret['success'] = True
                if callable(ret['cam_from_world'].matrix):
                    retmat = ret['cam_from_world'].matrix()
                else:
                    retmat = ret['cam_from_world'].matrix
                quaternion = Rotation.from_matrix(retmat[:3, :3]).as_quat()
                ret['qvec'] = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
                ret['tvec'] = retmat[:3, 3]
                inlier_mask = ret['inliers']

            if not (ret['success'] and ret['num_inliers'] > 0):
                success = False
                pose = None
            else:
                success = True
                pr_world_to_querycam = np.r_[ret['cam_from_world'].matrix(), [(0, 0, 0, 1)]]
                pose = np.linalg.inv(pr_world_to_querycam)
            return success, pose, inlier_mask
        else:
            return False, None, None
    except Exception as e:
        logger.warning(f'error during pnp: {e}')
        if pnp_stats is not None:
            pnp_stats.update(success=False, error=str(e))
        return False, None, None


def run_calibration(pts2D: np.ndarray, pts3D: np.ndarray, distortion: np.ndarray | None = None, mode: str = 'cv2',
                    reprojectionError: float = 5.0, fix_principle_points: bool = True,
                    focal_length_init: np.ndarray | None = None, img_size: tuple[int, int] | None = None,
                    no_ransac: bool = False) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """
    Perform camera calibration using 2D-3D correspondences.

    Args:
        pts2D: 2D points in the image.
        pts3D: Corresponding 3D points in the world.
        distortion: Distortion coefficients (optional).
        mode: PnP mode ('cv2', 'poselib', or 'pycolmap').
        reprojectionError: Maximum reprojection error for RANSAC.
        fix_principle_points: Whether to fix the principal points during calibration.
        focal_length_init: Initial focal length (optional).
        img_size: Image size (width, height).
        no_ransac: Whether to disable RANSAC.

    Returns:
        A tuple containing:
        - Success flag (True if calibration succeeded).
        - Refined pose matrix (camera-to-world).
        - Refined intrinsics matrix.
    """
    try:
        width, height = img_size
        intrinsics_matrix_init = np.eye(3)
        if focal_length_init is not None:
            intrinsics_matrix_init[0, 0] = focal_length_init[0]
            intrinsics_matrix_init[1, 1] = focal_length_init[1]
        else:
            intrinsics_matrix_init[0, 0] = max(width, height)
            intrinsics_matrix_init[1, 1] = max(width, height)
        intrinsics_matrix_init[0, 2] = width / 2 - 0.5
        intrinsics_matrix_init[1, 2] = height / 2 - 0.5

        success, pose_c2w_pred, _ = run_pnp(pts2D=pts2D, pts3D=pts3D, K=intrinsics_matrix_init, distortion=distortion,
                                            mode=mode, reprojectionError=reprojectionError, img_size=(width, height),
                                            no_ransac=no_ransac)
        flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_FIX_TANGENT_DIST + cv2.CALIB_FIX_K3
        if fix_principle_points:
            flags += cv2.CALIB_FIX_PRINCIPAL_POINT
        if distortion is None:
            flags += cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2
        pose_w2c_pred = inv_pose(pose_c2w_pred)
        rotation = Rotation.from_matrix(pose_w2c_pred[:3, :3])
        translation = pose_w2c_pred[:3, 3]

        reprojection_error, intrinsics_matrix, dist_coeffs, rvecs, tvecs = \
            cv2.calibrateCamera([pts3D.astype(np.float32)],
                                [pts2D.astype(np.float32)],
                                imageSize=(width, height),
                                cameraMatrix=intrinsics_matrix_init.astype(np.float32),
                                distCoeffs=np.zeros((5, 1), dtype=np.float32) if distortion is None else distortion,
                                rvecs=[rotation.as_rotvec().astype(np.float32)],
                                tvecs=[translation.astype(np.float32)],
                                flags=flags)
        pose_w2c_pred = np.eye(4)
        pose_w2c_pred[:3, :3] = Rotation.from_rotvec(np.squeeze(rvecs)).as_matrix()
        pose_w2c_pred[:3, 3] = np.squeeze(tvecs)
        pose_c2w_pred = inv_pose(pose_w2c_pred)

        return True, pose_c2w_pred, intrinsics_matrix
    except Exception as e:
        logger.warning(f'error during calibration: {e}')
        return False, None, None


def adhop_refinement(
    matcher: Matcher, correspondences_2d2d_init: Correspondences2D2D, image_query: np.ndarray, image_dop: np.ndarray,
    dsm: np.ndarray, intrinsics_matrix_gt: np.ndarray | None, use_intrinsics: bool, min_conf: float, num_points: int,
    reprojection_error: float, pnp_mode: str, fix_principle_points: bool, reproj_errors_init: np.ndarray, height: int,
    width: int, silent: bool = False
) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, Correspondences2D2D | None,
           Correspondences2D2D | None]:
    """
    Refine correspondences using AdHoP and perform pose estimation.

    Args:
        matcher: Matcher object for AdHoP refinement.
        correspondences_2d2d_init: Initial 2D-2D correspondences.
        image_query: Query image as a numpy array.
        image_dop: DOP image as a numpy array.
        dsm: DSM data as a numpy array.
        intrinsics_matrix_gt: Ground truth camera intrinsics matrix (3x3).
        use_intrinsics: Whether to use intrinsics for refinement (bool).
        min_conf: Minimum confidence threshold for correspondences (float).
        num_points: Minimum number of points to keep (int).
        reprojection_error: Reprojection error threshold (float).
        pnp_mode: PnP mode to use ('cv2', 'poselib', or 'pycolmap').
        fix_principle_points: Whether to fix principal points during optimization (bool).
        reproj_errors_init: Initial reprojection errors (numpy array).
        height: Height of the image (int).
        width: Width of the image (int).
        silent: Whether to suppress logging (bool).

    Returns:
        A tuple containing refined pose, intrinsics, inliers mask, reprojection errors, and correspondences.
    """
    # refine matches using AdHoP
    correspondences_2d2d_refined = matcher.adhop(correspondences_2d2d=correspondences_2d2d_init, img0=image_query,
                                                 img1=image_dop, min_conf=min_conf, keep_at_least=num_points,
                                                 normalized=True, silent=silent)

    success_refined = False
    pose_c2w_pred_refined = None
    intrinsics_matrix_pred_refined = None
    inliers_mask_refined = None
    opt_reproj_errors_refined = None
    correspondences_2d3d_refined = None

    if correspondences_2d2d_refined.is_valid:
        # lift to 3D
        correspondences_2d3d_refined = correspondences_2d2d_refined.to_2d3d(grid3d_1=dsm)

        if correspondences_2d3d_refined.is_valid:

            success_refined, pose_c2w_pred_refined, intrinsics_matrix_pred_refined, inliers_mask_refined, \
                opt_reproj_errors_refined = correspondences_2d3d_refined.calibrate(
                num_points=num_points, intrinsics_matrix=intrinsics_matrix_gt if use_intrinsics else None,
                width=width, height=height, reprojection_error_diag_ratio=None, reprojection_error=reprojection_error,
                pnp_mode=pnp_mode, fix_principle_points=fix_principle_points
            )

            if success_refined:
                reproj_error_init = np.nanmedian(reproj_errors_init)
                reproj_error_refined = np.nanmedian(opt_reproj_errors_refined)
                """reproj_score = (reproj_error_refined - reproj_error_init) / reproj_error_init if reproj_error_init > 0 else 0.0

                n_inliers_refined = np.count_nonzero(inliers_mask_refined)
                n_inliers_init = np.count_nonzero(inliers_mask_init)
                inliers_score = (n_inliers_refined - n_inliers_init) / n_inliers_init if n_inliers_init > 0 else 0.0

                conf_refined = correspondences_2d3d_refined.confidences[inliers_mask_refined].mean()
                conf_init = correspondences_2d3d_init.confidences[inliers_mask_init].mean()
                confidences_score = (conf_refined - conf_init) / conf_init if conf_init > 0 else 0.0

                overall_score = -2*reproj_score + inliers_score + confidences_score"""

                if reproj_error_init < reproj_error_refined:
                    success_refined = False

    return success_refined, pose_c2w_pred_refined, intrinsics_matrix_pred_refined, \
        inliers_mask_refined, opt_reproj_errors_refined, correspondences_2d2d_refined, correspondences_2d3d_refined


def proc_ransac_matches(mkpts0: np.ndarray, mkpts1: np.ndarray, ransac_method: str = DEFAULT_RANSAC_METHOD,
                        ransac_reproj_threshold: float = 3.0, ransac_confidence: float = 0.99,
                        ransac_max_iter: int = 2000, geometry_type: str = "Homography",
                        silent: bool = False) -> tuple[np.ndarray, np.ndarray] | dict:
    """
    Process matches using RANSAC to filter inliers.

    Args:
        mkpts0: Matched keypoints from the first image (N, 2).
        mkpts1: Matched keypoints from the second image (N, 2).
        ransac_method: RANSAC method to use (str).
        ransac_reproj_threshold: Reprojection error threshold for RANSAC (float).
        ransac_confidence: Confidence level for RANSAC (float).
        ransac_max_iter: Maximum number of iterations for RANSAC (int).
        geometry_type: Type of geometry to estimate ('Homography' or 'Fundamental').
        silent: Whether to suppress logging (bool).

    Returns:
        Filtered matches and the estimated transformation matrix.
    """
    if ransac_method.startswith("CV2"):
        if not silent:
            logger.info(f"ransac_method: {ransac_method}, geometry_type: {geometry_type}")
        return _filter_matches_opencv(
            mkpts0,
            mkpts1,
            ransac_zoo[ransac_method],
            ransac_reproj_threshold,
            ransac_confidence,
            ransac_max_iter,
            geometry_type,
        )
    elif ransac_method.startswith("POSELIB"):
        if not silent:
            logger.info(f"ransac_method: {ransac_method}, geometry_type: {geometry_type}")
        return _filter_matches_poselib(
            mkpts0,
            mkpts1,
            None,
            ransac_reproj_threshold,
            ransac_confidence,
            ransac_max_iter,
            geometry_type,
        )
    else:
        raise NotImplementedError
