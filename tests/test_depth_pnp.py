"""CPU-only geometry, native gate and integration regressions; no matcher run."""

import ast
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyproj
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "OrthoLoC" / "OrthoLoC"))
from ortholoc.depth import (DepthDSM, DepthPrior, ECEF_to_WGS84, WGS84_to_ECEF,
                            center_points_from_wgs84_poses, get_rotation_enu_in_ecef,
                            load_depth_priors, pose_to_wgs84)
from ortholoc.gravity import require_gravity_backend
import test_gravity_pnp as gravity_tests
from test_gravity_pnp import synthetic_scene

REFERENCE = Path("/home/amax/Documents/code/pilot_v2/pilot_v2_0702_60ms/Pilot_0528/pixloc")


def depth_scene(noise=0, outliers=0):
    lon, lat, height = 113.0, 28.0, 280.0
    centre = np.array(WGS84_to_ECEF([lon, lat, height]))
    Q = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = Q @ Rotation.from_euler("xyz", [38, -11, 53], degrees=True).as_matrix() @ np.diag([1, -1, -1])
    origin = centre + np.array([350, -200, 90])
    R = R_c2w.T
    t = -R @ (centre - origin)
    rt = np.column_stack([R, t])
    K = np.array([[900., 0, 315.7], [0, 880, 235.1], [0, 0, 1.]])
    pixel = [320, 240]
    depth = 220.
    point = centre + R_c2w @ (depth * np.linalg.solve(K, [*pixel, 1]))
    endpoint = ECEF_to_WGS84(point)
    proj = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4547", always_xy=True)
    x, y = proj.transform(*endpoint[:2])
    dsm = DepthDSM(np.full((501, 501), endpoint[2]), (x - 250, 1, 0, y + 250, 0, -1), "EPSG:4547", -9999)
    prior = DepthPrior(dsm, origin, depth, K, 640, 480)
    rng = np.random.default_rng(12)
    cam_points = rng.uniform([-60, -45, 180], [60, 45, 300], (100, 3))
    X = (cam_points - t) @ R
    points = cam_points[:, :2] / cam_points[:, 2:] * [900, 880] + K[:2, 2]
    points += rng.normal(0, noise, points.shape)
    if outliers:
        points[-outliers:] = rng.uniform([0, 0], [640, 480], (outliers, 2))
    camera = dict(model="PINHOLE", width=640, height=480, params=[900, 880, *K[:2, 2]])
    return points, X, camera, rt, prior, endpoint, centre


class DepthGeometryTest(unittest.TestCase):
    def test_absolute_ecef_and_reference_attitude_roundtrip(self):
        _, _, _, rt, prior, endpoint, centre = depth_scene()
        pose = pose_to_wgs84(rt, prior.ecef_origin)
        np.testing.assert_allclose(WGS84_to_ECEF(pose[:3]), centre, atol=1e-6)
        rebuilt = (get_rotation_enu_in_ecef(*pose[:2]) @
                   Rotation.from_euler("xyz", [pose[4], pose[3], pose[5]], degrees=True).as_matrix() @
                   np.diag([1, -1, -1]))
        np.testing.assert_allclose(rebuilt, rt[:, :3].T, atol=1e-13)
        details = prior.details(rt)
        np.testing.assert_allclose(details["depth_point_wgs84"], endpoint, atol=1e-6)
        self.assertLess(abs(prior(rt)), 1e-6)
        self.assertEqual(details["depth_pixel_uv"], [320, 240])
        self.assertNotEqual(details["depth_pixel_uv"], prior.K[:2, 2].tolist())

    def test_origin_shift_does_not_change_geometry_and_missing_origin_fails(self):
        _, _, _, rt, prior, endpoint, _ = depth_scene()
        delta = np.array([1000, -500, 300])
        shifted_rt = rt.copy()
        shifted_rt[:, 3] += rt[:, :3] @ delta
        shifted = DepthPrior(prior.dsm, prior.ecef_origin + delta, prior.depth, prior.K, 640, 480)
        np.testing.assert_allclose(shifted.details(shifted_rt)["depth_point_wgs84"], endpoint, atol=1e-6)
        wrong = DepthPrior(prior.dsm, [0, 0, 0], prior.depth, prior.K, 640, 480)
        self.assertFalse(np.isfinite(wrong(rt)))

    def test_original_reference_code_matches_copied_code(self):
        # Extract only the selected definitions; never import torch/GDAL/reference packages.
        if not REFERENCE.is_dir():
            self.skipTest("Reference project is not available on this machine")
        source = ast.parse((REFERENCE / "utils/transform.py").read_text())
        nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and
                 n.name in ("ECEF_to_WGS84", "WGS84_to_ECEF", "get_rotation_enu_in_ecef")]
        namespace = {"np": np, "pyproj": pyproj}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "reference_transform", "exec"), namespace)
        _, _, _, rt, prior, _, centre = depth_scene()
        np.testing.assert_allclose(namespace["ECEF_to_WGS84"](centre), ECEF_to_WGS84(centre), atol=1e-9)
        np.testing.assert_allclose(namespace["get_rotation_enu_in_ecef"](113, 28),
                                   get_rotation_enu_in_ecef(113, 28), atol=1e-15)
        source = ast.parse((REFERENCE / "pixlib/geometry/costs.py").read_text())
        nodes = [n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and
                 n.name in ("_center_points_from_wgs84_poses", "_query_dsm_heights")]
        namespace.update(Optional=__import__("typing").Optional, SciRotation=Rotation,
                         torch=SimpleNamespace(is_tensor=lambda _: False), map_coordinates=map_coordinates)
        namespace["ECEF_to_WGS84"] = lambda a: np.stack(ECEF_to_WGS84(a.T), axis=1)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "reference_costs", "exec"), namespace)
        pose = pose_to_wgs84(rt, prior.ecef_origin)
        original = namespace["_center_points_from_wgs84_poses"](
            SimpleNamespace(_wgs84_to_ecef=None), [pose], prior.depth, prior.intrinsics)
        copied = center_points_from_wgs84_poses([pose], prior.depth, prior.intrinsics)
        np.testing.assert_allclose(copied, original, atol=1e-9)
        cache = dict(area=prior.dsm.area, geotransform=prior.dsm.geotransform, proj_crs="EPSG:4547",
                     rows=prior.dsm.rows, cols=prior.dsm.cols, transformer_wgs84_to_proj=prior.dsm.transformer)
        original_h = namespace["_query_dsm_heights"](SimpleNamespace(_dsm_cache={"synthetic": cache}),
                                                      original, "synthetic")
        np.testing.assert_allclose(prior.dsm.query_heights(original), original_h, atol=1e-9)

    def test_dsm_bilinear_slope_boundary_nodata_and_nonfinite(self):
        area = np.arange(16, dtype=float).reshape(4, 4)
        dsm = DepthDSM(area, (100, 1, 0, 40, 0, -1), "EPSG:4326", -9999)
        self.assertAlmostEqual(dsm.query_heights([[101.25, 38.5, 0]])[0], 7.25)
        for p in [[103.1, 38, 0], [99.9, 38, 0], [101, 35.9, 0], [float("nan"), 38, 0]]:
            self.assertTrue(np.isnan(dsm.query_heights([p])[0]))
        self.assertEqual(dsm.query_heights([[103, 37, 0]])[0], 15)
        dsm.area[1, 1] = -9999
        self.assertTrue(np.isnan(dsm.query_heights([[101.25, 38.5, 0]])[0]))
        dsm.area[1, 1] = float("nan")
        self.assertTrue(np.isnan(dsm.query_heights([[101.25, 38.5, 0]])[0]))

    def test_named_depth_records_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.txt"
            # These npy paths need not exist; the text already contains the extracted depths.
            path.write_text("/renders/2_0.npy 220\n/renders/1_0.npy 200\n")
            self.assertEqual(load_depth_priors(path, ["1_0.png", "2_0.png"]), {"1_0.png": 200, "2_0.png": 220})
            for text in ["a.npy nan", "a.npy -1", "a.npy 0", "b.npy 1", "a.npy 1\na.npy 2", "# empty"]:
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    load_depth_priors(path, ["a.png"])
            with self.assertRaises(ValueError):
                load_depth_priors(path, ["a.png", "a.jpg"])

    def test_replay_independent_direct_ecef_check(self):
        spec = importlib.util.spec_from_file_location("depth_replay", REPO / "scripts/verify_depth_pnp.py")
        replay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(replay)
        x, X, camera, rt, prior, _, _ = depth_scene()
        info = dict(depth_threshold_m=10, num_inliers=len(x), inliers=np.ones(len(x), dtype=bool))
        residual, _ = replay.verify_returned_pose(
            rt, info, x, X, camera, {"max_reproj_error": 2}, prior, None, None, False)
        self.assertLess(abs(residual), 1e-6)


class NativeDepthPnPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = require_gravity_backend()
        cls.options = dict(max_reproj_error=2, max_iterations=300, min_iterations=30, seed=7)

    def solve(self, x, X, camera, evaluator, use_gravity=False, up_cam=None, up_world=None, threshold=10):
        return self.backend.estimate_absolute_pose(
            x, X, camera, self.options, [0, 0, 1] if up_cam is None else up_cam,
            [0, 0, 1] if up_world is None else up_world, 2,
            depth_evaluator=evaluator, depth_threshold_m=threshold, use_gravity=use_gravity)

    def test_physical_depth_only_and_combined_gate(self):
        x, X, camera, expected, prior, _, centre = depth_scene(noise=0.2, outliers=20)
        # True ellipsoidal Up, rather than the geocentric centre direction.
        up = get_rotation_enu_in_ecef(113, 28)[:, 2]
        for use_gravity in (False, True):
            rt, info = self.solve(x, X, camera, prior, use_gravity, expected[:, :3] @ up, up)
            self.assertTrue(info["success"])
            self.assertLess(abs(prior(rt)), 10)
            if not use_gravity:
                self.assertGreater(info["depth_rejected_hypotheses"], 0)
            self.assertEqual(info["gravity_enabled"], use_gravity)
            self.assertTrue(info["depth_enabled"])
            np.testing.assert_allclose(rt[:, :3], expected[:, :3], atol=0.001)

    def test_exact_threshold_and_all_rejected_no_placeholder(self):
        x, X, camera, _, _ = synthetic_scene()
        for residual, success in [(9.99, True), (10, True), (-10, True), (10.01, False),
                                  (-10.01, False), (float("nan"), False)]:
            rt, info = self.solve(x, X, camera, lambda p: residual)
            with self.subTest(residual=residual):
                self.assertEqual(info["success"], success)
                if not success:
                    self.assertIsNone(rt)
                    self.assertEqual(info["generated_hypotheses"], info["rejected_hypotheses"])
                    self.assertEqual(info["num_inliers"], 0)
                if np.isnan(residual):
                    self.assertEqual(info["depth_rejected_hypotheses"], info["invalid_depth_hypotheses"])

    def test_gravity_gate_precedes_depth_callback(self):
        x, X, camera, R, _ = synthetic_scene()
        def forbidden(rt):
            raise AssertionError("Rejected gravity hypothesis must not reach depth gate")
        up = np.array([0., 0., 1.])
        rt, info = self.solve(x, X, camera, forbidden, True, -(R @ up), up)
        self.assertIsNone(rt)
        self.assertEqual(info["gravity_rejected_hypotheses"], info["generated_hypotheses"])
        self.assertEqual(info["depth_rejected_hypotheses"], 0)
        self.assertIsNone(info["closest_depth_candidate"])

    def test_closest_candidate_is_minimum_of_gravity_valid_p3p_models(self):
        x, X, camera, R, _ = synthetic_scene(noise=0.2)
        up = np.array([0., 0., 1.])
        camera_up = R @ up
        observed = []
        def reject(rt):
            angle = np.degrees(np.arccos(np.clip(camera_up.dot(rt[:, :3] @ up), -1, 1)))
            self.assertLessEqual(angle, 2 + 1e-6)
            residual = -(1000 + rt[2, 3] ** 2)
            observed.append((rt.copy(), residual))
            return residual
        rt, info = self.solve(x, X, camera, reject, True, camera_up, up)
        self.assertIsNone(rt)
        self.assertGreater(len(observed), 0)
        self.assertEqual(len(observed), info["generated_hypotheses"] - info["nonfinite_hypotheses"] -
                         info["gravity_rejected_hypotheses"])
        expected, residual = min(observed, key=lambda item: abs(item[1]))
        closest = info["closest_depth_candidate"]
        self.assertEqual(closest["stage"], "P3P_before_MSAC")
        self.assertFalse(closest["depth_passed"])
        self.assertEqual(closest["depth_residual_m"], residual)
        np.testing.assert_array_equal(closest["pose_w2c_local_ecef"], expected)

    def test_failure_snapshot_and_read_only_replay(self):
        import subprocess
        import json
        import rasterio
        from rasterio.transform import Affine
        x, X, camera, expected, prior, _, _ = depth_scene()
        up = get_rotation_enu_in_ecef(113, 28)[:, 2]
        camera_up = expected[:, :3] @ up
        camera = dict(camera, params=[camera["params"][0], camera["params"][1],
                                     camera["params"][2] + .5, camera["params"][3] + .5])
        x = x + .5
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            dsm_path = directory / "synthetic.tif"
            with rasterio.open(dsm_path, "w", driver="GTiff", width=prior.dsm.cols,
                               height=prior.dsm.rows, count=1, dtype="float64", crs="EPSG:4547",
                               transform=Affine.from_gdal(*prior.dsm.geotransform), nodata=-9999) as dst:
                dst.write(prior.dsm.area, 1)
            failed_prior = DepthPrior(DepthDSM.from_file(dsm_path), prior.ecef_origin,
                                      prior.depth + 80, prior.K, 640, 480,
                                      diagnostic_dir=directory / "failures", image_name="frame.png")
            rt, stats = self.solve(x, X, camera, failed_prior, True, camera_up, up)
            self.assertIsNone(rt)
            self.assertIsNotNone(stats["closest_depth_candidate"])
            stats["closest_depth_candidate"]["pose_w2c_local_ecef"] = (
                stats["closest_depth_candidate"]["pose_w2c_local_ecef"].tolist())
            paths = failed_prior.save_failure(x, X, camera, self.options, stats, camera_up, up, 2)
            with np.load(paths["failure_snapshot_path"], allow_pickle=False) as frozen:
                np.testing.assert_array_equal(frozen["points2D"], x)
                np.testing.assert_array_equal(frozen["points3D"], X)
                np.testing.assert_array_equal(frozen["K"], prior.K)
                np.testing.assert_array_equal(frozen["ecef_origin"], prior.ecef_origin)
                meta = json.loads(str(frozen["metadata"]))
                self.assertEqual(meta["options"], self.options)
                self.assertEqual(meta["points2D_convention"], "COLMAP_plus_0.5")
            report = json.loads(Path(paths["failure_report_path"]).read_text())
            self.assertEqual(len(report["closest_depth_candidate"]["dsm_neighbors_m"]), 2)
            self.assertIn("camera_center_absolute_ecef_m", report["closest_depth_candidate"])
            before = {p.name: p.read_bytes() for p in (directory / "failures").iterdir()}
            replay = subprocess.run([sys.executable, "-B", str(REPO / "scripts/replay_depth_failure.py"),
                                     "--snapshot", paths["failure_snapshot_path"]],
                                    capture_output=True, text=True, check=True)
            self.assertTrue(json.loads(replay.stdout)["replay_matches_failure"])
            self.assertEqual(before, {p.name: p.read_bytes() for p in (directory / "failures").iterdir()})
            with self.assertRaises(FileExistsError):
                failed_prior.save_failure(x, X, camera, self.options, stats, camera_up, up, 2)

    def test_no_finite_depth_residual_has_no_closest_candidate(self):
        x, X, camera, _, _ = synthetic_scene()
        rt, info = self.solve(x, X, camera, lambda rt: float("nan"))
        self.assertIsNone(rt)
        self.assertIsNone(info["closest_depth_candidate"])

    def test_refinement_rollback_preserves_depth_feasibility(self):
        x, X, camera, R, _ = synthetic_scene(noise=1.0)
        # Synthetic pose-dependent scalar residual with a feasible shell away from the BA optimum.
        up = gravity_tests.ecef_up(113, 28)
        camera_up = R @ up
        axis = np.cross(camera_up, [1, 0, 0])
        axis /= np.linalg.norm(axis)
        shifted_up = Rotation.from_rotvec(axis * np.deg2rad(0.4)).apply(camera_up)
        def evaluator(rt):
            angle = np.degrees(np.arccos(np.clip((rt[:, :3] @ up).dot(shifted_up), -1, 1)))
            return angle * 1000
        options = dict(self.options, max_reproj_error=3, max_iterations=500, min_iterations=100)
        rt, info = self.backend.estimate_absolute_pose(
            x, X, camera, options, [0, 0, 1], [0, 0, 1], 2,
            depth_evaluator=evaluator, depth_threshold_m=200, use_gravity=False)
        self.assertTrue(info["success"])
        self.assertLessEqual(abs(evaluator(rt)), 200)
        self.assertGreater(info["rejected_refinements"], 0)

    def test_callback_errors_and_invalid_threshold_not_silently_ignored(self):
        x, X, camera, _, _ = synthetic_scene()
        with self.assertRaises(ValueError):
            self.solve(x, X, camera, lambda p: 0, threshold=-1)
        with self.assertRaises(ValueError):
            self.solve(x, X, camera, 4)
        def error(rt):
            raise RuntimeError("depth geometry failure")
        with self.assertRaisesRegex(RuntimeError, "depth geometry failure"):
            self.solve(x, X, camera, error)


class DepthIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gravity_tests.PnPIntegrationTest.setUpClass.__func__(cls)

    def test_depth_passthrough_and_opencv_pixel_convention(self):
        x, X, _, rt, prior, _, _ = depth_scene()
        from ortholoc.correspondences.Correspondences2D3D import Correspondences2D3D
        correspondences = Correspondences2D3D(pts0=x, pts1=X, confidences=np.ones(len(x)), is_normalized=False)
        stats = {}
        success, c2w, _, mask, _ = correspondences.calibrate(
            num_points=None, width=640, height=480, intrinsics_matrix=prior.K,
            depth_prior=prior, depth_threshold_m=10, ransac_seed=7, pnp_stats=stats)
        self.assertTrue(success)
        self.assertEqual(c2w.dtype, np.float64)
        self.assertEqual(stats["prior_mode"], "depth")
        self.assertEqual(stats["depth_pixel_uv"], [320, 240])
        self.assertLess(stats["depth_error_m"], 1e-5)
        self.assertEqual(mask.sum(), len(x))
        np.testing.assert_allclose(c2w[:3, :3], rt[:, :3].T, atol=1e-8)

    def test_failed_wrapper_freezes_exact_native_inputs(self):
        import json
        from unittest.mock import Mock
        x, X, camera, _, _ = synthetic_scene()
        K = np.array([[900., 0, 320.], [0, 880., 240.], [0, 0, 1.]])
        prior = Mock(side_effect=lambda rt: 1000.)
        prior.diagnostic_dir = "diagnostics_enabled"
        prior.save_failure.return_value = {"failure_snapshot_path": "/test/frame.npz"}
        stats = {}
        success, c2w, mask = self.pose.run_pnp(x, X, K, mode="poselib", img_size=(640, 480),
                                              depth_prior=prior, ransac_seed=7, pnp_stats=stats)
        self.assertFalse(success)
        self.assertIsNone(c2w)
        self.assertIsNone(mask)
        args = prior.save_failure.call_args.args
        np.testing.assert_array_equal(args[0], x + .5)
        np.testing.assert_array_equal(args[1], X)
        self.assertEqual(args[2]["params"], [900, 880, 320.5, 240.5])
        self.assertEqual(args[3]["seed"], 7)
        self.assertEqual(stats["failure_snapshot_path"], "/test/frame.npz")
        json.dumps(stats, allow_nan=False)

    def test_main_passes_depth_context_and_keeps_local_pose(self):
        from unittest.mock import Mock
        import logging
        import os
        source = ast.parse((REPO / "main.py").read_text())
        node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and
                    n.name == "run_ortholoc_localization_on_crop")
        _, _, _, rt, prior, _, _ = depth_scene()
        c2w = np.column_stack([rt[:, :3].T, -rt[:, :3].T @ rt[:, 3]])
        matches = Mock(is_valid=True)
        matches.__len__ = Mock(return_value=100)
        matches.take_min_conf.return_value = matches
        matches.take_covisible.return_value = matches
        correspondences = Mock()
        matches.to_2d3d.return_value = correspondences
        correspondences.calibrate.return_value = (True, c2w, prior.K, np.ones(100, bool), np.zeros(100))
        matcher = Mock(angles=[0])
        matcher.run.return_value = [matches]
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        utils = SimpleNamespace(io=SimpleNamespace(load_image=lambda _: image), pose=self.pose)
        namespace = dict(np=np, os=os, logging=logging, load_ortholoc_components=lambda: (utils, None, None))
        exec(compile(ast.Module(body=[node], type_ignores=[]), "main_depth_forward", "exec"), namespace)
        result = namespace[node.name]("test.png", image, np.zeros((480, 640, 3)), prior.K, matcher, [0],
                                      depth_prior=prior, depth_threshold_m=10, pnp_seed=0)
        self.assertIs(correspondences.calibrate.call_args.kwargs["depth_prior"], prior)
        self.assertEqual(correspondences.calibrate.call_args.kwargs["depth_threshold_m"], 10)
        np.testing.assert_allclose(result["pose_c2w"][:3], c2w, atol=1e-12)

    def test_actual_dsm_can_load_after_solver_imports(self):
        path = Path("/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3_DSM_merge.tif")
        if not path.is_file():
            self.skipTest("Local DSM not available on this machine")
        dsm = DepthDSM.from_file(path)
        self.assertEqual(dsm.area.shape, (5191, 8921))
        row = dsm.rows // 2
        valid_cols = np.flatnonzero(np.isfinite(dsm.area[row]) & (dsm.area[row] != dsm.nodata))
        self.assertGreater(len(valid_cols), 0)
        col = valid_cols[len(valid_cols) // 2]
        gt = dsm.geotransform
        lon, lat = pyproj.Transformer.from_crs("EPSG:4547", "EPSG:4326", always_xy=True).transform(
            gt[0] + col * gt[1], gt[3] + row * gt[5])
        self.assertAlmostEqual(dsm.query_heights([[lon, lat, 0]])[0], dsm.area[row, col], places=4)


class DepthLauncherTest(unittest.TestCase):
    launch_without_localization = gravity_tests.LauncherConfigTest.launch_without_localization

    def test_depth_modes_and_paths(self):
        for mode in ("depth", "gravity_depth"):
            args = self.launch_without_localization(["--ortholoc_pnp_prior", mode, "--ortholoc_pnp_seed", "0"])
            self.assertEqual(args.ortholoc_pnp_prior, mode)
            self.assertEqual(args.ortholoc_depth_prior_file,
                             "/media/amax/AE0E2AFD0E2ABE69/datasets/depth_1/DJI_20250612193930_0012_V.txt")
            self.assertEqual(args.ortholoc_depth_threshold_m, 10)

    def test_depth_path_and_threshold_overrides(self):
        args = self.launch_without_localization(
            ["--ortholoc_pnp_prior", "depth", "--ortholoc_depth_prior_file", "/cli/depth.txt",
             "--ortholoc_depth_threshold_m", "12"], {"ORTHOLOC_DEPTH_PRIOR_FILE": "/env/depth.txt"})
        self.assertEqual(args.ortholoc_depth_prior_file, "/cli/depth.txt")
        self.assertEqual(args.ortholoc_depth_threshold_m, 12)


if __name__ == "__main__":
    unittest.main()
