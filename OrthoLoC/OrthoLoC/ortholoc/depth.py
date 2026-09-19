"""CPU depth/DSM gate for translated-ECEF PoseLib hypotheses.

Geometry copied from Pilot_0528/pixloc/utils/transform.py and
pixloc/pixlib/geometry/costs.py (_center_points_from_wgs84_poses and the CPU
branch of _query_dsm_heights). Adaptations: explicit measurement pixel/type,
candidate camera centre instead of render pose, and strict invalid DSM handling.
No imports or writes to the reference project, and no CUDA/GDAL dependency.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pyproj
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation as SciRotation


# Same CRS conversions as transform.py; cache the converters for RANSAC calls.
_ecef_crs = {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"}
_ecef_to_wgs84 = pyproj.Transformer.from_crs(_ecef_crs, "EPSG:4326", always_xy=True)
_wgs84_to_ecef = pyproj.Transformer.from_crs("EPSG:4326", _ecef_crs, always_xy=True)


def ECEF_to_WGS84(pos):
    xpjr, ypjr, zpjr = pos
    lon, lat, height = _ecef_to_wgs84.transform(xpjr, ypjr, zpjr, radians=False)
    return [lon, lat, height]


def WGS84_to_ECEF(pos):
    lon, lat, height = pos
    xpjr, ypjr, zpjr = _wgs84_to_ecef.transform(lon, lat, height, radians=False)
    return [xpjr, ypjr, zpjr]


def get_rotation_enu_in_ecef(lon, lat):
    latitude_rad = np.radians(lat)
    longitude_rad = np.radians(lon)
    up = np.array([
        np.cos(longitude_rad) * np.cos(latitude_rad),
        np.sin(longitude_rad) * np.cos(latitude_rad),
        np.sin(latitude_rad)
    ])
    east = np.array([-np.sin(longitude_rad), np.cos(longitude_rad), 0])
    north = np.cross(up, east)
    local_to_world = np.zeros((3, 3))
    local_to_world[:, 0] = east
    local_to_world[:, 1] = north
    local_to_world[:, 2] = up
    return local_to_world


def pose_to_wgs84(rt, ecef_origin):
    """Local w2c -> absolute camera centre and costs.py's WGS84 attitude.

    X_cam = R (X_abs - O) + t. Thus C_abs = -R.T t + O.
    costs.py reconstructs R_c2w = Q_enu_ecef @ EulerXYZ(pitch,roll,yaw) @ F.
    This adapter does NOT pass OrthoLoC's pitch-180 Euler angles unconverted.
    """
    rt = np.asarray(rt, dtype=np.float64)
    R_c2w = rt[:, :3].T
    centre = -R_c2w @ rt[:, 3] + np.asarray(ecef_origin, dtype=np.float64)
    lon, lat, height = ECEF_to_WGS84(centre)
    Q = get_rotation_enu_in_ecef(lon, lat)
    rotation = Q.T @ R_c2w @ np.diag([1.0, -1.0, -1.0])
    # Gimbal lock still produces an equivalent rotation; no yaw prior is used.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        pitch, roll, yaw = SciRotation.from_matrix(rotation).as_euler("xyz", degrees=True)
    return [lon, lat, height, roll, pitch, yaw]


def center_points_from_wgs84_poses(wgs84_poses, depth, K, pixel=None):
    """Copied CPU backprojection from costs.py; K=[w,h,fx,fy,cx,cy]."""
    if len(wgs84_poses) == 0:
        return []
    _, _, fx, fy, cx, cy = K
    depth = float(depth)
    poses_np = np.asarray(wgs84_poses, dtype=np.float64)
    lon, lat, alt = poses_np[:, 0], poses_np[:, 1], poses_np[:, 2]
    roll, pitch, yaw = poses_np[:, 3], poses_np[:, 4], poses_np[:, 5]
    x_ecef, y_ecef, z_ecef = _wgs84_to_ecef.transform(lon.tolist(), lat.tolist(), alt.tolist(), radians=False)
    t_c2w = np.stack([x_ecef, y_ecef, z_ecef], axis=1)
    euler_angles = np.stack([pitch, roll, yaw], axis=1)
    rot_pose_in_enu = SciRotation.from_euler("xyz", euler_angles, degrees=True).as_matrix()
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    up = np.stack([
        np.cos(lon_rad) * np.cos(lat_rad),
        np.sin(lon_rad) * np.cos(lat_rad), np.sin(lat_rad)
    ], axis=1)
    east = np.stack([-np.sin(lon_rad), np.cos(lon_rad), np.zeros_like(lon_rad)], axis=1)
    north = np.cross(up, east)
    rot_enu_in_ecef = np.stack([east, north, up], axis=2)
    R_c2w = np.einsum("nij,njk->nik", rot_enu_in_ecef, rot_pose_in_enu)
    R_c2w[:, :, 1] *= -1
    R_c2w[:, :, 2] *= -1
    K_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    K_inv = np.linalg.inv(K_matrix)
    pixel_center = np.array([cx, cy, 1.0] if pixel is None else [*pixel, 1.0])
    cam_vec = K_inv @ (depth * pixel_center)
    point_3d_ecef = np.einsum("nij,j->ni", R_c2w, cam_vec) + t_c2w
    # transform.py's scalar API also accepts three arrays, unlike costs.py's Nx3 API.
    world_points = np.stack(ECEF_to_WGS84(point_3d_ecef.T.tolist()), axis=1)
    return world_points.tolist()


class DepthDSM:
    """Read-only full DSM, with costs.py's affine/bilinear lookup convention."""

    def __init__(self, area, geotransform, proj_crs, nodata=None):
        self.area = np.asarray(area, dtype=np.float64)
        self.geotransform = tuple(geotransform)
        if len(self.geotransform) != 6 or self.geotransform[2] != 0 or self.geotransform[4] != 0:
            raise ValueError("Depth DSM lookup requires the existing axis-aligned raster convention")
        if self.area.ndim != 2 or self.geotransform[1] == 0 or self.geotransform[5] == 0:
            raise ValueError("Invalid DSM array or geotransform")
        self.nodata = nodata
        self.proj_crs = pyproj.CRS.from_user_input(proj_crs).to_string()
        self.source_path = None
        self.rows, self.cols = self.area.shape
        self.transformer = pyproj.Transformer.from_crs("EPSG:4326", proj_crs, always_xy=True)

    @classmethod
    def from_file(cls, path):
        # Do not import osgeo or write dataset caches in the localization worker.
        import rasterio
        with rasterio.open(path) as dataset:
            if dataset.crs is None:
                raise ValueError(f"DSM has no CRS: {path}")
            result = cls(dataset.read(1), dataset.transform.to_gdal(), dataset.crs, dataset.nodata)
        result.source_path = str(Path(path).resolve())
        return result

    def lookup_details(self, point):
        """Raw lookup coordinates/neighbours, for failure reports only."""
        x, y = self.transformer.transform(*point[:2])
        gt = self.geotransform
        col, row = (x - gt[0]) / gt[1], (y - gt[3]) / gt[5]
        neighbors = None
        if np.isfinite([col, row]).all() and 0 <= col <= self.cols - 1 and 0 <= row <= self.rows - 1:
            c0, r0 = int(np.floor(col)), int(np.floor(row))
            c1, r1 = min(c0 + 1, self.cols - 1), min(r0 + 1, self.rows - 1)
            values = self.area[np.ix_([r0, r1], [c0, c1])]
            neighbors = [[float(v) if np.isfinite(v) else None for v in line] for line in values]
        return {"dsm_projected_xy_m": [float(v) if np.isfinite(v) else None for v in (x, y)],
                "dsm_row_col": [float(v) if np.isfinite(v) else None for v in (row, col)],
                "dsm_neighbors_m": neighbors}

    def query_heights(self, wgs84_points):
        if not len(wgs84_points):
            return []
        points_array = np.asarray(wgs84_points, dtype=np.float64)
        lons, lats = points_array[:, 0], points_array[:, 1]
        x_proj, y_proj = self.transformer.transform(lons.tolist(), lats.tolist())
        x_proj, y_proj = np.asarray(x_proj), np.asarray(y_proj)
        x_origin, x_pixel_size, _, y_origin, _, y_pixel_size = self.geotransform
        col_float = (x_proj - x_origin) / x_pixel_size
        row_float = (y_proj - y_origin) / y_pixel_size
        heights = np.full((len(wgs84_points),), np.nan, dtype=np.float64)
        # Unlike the reference, never treat interpolation outside the last sample as height 0.
        valid = (np.isfinite(col_float) & np.isfinite(row_float) &
                 (col_float >= 0) & (col_float <= self.cols - 1) &
                 (row_float >= 0) & (row_float <= self.rows - 1))
        ids = np.flatnonzero(valid)
        if len(ids):
            coords_x, coords_y = col_float[ids], row_float[ids]
            x0, y0 = np.floor(coords_x).astype(int), np.floor(coords_y).astype(int)
            x1, y1 = np.minimum(x0 + 1, self.cols - 1), np.minimum(y0 + 1, self.rows - 1)
            neighbors = np.stack([self.area[y0, x0], self.area[y0, x1],
                                  self.area[y1, x0], self.area[y1, x1]])
            usable = np.isfinite(neighbors).all(axis=0)
            if self.nodata is not None:
                usable &= (neighbors != self.nodata).all(axis=0)
            if np.any(usable):
                height_array = map_coordinates(self.area, [coords_y[usable], coords_x[usable]], order=1)
                heights[ids[usable]] = height_array
        return heights.tolist()


class DepthPrior:
    """Native gate callback: signed endpoint height minus DSM height, metres."""

    def __init__(self, dsm, ecef_origin, depth, K, width, height,
                 diagnostic_dir=None, image_name=None):
        self.dsm = dsm
        self.ecef_origin = np.asarray(ecef_origin, dtype=np.float64)
        self.depth = float(depth)
        self.K = np.asarray(K, dtype=np.float64)
        if self.ecef_origin.shape != (3,) or not np.isfinite(self.ecef_origin).all():
            raise ValueError("Depth prior requires a finite absolute ECEF crop origin")
        if not np.isfinite(self.depth) or self.depth <= 0:
            raise ValueError("Depth prior must be finite and positive")
        if self.K.shape != (3, 3) or not np.isfinite(self.K).all() or min(self.K[0, 0], self.K[1, 1]) <= 0:
            raise ValueError("Depth prior requires finite known intrinsics")
        # Confirmed by the user: depth_1 is camera-Z at the original array centre.
        self.pixel = [width // 2, height // 2]
        self.intrinsics = [width, height, self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]]
        self.diagnostic_dir = Path(diagnostic_dir) if diagnostic_dir is not None else None
        self.image_name = Path(image_name).name if image_name is not None else None
        if self.diagnostic_dir is not None and self.image_name is None:
            raise ValueError("Failure diagnostics require the query image name")

    def details(self, rt):
        pose_wgs84 = pose_to_wgs84(rt, self.ecef_origin)
        point = center_points_from_wgs84_poses(
            [pose_wgs84], self.depth, self.intrinsics, self.pixel)[0]
        dsm_height = self.dsm.query_heights([point])[0]
        residual = point[2] - dsm_height
        return {"depth_prior_m": self.depth, "depth_type": "z_depth",
                "depth_pixel_convention": "image_center", "depth_pixel_uv": self.pixel,
                "depth_ecef_origin": self.ecef_origin.tolist(), "depth_camera_wgs84": pose_wgs84[:3],
                "depth_point_wgs84": point, "depth_dsm_height_m": dsm_height if np.isfinite(dsm_height) else None,
                "depth_residual_m": residual if np.isfinite(residual) else None,
                "depth_error_m": abs(residual) if np.isfinite(residual) else None}

    def __call__(self, rt):
        residual = self.details(rt)["depth_residual_m"]
        return float("nan") if residual is None else residual

    def save_failure(self, points2D, points3D, camera, options, stats,
                     camera_up, world_up, gravity_threshold_deg):
        """Freeze the exact native inputs, after all sampling and COLMAP +0.5."""
        if self.diagnostic_dir is None:
            return {}
        self.diagnostic_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(self.image_name).stem
        snapshot = self.diagnostic_dir / f"{stem}.npz"
        report_path = self.diagnostic_dir / f"{stem}.json"
        metadata = {"format_version": 1, "image": self.image_name,
                    "points2D_convention": "COLMAP_plus_0.5", "points3D_frame": "translated_ECEF_metres",
                    "pose_convention": "world_to_camera", "camera": camera, "options": options,
                    "gravity_threshold_deg": gravity_threshold_deg,
                    "depth_threshold_m": stats["depth_threshold_m"],
                    "depth_prior_m": self.depth, "depth_type": "z_depth", "depth_pixel_uv": self.pixel,
                    "dsm_path": self.dsm.source_path, "dsm_crs": self.dsm.proj_crs,
                    "dsm_geotransform": self.dsm.geotransform, "dsm_nodata": self.dsm.nodata,
                    "stats": {k: v for k, v in stats.items() if k != "inliers"}}
        encoded = np.array(json.dumps(metadata, allow_nan=False))
        # Exclusive creation: never silently replace an existing failure snapshot.
        with snapshot.open("xb") as file:
            np.savez_compressed(file, points2D=points2D, points3D=points3D, K=self.K,
                                ecef_origin=self.ecef_origin, camera_up=camera_up, world_up=world_up,
                                metadata=encoded)
        report = dict(metadata, ecef_origin=self.ecef_origin.tolist(), K=self.K.tolist())
        closest = stats.get("closest_depth_candidate")
        if closest is not None:
            rt = np.asarray(closest["pose_w2c_local_ecef"], dtype=np.float64)
            report["closest_depth_candidate"] = dict(closest, **self.details(rt))
            report["closest_depth_candidate"].update(
                self.dsm.lookup_details(report["closest_depth_candidate"]["depth_point_wgs84"]))
            report["closest_depth_candidate"]["camera_center_absolute_ecef_m"] = (
                -rt[:, :3].T @ rt[:, 3] + self.ecef_origin).tolist()
            report["closest_depth_candidate"]["camera_pose_wgs84_reference"] = pose_to_wgs84(
                rt, self.ecef_origin)
            report["closest_depth_candidate"]["pose_w2c_absolute_ecef"] = np.column_stack(
                [rt[:, :3], rt[:, 3] - rt[:, :3] @ self.ecef_origin]).tolist()
            report["closest_depth_candidate"]["point_absolute_ecef_m"] = WGS84_to_ECEF(
                report["closest_depth_candidate"]["depth_point_wgs84"])
        else:
            report["closest_depth_candidate"] = None
        with report_path.open("x", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        return {"failure_snapshot_path": str(snapshot), "failure_report_path": str(report_path)}


def load_depth_priors(path, query_names):
    """Align query basenames/stems to image-or-npy-path depth records, never row indices."""
    by_stem = {}
    for name in query_names:
        stem = Path(name).stem
        if stem in by_stem:
            raise ValueError(f"Ambiguous query stem for depth prior: {stem}")
        by_stem[stem] = name
    priors = {}
    for line_num, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_num}: expected image/depth-array path and depth in metres")
        name = by_stem.get(Path(fields[0]).stem)
        if name is None:
            raise ValueError(f"{path}:{line_num}: no query image matches {fields[0]}")
        if name in priors:
            raise ValueError(f"{path}:{line_num}: duplicate depth prior for {name}")
        value = float(fields[1])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{path}:{line_num}: depth must be finite and positive")
        priors[name] = value
    missing = set(query_names) - priors.keys()
    if not priors or missing:
        raise ValueError(f"Depth priors missing for {len(missing)} images, e.g. {sorted(missing)[:5]}")
    return priors
