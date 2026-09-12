import numpy as np
import pyproj
from scipy.spatial.transform import Rotation as R

WGS84_TO_ECEF = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
ECEF_TO_WGS84 = pyproj.Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)


def normalize_K(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape}")
    if abs(K[2, 2]) < 1e-12:
        raise ValueError("Invalid K: K[2,2] == 0")
    return K / K[2, 2]


def convert_euler_to_matrix(euler_xyz):
    """
    Keep the same convention as the user's original CGCS2000 code.
    Input euler_xyz order: [yaw, pitch, roll].
    """
    ret = R.from_euler(
        'xyz',
        [float(euler_xyz[1] - 90.0), float(euler_xyz[2]), float(-euler_xyz[0])],
        degrees=True,
    )
    return ret.as_matrix().astype(np.float64)


def wgs84_to_ecef(lon, lat, h):
    x, y, z = WGS84_TO_ECEF.transform(lon, lat, h)
    if np.isscalar(x):
        return np.array([x, y, z], dtype=np.float64)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(z, dtype=np.float64)


def ecef_to_wgs84(x, y, z):
    lon, lat, h = ECEF_TO_WGS84.transform(x, y, z)
    return np.array([lon, lat, h], dtype=np.float64)


def enu_to_ecef_rotation(lon_deg: float, lat_deg: float) -> np.ndarray:
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0], dtype=np.float64)
    north = np.array([
        -np.sin(lat) * np.cos(lon),
        -np.sin(lat) * np.sin(lon),
        np.cos(lat),
    ], dtype=np.float64)
    up = np.array([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ], dtype=np.float64)
    return np.column_stack([east, north, up])


def transform_colmap_pose_intrinsic(pose_data, K: np.ndarray):
    lon, lat, alt, roll, pitch, yaw = map(float, pose_data)

    # Keep the same preprocessing as the user's original stable code.
    yaw_proc = -yaw
    pitch_proc = pitch - 90.0
    euler_xyz = [yaw_proc, pitch_proc, roll]

    r_local_c2w = convert_euler_to_matrix(euler_xyz)
    t_c2w_ecef = wgs84_to_ecef(lon, lat, alt)

    # Map the original local projected-world convention onto a local ENU frame,
    # then lift it into ECEF.
    r_ecef_from_enu = enu_to_ecef_rotation(lon, lat)
    r_ecef_from_cam = r_ecef_from_enu @ r_local_c2w

    pose_w2c = np.eye(4, dtype=np.float64)
    pose_w2c[:3, :3] = r_ecef_from_cam.T
    pose_w2c[:3, 3] = -r_ecef_from_cam.T @ t_c2w_ecef

    K = normalize_K(K)

    # print("\n[DEBUG][transform_colmap_pose_intrinsic]")
    # print("pose_data =", pose_data)
    # print("yaw_proc, pitch_proc, roll =", yaw_proc, pitch_proc, roll)
    # print("r_local_c2w =\n", r_local_c2w)
    # print("r_ecef_from_cam =\n", r_ecef_from_cam)
    # print("t_c2w_ecef =", t_c2w_ecef)
    # print("K(normalized) =\n", K)
    # print("pose_w2c =\n", pose_w2c)

    return pose_w2c, K, {
        "lon": lon,
        "lat": lat,
        "alt": alt,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "yaw_processed": yaw_proc,
        "pitch_processed": pitch_proc,
    }
