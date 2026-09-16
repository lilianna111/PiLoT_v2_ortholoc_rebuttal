from typing import Optional
import numpy as np
from scipy.ndimage import map_coordinates

from .transform_colmap import ecef_to_wgs84, normalize_K


class RayCastError(RuntimeError):
    pass


def is_valid_dsm_value(z, nodata=None, min_elevation: float = -1000.0, max_elevation: float = 10000.0):
    if not np.isfinite(z):
        return False
    if nodata is not None and np.isclose(z, nodata):
        return False
    if z <= -9990:
        return False
    if z < min_elevation or z > max_elevation:
        return False
    return True


def lonlat_to_rowcol_float(lon: float, lat: float, geotransform):
    x_origin, x_pixel_size, _, y_origin, _, y_pixel_size = geotransform
    col = (lon - x_origin) / x_pixel_size
    row = (lat - y_origin) / y_pixel_size
    return float(row), float(col)


def sample_dsm_height_wgs84(dsm_array: np.ndarray, geotransform, lon: float, lat: float, nodata=None):
    row, col = lonlat_to_rowcol_float(lon, lat, geotransform)
    h, w = dsm_array.shape

    if row < 0 or row > h - 1 or col < 0 or col > w - 1:
        return None, row, col

    z = map_coordinates(
        dsm_array,
        [[row], [col]],
        order=1,
        mode="nearest",
        prefilter=False,
    )[0]

    if not is_valid_dsm_value(z, nodata=nodata):
        return None, row, col

    return float(z), row, col


def pixel_ray_direction_ecef(K: np.ndarray, pose_w2c: np.ndarray, uv):
    K = normalize_K(K)
    u, v = float(uv[0]), float(uv[1])

    k_inv = np.linalg.inv(K)
    d_cam = k_inv @ np.array([u, v, 1.0], dtype=np.float64)
    d_cam /= np.linalg.norm(d_cam)

    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]

    cam_center_ecef = -R.T @ t
    d_ecef = R.T @ d_cam
    d_ecef /= np.linalg.norm(d_ecef)
    return cam_center_ecef, d_ecef


def _point_on_ray(cam_center_ecef, d_ecef, t):
    return cam_center_ecef + t * d_ecef


def _eval_ray_point(dsm_array, geotransform, nodata, cam_center_ecef, d_ecef, t):
    p = _point_on_ray(cam_center_ecef, d_ecef, t)
    lon, lat, h_ray = ecef_to_wgs84(*p)
    h_dsm, row, col = sample_dsm_height_wgs84(dsm_array, geotransform, lon, lat, nodata=nodata)
    return p, lon, lat, h_ray, h_dsm, row, col


def ray_cast_to_wgs84_dsm(
    dsm_array: np.ndarray,
    geotransform,
    K: np.ndarray,
    pose_w2c: np.ndarray,
    uv,
    num_samples: int = 48,          # 这里只保留接口，实际默认用很小值
    t_near: float = 1.0,
    t_far: float = 20000.0,
    bisection_steps: int = 10,
    nodata=None,
    fallback_height: Optional[float] = None,
):
    cam_center_ecef, d_ecef = pixel_ray_direction_ecef(K, pose_w2c, uv)

    # 1) 粗采样：只采 48 个点
    ts = np.linspace(t_near, t_far, int(num_samples), dtype=np.float64)

    prev_valid = None
    best = None

    for t in ts:
        p, lon, lat, h_ray, h_dsm, row, col = _eval_ray_point(
            dsm_array, geotransform, nodata, cam_center_ecef, d_ecef, t
        )

        if h_dsm is None:
            prev_valid = None
            continue

        diff = h_ray - h_dsm
        cur = (t, p, lon, lat, h_ray, h_dsm, row, col, diff)

        if best is None or abs(diff) < abs(best[-1]):
            best = cur

        if prev_valid is not None and diff * prev_valid[-1] <= 0:
            # 2) 只在局部区间做二分
            lo_t = prev_valid[0]
            hi_t = t
            lo_diff = prev_valid[-1]

            for _ in range(int(bisection_steps)):
                mid_t = 0.5 * (lo_t + hi_t)
                _, lon_m, lat_m, h_mid, h_dsm_mid, row_m, col_m = _eval_ray_point(
                    dsm_array, geotransform, nodata, cam_center_ecef, d_ecef, mid_t
                )

                if h_dsm_mid is None:
                    hi_t = mid_t
                    continue

                diff_mid = h_mid - h_dsm_mid
                if diff_mid * lo_diff <= 0:
                    hi_t = mid_t
                else:
                    lo_t = mid_t
                    lo_diff = diff_mid

            hit_t = 0.5 * (lo_t + hi_t)
            hit, lon_h, lat_h, h_hit, h_dsm_hit, row_h, col_h = _eval_ray_point(
                dsm_array, geotransform, nodata, cam_center_ecef, d_ecef, hit_t
            )

            if h_dsm_hit is not None:
                row_i = int(np.clip(round(row_h), 0, dsm_array.shape[0] - 1))
                col_i = int(np.clip(round(col_h), 0, dsm_array.shape[1] - 1))
                return hit, np.array([lon_h, lat_h, h_dsm_hit], dtype=np.float64), (row_i, col_i), {
                    "source": "dsm_intersection",
                    "row_float": float(row_h),
                    "col_float": float(col_h),
                    "row_int": row_i,
                    "col_int": col_i,
                }

        prev_valid = cur

    # 3) 没找到过零点，就直接返回最近点
    if best is not None:
        _, p, lon, lat, _, h_dsm, row, col, _ = best
        row_i = int(np.clip(round(row), 0, dsm_array.shape[0] - 1))
        col_i = int(np.clip(round(col), 0, dsm_array.shape[1] - 1))
        return p, np.array([lon, lat, h_dsm], dtype=np.float64), (row_i, col_i), {
            "source": "nearest_valid_sample",
            "row_float": float(row),
            "col_float": float(col),
            "row_int": row_i,
            "col_int": col_i,
        }

    # 4) 全程无有效 DSM，用 fallback 平面
    if fallback_height is not None:
        best_t = None
        best_abs = None
        ts2 = np.linspace(t_near, t_far, 48, dtype=np.float64)
        for t in ts2:
            p = _point_on_ray(cam_center_ecef, d_ecef, t)
            lon, lat, h_ray = ecef_to_wgs84(*p)
            diff = h_ray - fallback_height
            if best_abs is None or abs(diff) < best_abs:
                best_abs = abs(diff)
                best_t = t

        p = _point_on_ray(cam_center_ecef, d_ecef, best_t)
        lon, lat, _ = ecef_to_wgs84(*p)
        row, col = lonlat_to_rowcol_float(lon, lat, geotransform)
        row_i = int(np.clip(round(row), 0, dsm_array.shape[0] - 1))
        col_i = int(np.clip(round(col), 0, dsm_array.shape[1] - 1))
        return p, np.array([lon, lat, fallback_height], dtype=np.float64), (row_i, col_i), {
            "source": "fallback_plane",
            "row_float": float(row),
            "col_float": float(col),
            "row_int": row_i,
            "col_int": col_i,
        }

    raise RayCastError("No ray/DSM intersection found.")
