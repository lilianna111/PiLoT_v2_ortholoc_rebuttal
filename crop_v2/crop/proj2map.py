from __future__ import annotations

import os
import time

import cv2
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import Affine

from .ray_casting import ray_cast_to_wgs84_dsm
from .transform_colmap import normalize_K, transform_colmap_pose_intrinsic, wgs84_to_ecef


def read_raster(path: str):
    with rasterio.open(path) as ds:
        data = ds.read()
        transform = ds.transform
        profile = ds.profile
        nodata = ds.nodata
    return data, transform, profile, nodata


def read_dsm_dom(dsm_path: str, dom_path: str):
    dsm, dsm_transform, dsm_profile, dsm_nodata = read_raster(dsm_path)
    dom, dom_transform, dom_profile, _ = read_raster(dom_path)

    if dsm.shape[0] != 1:
        raise ValueError(f"DSM should have 1 band, got shape {dsm.shape}")

    dsm_array = dsm[0].astype(np.float64)
    dom_array = np.transpose(dom[:3], (1, 2, 0)) if dom.shape[0] >= 3 else np.transpose(np.repeat(dom, 3, axis=0), (1, 2, 0))
    dom_array = np.ascontiguousarray(dom_array)
    return dsm_array, dsm_transform, dsm_profile, dsm_nodata, dom_array, dom_transform, dom_profile


def affine_to_gdal_tuple(transform: Affine):
    return (transform.c, transform.a, transform.b, transform.f, transform.d, transform.e)


def image_corners(image_width: int, image_height: int):
    return np.array([
        [0, 0],
        [image_width - 1, 0],
        [image_width - 1, image_height - 1],
        [0, image_height - 1],
    ], dtype=np.float64)


def _prepare_dom_for_warp(dom_crop: np.ndarray) -> np.ndarray:
    if dom_crop.dtype == np.uint8:
        return dom_crop
    arr = dom_crop.astype(np.float32)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - mn) / (mx - mn)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def save_crop_debug_overlay(dom_array, hit_rc_list, save_path):
    img = _prepare_dom_for_warp(dom_array).copy()
    pts = []
    for rc in hit_rc_list:
        row, col = rc
        pts.append([int(col), int(row)])
    pts = np.array(pts, dtype=np.int32)

    for i, (x, y) in enumerate(pts):
        cv2.circle(img, (x, y), 14, (255, 0, 0), -1)
        cv2.putText(img, str(i), (x + 18, y - 18), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)

    if len(pts) >= 2:
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=5)

    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def compute_min_valid_dsm_height(dsm_array: np.ndarray, dsm_nodata):
    valid = np.isfinite(dsm_array)
    valid &= (dsm_array > 0)

    if dsm_nodata is not None:
        valid &= ~np.isclose(dsm_array, dsm_nodata)

    if not np.any(valid):
        raise RuntimeError("DSM has no valid height > 0 for fallback.")

    min_h = float(np.min(dsm_array[valid]))
    # print(f"[DEBUG] min_valid_dsm_height = {min_h:.6f}")
    return min_h


def move_uv_towards_center(uv, center, ratio: float):
    uv = np.asarray(uv, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    return uv + ratio * (center - uv)


def build_corner_candidate_uvs(image_width: int, image_height: int):
    corners = image_corners(image_width, image_height)
    center = np.array([(image_width - 1) * 0.5, (image_height - 1) * 0.5], dtype=np.float64)

    candidate_uvs = []
    for uv in corners:
        # 原始角点 -> 往中心缩 1/8 -> 往中心缩 1/4
        cands = [
            uv,
            move_uv_towards_center(uv, center, 1.0 / 8.0),
            move_uv_towards_center(uv, center, 1.0 / 4.0),
        ]
        candidate_uvs.append(cands)
    return candidate_uvs


def crop_dsm_dom_point(
    dsm_path: str,
    dom_array: np.ndarray,
    dsm_array: np.ndarray,
    dsm_transform: Affine,
    dsm_nodata,
    K: np.ndarray,
    pose_w2c: np.ndarray,
    image_width: int,
    image_height: int,
    output_dir: str,
    name: str,
    crop_padding: int = 0,
    num_samples: int = 12000,
    t_near: float = 1.0,
    t_far: float = 20000.0,
    geotransform=None,
    min_valid_dsm_height=None,
    corner_candidate_uvs=None,
):

    t0_prep = time.perf_counter()

    if geotransform is None:
        geotransform = affine_to_gdal_tuple(dsm_transform)

    if min_valid_dsm_height is None:
        min_valid_dsm_height = compute_min_valid_dsm_height(dsm_array, dsm_nodata)

    if corner_candidate_uvs is None:
        corner_candidate_uvs = build_corner_candidate_uvs(image_width, image_height)

    # print(f"[TIMING] crop_dsm_dom_point prepare: {time.perf_counter() - t0_prep:.3f}s")
    # t0_prep = time.perf_counter()
    # geotransform = affine_to_gdal_tuple(dsm_transform)
    # min_valid_dsm_height = compute_min_valid_dsm_height(dsm_array, dsm_nodata)
    # corner_candidate_uvs = build_corner_candidate_uvs(image_width, image_height)
    # print(f"[TIMING] crop_dsm_dom_point prepare: {time.perf_counter() - t0_prep:.3f}s")

    # print("\n[DEBUG][crop_dsm_dom_point]")
    # print("image_width,height =", image_width, image_height)
    # print("corner_candidate_uvs =", corner_candidate_uvs)

    world_points_ecef = []
    hit_lonlat = []
    dsm_indices = []
    used_uvs = []

    t0_ray = time.perf_counter()
    for i, cand_list in enumerate(corner_candidate_uvs):
        success = False
        last_err = None

        for j, uv in enumerate(cand_list):
            try:
                # 前两个候选只允许真实 DSM 交点
                hit_ecef, hit_llh, hit_rc = ray_cast_to_wgs84_dsm(
                    dsm_array=dsm_array,
                    geotransform=geotransform,
                    K=K,
                    pose_w2c=pose_w2c,
                    uv=uv,
                    num_samples=num_samples,
                    t_near=t_near,
                    t_far=t_far,
                    nodata=dsm_nodata,
                    fallback_height=None,
                )
                # print(f"[DEBUG] corner {i}, candidate {j}: uv={uv}, hit_llh={hit_llh}, hit_rc={hit_rc}")
                world_points_ecef.append(hit_ecef)
                hit_lonlat.append(hit_llh)
                dsm_indices.append(hit_rc)
                used_uvs.append(uv)
                success = True
                break
            except Exception as e:
                # print(f"[WARN] corner {i}, candidate {j} failed: uv={uv}, err={e}")
                last_err = e

        if success:
            continue

        # 三个候选都打不到时，用最小有效 DSM 高度兜底
        uv = cand_list[-1]
        hit_ecef, hit_llh, hit_rc = ray_cast_to_wgs84_dsm(
            dsm_array=dsm_array,
            geotransform=geotransform,
            K=K,
            pose_w2c=pose_w2c,
            uv=uv,
            num_samples=num_samples,
            t_near=t_near,
            t_far=t_far,
            nodata=dsm_nodata,
            fallback_height=min_valid_dsm_height,
        )
        # print(f"[WARN] corner {i}: all candidates failed, use fallback height. uv={uv}, hit_llh={hit_llh}, hit_rc={hit_rc}")
        world_points_ecef.append(hit_ecef)
        hit_lonlat.append(hit_llh)
        dsm_indices.append(hit_rc)
        used_uvs.append(uv)

    # print(f"[TIMING] crop_dsm_dom_point ray_cast (4 corners): {time.perf_counter() - t0_ray:.3f}s")
    # save_debug_overlay 已注释以节省耗时 (~2s)
    # t0_overlay = time.perf_counter()
    # os.makedirs(output_dir, exist_ok=True)
    # overlay_path = os.path.join(output_dir, f"{name}_debug_overlay.png")
    # save_crop_debug_overlay(dom_array, dsm_indices, overlay_path)
    # print(f"[TIMING] crop_dsm_dom_point save_debug_overlay: {time.perf_counter() - t0_overlay:.3f}s")
    overlay_path = None

    t0_warp = time.perf_counter()
    rows, cols = zip(*dsm_indices)
    h, w = dsm_array.shape

    row_min = max(min(rows) - int(crop_padding), 0)
    row_max = min(max(rows) + int(crop_padding) + 1, h)
    col_min = max(min(cols) - int(crop_padding), 0)
    col_max = min(max(cols) + int(crop_padding) + 1, w)

    if row_max <= row_min or col_max <= col_min:
        raise RuntimeError("Computed crop is empty.")

    dsm_crop = dsm_array[row_min:row_max, col_min:col_max]
    dom_crop = dom_array[row_min:row_max, col_min:col_max]

    xs = np.arange(col_min, col_max, dtype=np.float64) * dsm_transform.a + dsm_transform.c
    ys = np.arange(row_min, row_max, dtype=np.float64) * dsm_transform.e + dsm_transform.f
    lon_grid, lat_grid = np.meshgrid(xs, ys)

    dsm_poly_local = np.array([[c - col_min, r - row_min] for r, c in dsm_indices], dtype=np.float32)

    width_top = np.linalg.norm(dsm_poly_local[1] - dsm_poly_local[0])
    width_bottom = np.linalg.norm(dsm_poly_local[2] - dsm_poly_local[3])
    height_left = np.linalg.norm(dsm_poly_local[3] - dsm_poly_local[0])
    height_right = np.linalg.norm(dsm_poly_local[2] - dsm_poly_local[1])

    out_w = max(int(round(max(width_top, width_bottom))), 1)
    out_h = max(int(round(max(height_left, height_right))), 1)

    dst_corners = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)
    H = cv2.getPerspectiveTransform(dsm_poly_local, dst_corners)

    dom_warp = cv2.warpPerspective(_prepare_dom_for_warp(dom_crop), H, (out_w, out_h), flags=cv2.INTER_LINEAR)
    h_warp = cv2.warpPerspective(dsm_crop.astype(np.float32), H, (out_w, out_h), flags=cv2.INTER_LINEAR)
    lon_warp = cv2.warpPerspective(lon_grid.astype(np.float32), H, (out_w, out_h), flags=cv2.INTER_LINEAR)
    lat_warp = cv2.warpPerspective(lat_grid.astype(np.float32), H, (out_w, out_h), flags=cv2.INTER_LINEAR)

    X_warp, Y_warp, Z_warp = wgs84_to_ecef(lon_warp, lat_warp, h_warp)
    xyz_ecef = np.stack([X_warp, Y_warp, Z_warp], axis=-1).astype(np.float32)
    # xyz_ecef = np.nan_to_num(xyz_ecef, nan=0.0, posinf=0.0, neginf=0.0)
    # print(f"[TIMING] crop_dsm_dom_point warp+wgs84: {time.perf_counter() - t0_warp:.3f}s")

    return {
        "dom_warp": dom_warp,
        "xyz_ecef": xyz_ecef,
        "dsm_indices": np.array(dsm_indices, dtype=np.int32),
        "world_points_ecef": np.array(world_points_ecef, dtype=np.float64),
        "hit_lonlat": np.array(hit_lonlat, dtype=np.float64),
        "used_uvs": np.array(used_uvs, dtype=np.float64),
        "crop_bbox": (row_min, row_max, col_min, col_max),
        "homography": H,
        "debug_overlay_path": overlay_path,
    }


def save_outputs(dom_warp: np.ndarray, xyz_ecef: np.ndarray, output_dir: str, name: str):
    os.makedirs(output_dir, exist_ok=True)
    dom_path = os.path.join(output_dir, f"{name}_dom.png")
    dsm_path = os.path.join(output_dir, f"{name}_dsm.npy")
    Image.fromarray(dom_warp).save(dom_path)
    np.save(dsm_path, xyz_ecef)
    return dom_path, dsm_path


def generate_ref_map(
    dsm_path: str,
    dom_path: str,
    pose_data,
    K: np.ndarray,
    image_width: int,
    image_height: int,
    output_dir: str,
    name: str = "crop",
    crop_padding: int = 0,
    num_samples: int = 12000,
    t_near: float = 1.0,
    t_far: float = 20000.0,
):
    t0_total = time.perf_counter()
    K = normalize_K(K)

    t0 = time.perf_counter()
    dsm_array, dsm_transform, _, dsm_nodata, dom_array, _, _ = read_dsm_dom(dsm_path, dom_path)
    # print(f"[TIMING] read_dsm_dom: {time.perf_counter() - t0:.3f}s")
    t0 = time.perf_counter()
    pose_w2c, K, _ = transform_colmap_pose_intrinsic(pose_data, K)
    # print(f"[TIMING] transform_colmap_pose_intrinsic: {time.perf_counter() - t0:.3f}s")
    t0 = time.perf_counter()
    result = crop_dsm_dom_point(
        dsm_path=dsm_path,
        dom_array=dom_array,
        dsm_array=dsm_array,
        dsm_transform=dsm_transform,
        dsm_nodata=dsm_nodata,
        K=K,
        pose_w2c=pose_w2c,
        image_width=image_width,
        image_height=image_height,
        output_dir=output_dir,
        name=name,
        crop_padding=crop_padding,
        num_samples=num_samples,
        t_near=t_near,
        t_far=t_far,
    )
    # print(f"[TIMING] crop_dsm_dom_point total: {time.perf_counter() - t0:.3f}s")
    t0 = time.perf_counter()
    dom_out, dsm_out = save_outputs(result["dom_warp"], result["xyz_ecef"], output_dir, name)
    # print(f"[TIMING] save_outputs: {time.perf_counter() - t0:.3f}s")
    result["dom_path"] = dom_out
    result["dsm_path"] = dsm_out
    # print(f"[TIMING] generate_ref_map 总耗时: {time.perf_counter() - t0_total:.3f}s")
    return result