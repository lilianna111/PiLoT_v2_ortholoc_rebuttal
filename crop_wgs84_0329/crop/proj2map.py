from __future__ import annotations

import os

import cv2
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import Affine

from .transform_colmap import (
    ecef_to_wgs84,
    enu_to_ecef_rotation,
    normalize_K,
    transform_colmap_pose_intrinsic,
    wgs84_to_ecef,
)


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
    if dom.shape[0] >= 3:
        dom_array = dom[:3]
    else:
        dom_array = np.repeat(dom, 3, axis=0)
    dom_array = np.ascontiguousarray(dom_array)
    return dsm_array, dsm_transform, dsm_profile, dsm_nodata, dom_array, dom_transform, dom_profile


def affine_to_gdal_tuple(transform: Affine):
    return (transform.c, transform.a, transform.b, transform.f, transform.d, transform.e)


def compute_min_valid_dsm_height(dsm_array: np.ndarray, dsm_nodata):
    valid = _valid_dsm_mask(dsm_array, dsm_nodata)
    if not np.any(valid):
        raise RuntimeError("DSM has no valid height for fallback.")
    return float(np.min(dsm_array[valid]))


def compute_area_min_z(dsm_array: np.ndarray, dsm_nodata):
    return _compute_area_min_z(dsm_array, dsm_nodata)


def image_corners(image_width: int, image_height: int):
    return np.array([
        [0, 0],
        [image_width - 1, 0],
        [image_width - 1, image_height - 1],
        [0, image_height - 1],
    ], dtype=np.float64)


def move_uv_towards_center(uv, center, ratio: float):
    uv = np.asarray(uv, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    return uv + ratio * (center - uv)


def build_corner_candidate_uvs(image_width: int, image_height: int):
    corners = image_corners(image_width, image_height)
    center = np.array([(image_width - 1) * 0.5, (image_height - 1) * 0.5], dtype=np.float64)

    candidate_uvs = []
    for uv in corners:
        candidate_uvs.append([
            uv,
            move_uv_towards_center(uv, center, 1.0 / 8.0),
            move_uv_towards_center(uv, center, 1.0 / 4.0),
        ])
    return candidate_uvs


def _prepare_dom_for_vis(dom_crop: np.ndarray) -> np.ndarray:
    if dom_crop.ndim == 3 and dom_crop.shape[0] == 3:
        img = np.transpose(dom_crop, (1, 2, 0))
    else:
        img = dom_crop

    if img.dtype == np.uint8:
        return np.ascontiguousarray(img)

    arr = img.astype(np.float32)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - mn) / (mx - mn)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _valid_dsm_mask(dsm_array: np.ndarray, nodata) -> np.ndarray:
    valid = np.isfinite(dsm_array)
    valid &= (dsm_array > -1000.0)
    valid &= (dsm_array < 10000.0)
    valid &= (dsm_array > -9990.0)
    if nodata is not None:
        valid &= ~np.isclose(dsm_array, nodata)
    return valid


def _compute_area_min_z(dsm_array: np.ndarray, nodata) -> float:
    valid = _valid_dsm_mask(dsm_array, nodata)
    if not np.any(valid):
        raise RuntimeError("DSM has no valid elevation values.")
    return float(np.median(dsm_array[valid]))


def ecef_to_lonlatalt(x, y, z):
    llh = ecef_to_wgs84(x, y, z)
    return float(llh[0]), float(llh[1]), float(llh[2])


def get_enu_to_ecef_rotation(lon_deg, lat_deg):
    return enu_to_ecef_rotation(lon_deg, lat_deg)


def get_ecef_to_enu_rotation(lon_deg, lat_deg):
    return get_enu_to_ecef_rotation(lon_deg, lat_deg).T


def ecef_points_to_local_enu_xy(points_ecef, origin_ecef=None):
    pts = np.asarray(points_ecef, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]

    if origin_ecef is None:
        origin_ecef = pts.mean(axis=0)
    origin_ecef = np.asarray(origin_ecef, dtype=np.float64)

    lon0, lat0, _ = ecef_to_lonlatalt(*origin_ecef)
    r_ecef_to_enu = get_ecef_to_enu_rotation(lon0, lat0)
    local = (pts - origin_ecef[None, :]) @ r_ecef_to_enu.T
    return local[:, :2], origin_ecef


def build_ecef_grid_from_geo(xs, ys, zz):
    xx, yy = np.meshgrid(xs, ys)
    x, y, z = wgs84_to_ecef(xx, yy, zz)
    return np.stack([x, y, z], axis=-1).astype(np.float64)


def ray_intersect_altitude_plane(ray_origin_ecef, ray_dir_ecef, target_alt, max_iter=60):
    ray_origin_ecef = np.asarray(ray_origin_ecef, dtype=np.float64)
    ray_dir_ecef = np.asarray(ray_dir_ecef, dtype=np.float64)
    ray_dir_ecef = ray_dir_ecef / np.linalg.norm(ray_dir_ecef)

    def f(t):
        p = ray_origin_ecef + t * ray_dir_ecef
        _, _, alt = ecef_to_lonlatalt(*p)
        return alt - target_alt

    t0 = 0.0
    t1 = 100.0
    f0 = f(t0)
    f1 = f(t1)

    expand_count = 0
    while f0 * f1 > 0 and expand_count < 40:
        t1 *= 2.0
        f1 = f(t1)
        expand_count += 1

    if f0 * f1 > 0:
        return None

    for _ in range(max_iter):
        tm = 0.5 * (t0 + t1)
        fm = f(tm)
        if abs(fm) < 1e-3:
            return ray_origin_ecef + tm * ray_dir_ecef
        if f0 * fm <= 0:
            t1, f1 = tm, fm
        else:
            t0, f0 = tm, fm

    return ray_origin_ecef + 0.5 * (t0 + t1) * ray_dir_ecef


def pixel_to_world(K, pose_w2c, uv, target_z):
    u, v = uv
    k_inv = np.linalg.inv(K)
    dir_cam = k_inv @ np.array([u, v, 1.0], dtype=np.float64)
    dir_cam = dir_cam / np.linalg.norm(dir_cam)

    r = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = r.T
    c2w[:3, 3] = -r.T @ t

    ray_dir = c2w[:3, :3] @ dir_cam
    ray_dir = ray_dir / np.linalg.norm(ray_dir)
    cam_center = c2w[:3, 3]

    world_pt = ray_intersect_altitude_plane(cam_center, ray_dir, target_z)
    if world_pt is None:
        raise RuntimeError(f"Failed to project pixel {uv} onto altitude={target_z}.")
    return world_pt


def pixel_to_ray_dir(K, pose_w2c, uv):
    u, v = uv
    k_inv = np.linalg.inv(K)
    dir_cam = k_inv @ np.array([u, v, 1.0], dtype=np.float64)
    dir_cam = dir_cam / np.linalg.norm(dir_cam)

    r = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = r.T
    c2w[:3, 3] = -r.T @ t

    ray_dir = c2w[:3, :3] @ dir_cam
    return ray_dir / np.linalg.norm(ray_dir)


def geo_coords_to_dsm_index(lon, lat, transform):
    col, row = ~transform * (lon, lat)
    return int(row), int(col)


def ray_intersect_dsm(ray_origin, ray_dir, dsm_data, dsm_transform, nodata, max_distance=10000.0):
    ray_origin = np.asarray(ray_origin, dtype=np.float64)
    ray_dir = np.asarray(ray_dir, dtype=np.float64)
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    step = 1.0
    num_samples = int(max_distance / step)
    best_point = None
    best_alt = -np.inf

    for i in range(num_samples):
        distance = i * step
        point_ecef = ray_origin + ray_dir * distance
        lon, lat, point_alt = ecef_to_lonlatalt(*point_ecef)
        row, col = geo_coords_to_dsm_index(lon, lat, dsm_transform)

        if 0 <= row < dsm_data.shape[0] and 0 <= col < dsm_data.shape[1]:
            dsm_z = dsm_data[row, col]
            if not _valid_dsm_mask(np.array([[dsm_z]], dtype=np.float64), nodata)[0, 0]:
                continue
            if point_alt <= dsm_z + 0.1 and dsm_z > best_alt:
                best_alt = dsm_z
                best_point = wgs84_to_ecef(lon, lat, dsm_z)

    return best_point


def get_height_from_center_pixel(K, pose_w2c, center_point, dsm_data, dsm_transform, nodata):
    r = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = r.T
    c2w[:3, 3] = -r.T @ t
    cam_center = c2w[:3, 3]

    ray_dir = pixel_to_ray_dir(K, pose_w2c, center_point)
    intersect_point = ray_intersect_dsm(cam_center, ray_dir, dsm_data, dsm_transform, nodata)

    if intersect_point is not None:
        lon, lat, height = ecef_to_lonlatalt(*intersect_point)
        print(f"center DSM hit: {height:.2f}m @ lon={lon:.8f}, lat={lat:.8f}")
        return intersect_point
    return None


def calculate_polygon_area(vertices):
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0


def clip_line_to_rect(p1, p2, rect):
    x1, y1 = p1
    x2, y2 = p2
    x_min, y_min, x_max, y_max = rect

    if (x1 < x_min and x2 < x_min) or (x1 > x_max and x2 > x_max) or \
       (y1 < y_min and y2 < y_min) or (y1 > y_max and y2 > y_max):
        return []

    if x_min <= x1 <= x_max and x_min <= x2 <= x_max and \
       y_min <= y1 <= y_max and y_min <= y2 <= y_max:
        return [(x1, y1), (x2, y2)]

    points = []
    dx = x2 - x1
    dy = y2 - y1

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        if x_min <= x1 <= x_max and y_min <= y1 <= y_max:
            return [(x1, y1)]
        return []

    if abs(dx) > 1e-9:
        for x_bound in [x_min, x_max]:
            t = (x_bound - x1) / dx
            if 0 <= t <= 1:
                y = y1 + t * dy
                if y_min <= y <= y_max:
                    points.append((x_bound, y))

    if abs(dy) > 1e-9:
        for y_bound in [y_min, y_max]:
            t = (y_bound - y1) / dy
            if 0 <= t <= 1:
                x = x1 + t * dx
                if x_min <= x <= x_max:
                    points.append((x, y_bound))

    if x_min <= x1 <= x_max and y_min <= y1 <= y_max:
        points.append((x1, y1))
    if x_min <= x2 <= x_max and y_min <= y2 <= y_max:
        points.append((x2, y2))

    if len(points) == 0:
        return []
    if len(points) == 1:
        return [points[0]]

    def dist_to_p1(p):
        return ((p[0] - x1) ** 2 + (p[1] - y1) ** 2) ** 0.5

    sorted_points = sorted(set(points), key=dist_to_p1)
    return [sorted_points[0], sorted_points[-1]] if len(sorted_points) >= 2 else sorted_points


def clip_polygon_to_rect(polygon, rect):
    if len(polygon) == 0:
        return []

    clipped_points = []
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        clipped_edge = clip_line_to_rect(p1, p2, rect)
        for p in clipped_edge:
            if len(clipped_points) == 0 or clipped_points[-1] != p:
                clipped_points.append(p)

    if len(clipped_points) > 1 and clipped_points[0] == clipped_points[-1]:
        clipped_points = clipped_points[:-1]
    return clipped_points


def calculate_polygon_rect_intersection_area(polygon, rect):
    clipped_polygon = clip_polygon_to_rect(polygon, rect)
    if len(clipped_polygon) < 3:
        return 0.0
    return calculate_polygon_area(clipped_polygon)


def create_polygon_mask(image_shape_hw, polygon_vertices):
    height, width = image_shape_hw
    mask = np.zeros((height, width), dtype=np.uint8)
    if polygon_vertices is None or len(polygon_vertices) < 3:
        mask[:, :] = 255
        return mask

    pts = np.array(polygon_vertices, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


def apply_white_mask_to_dom_crop(dom_crop, mask):
    if dom_crop.ndim == 3 and dom_crop.shape[0] == 3:
        img = np.transpose(dom_crop, (1, 2, 0)).copy()
        transposed = True
    else:
        img = dom_crop.copy()
        transposed = False

    img = np.clip(img, 0, 255).astype(np.uint8)
    img[mask == 0] = 255

    if transposed:
        return np.transpose(img, (2, 0, 1))
    return img


def find_short_parallel_edge(world_points, parallel_threshold=0.15):
    if len(world_points) < 4:
        return None

    pts_xy, origin_ecef = ecef_points_to_local_enu_xy(np.asarray(world_points, dtype=np.float64))

    edges = []
    for i in range(4):
        p1 = pts_xy[i]
        p2 = pts_xy[(i + 1) % 4]
        vec = p2 - p1
        length = np.linalg.norm(vec)
        direction = vec / length if length > 1e-12 else vec
        edges.append((vec, direction, length, i, (i + 1) % 4))

    parallel_pairs = []
    for i in range(4):
        for j in range(i + 1, 4):
            _, dir1, len1, _, _ = edges[i]
            _, dir2, len2, _, _ = edges[j]
            parallel_degree = abs(np.dot(dir1, dir2))
            if parallel_degree > (1 - parallel_threshold):
                parallel_pairs.append((parallel_degree, i, j, len1, len2))

    if not parallel_pairs:
        return None

    _, edge1_idx, edge2_idx, len1, len2 = max(parallel_pairs, key=lambda x: x[0])
    chosen = edge1_idx if len1 < len2 else edge2_idx
    _, direction, length, start_idx, end_idx = edges[chosen]
    return length, direction, start_idx, end_idx, origin_ecef


def build_rotated_rectangle_from_center_and_edge(center_world, edge_length_along_short_direction,
                                                 short_edge_direction, image_aspect_ratio):
    center_world = np.asarray(center_world, dtype=np.float64)
    center_xy, origin_ecef = ecef_points_to_local_enu_xy(center_world, origin_ecef=center_world)
    center_xy = center_xy[0]

    width_along_u = float(edge_length_along_short_direction)
    if image_aspect_ratio <= 0:
        image_aspect_ratio = 1.0
    height_along_v = width_along_u / image_aspect_ratio

    u = np.array(short_edge_direction, dtype=np.float64)
    if np.linalg.norm(u) < 1e-12:
        u = np.array([1.0, 0.0], dtype=np.float64)
    else:
        u = u / np.linalg.norm(u)

    v = np.array([-u[1], u[0]], dtype=np.float64)

    half_u = width_along_u / 2.0
    half_v = height_along_v / 2.0

    rect_enu = [
        center_xy + half_u * u + half_v * v,
        center_xy + half_u * u - half_v * v,
        center_xy - half_u * u - half_v * v,
        center_xy - half_u * u + half_v * v,
    ]

    lon0, lat0, _ = ecef_to_lonlatalt(*origin_ecef)
    r_enu_to_ecef = get_enu_to_ecef_rotation(lon0, lat0)

    rect_ecef = []
    for enu_xy in rect_enu:
        enu_vec = np.array([enu_xy[0], enu_xy[1], 0.0], dtype=np.float64)
        rect_ecef.append(origin_ecef + r_enu_to_ecef @ enu_vec)
    return rect_ecef


def save_crop_points_on_dom(dom_crop, dsm_indices, rotated_rect_vertices_relative, save_path, center_index=None):
    img = _prepare_dom_for_vis(dom_crop).copy()

    if dsm_indices is not None and len(dsm_indices) > 0:
        pts = np.array(dsm_indices, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        for idx, (x, y) in enumerate(np.array(dsm_indices, dtype=np.int32)):
            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 255), -1)
            cv2.putText(img, str(idx), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    if rotated_rect_vertices_relative is not None and len(rotated_rect_vertices_relative) > 0:
        rect_points = np.array(rotated_rect_vertices_relative, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [rect_points], isClosed=True, color=(0, 255, 0), thickness=3)
        for idx, (x, y) in enumerate(np.array(rotated_rect_vertices_relative, dtype=np.int32)):
            cv2.circle(img, (int(x), int(y)), 6, (0, 255, 0), -1)
            cv2.putText(img, f"R{idx}", (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    if center_index is not None:
        cx, cy = map(int, center_index)
        cv2.circle(img, (cx, cy), 8, (255, 0, 255), -1)
        cv2.putText(img, "C", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, "C", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    Image.fromarray(img).save(save_path)


def save_points_on_full_dom(dom_data, dsm_indices, save_path, center_index=None):
    img = _prepare_dom_for_vis(dom_data).copy()
    pts = np.array(dsm_indices, dtype=np.int32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected Nx2 points, got shape={pts.shape}")

    if len(pts) > 0:
        pts_poly = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts_poly], isClosed=True, color=(255, 0, 0), thickness=4)
        for idx, (x, y) in enumerate(pts):
            cv2.circle(img, (int(x), int(y)), 10, (0, 255, 255), -1)
            cv2.putText(img, str(idx), (int(x) + 12, int(y) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, str(idx), (int(x) + 12, int(y) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    if center_index is not None:
        cx, cy = map(int, center_index)
        cv2.circle(img, (cx, cy), 14, (255, 0, 255), -1)
        cv2.putText(img, "C", (cx + 16, cy - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, "C", (cx + 16, cy - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    Image.fromarray(img).save(save_path)


def crop_dsm_dom_angle(dom_data, dsm_data, dsm_transform, dsm_nodata,
                       K, pose_w2c, image_points, center_point, area_minZ,
                       crop_padding=0):
    world_points = []
    dsm_indices = []

    for uv in image_points:
        xyz = pixel_to_world(K, pose_w2c, uv, target_z=area_minZ)
        world_points.append(xyz)
        lon_i, lat_i, _ = ecef_to_lonlatalt(*xyz)
        dsm_indices.append(geo_coords_to_dsm_index(lon_i, lat_i, dsm_transform))

    center_xyz = pixel_to_world(K, pose_w2c, center_point, target_z=area_minZ)
    center_lon, center_lat, _ = ecef_to_lonlatalt(*center_xyz)
    center_dsm_index = geo_coords_to_dsm_index(center_lon, center_lat, dsm_transform)
    short_edge_info = find_short_parallel_edge(world_points)
    if short_edge_info is None:
        raise RuntimeError("Unable to find short parallel edge from the projected quadrilateral.")

    short_len, short_dir, _, _, _ = short_edge_info

    img_width = max(pt[0] for pt in image_points) - min(pt[0] for pt in image_points) + 1
    img_height = max(pt[1] for pt in image_points) - min(pt[1] for pt in image_points) + 1
    image_aspect_ratio = img_width / img_height if img_height > 0 else 1.0

    rect_vertices_ecef = build_rotated_rectangle_from_center_and_edge(
        center_xyz,
        edge_length_along_short_direction=short_len,
        short_edge_direction=short_dir,
        image_aspect_ratio=image_aspect_ratio,
    )

    rect_indices = []
    for vertex in rect_vertices_ecef:
        lon_v, lat_v, _ = ecef_to_lonlatalt(*vertex)
        rect_indices.append(geo_coords_to_dsm_index(lon_v, lat_v, dsm_transform))

    rows, cols = zip(*rect_indices)
    dsm_height, dsm_width = dsm_data.shape
    row_min = int(max(min(rows) - crop_padding, 0))
    row_max = int(min(max(rows) + crop_padding, dsm_height))
    col_min = int(max(min(cols) - crop_padding, 0))
    col_max = int(min(max(cols) + crop_padding, dsm_width))

    if row_max <= row_min or col_max <= col_min:
        raise RuntimeError("Computed crop is empty. Check pose, DSM coverage, or projection logic.")

    dsm_crop = dsm_data[row_min:row_max, col_min:col_max]
    dom_crop = dom_data[:, row_min:row_max, col_min:col_max]

    xs = np.arange(col_min, col_max, dtype=np.float64) * dsm_transform.a + dsm_transform.c
    ys = np.arange(row_min, row_max, dtype=np.float64) * dsm_transform.e + dsm_transform.f
    zz = dsm_crop.astype(np.float64)

    # DOM 和 DSM 在边缘处可能存在 1 像素级尺寸不一致，后续 mask/点云必须先对齐。
    common_h = min(zz.shape[0], dom_crop.shape[1], len(ys))
    common_w = min(zz.shape[1], dom_crop.shape[2], len(xs))
    if common_h <= 0 or common_w <= 0:
        raise RuntimeError("Computed crop has no overlapping DOM/DSM area after alignment.")

    if (
        zz.shape[0] != common_h or zz.shape[1] != common_w or
        dom_crop.shape[1] != common_h or dom_crop.shape[2] != common_w
    ):
        zz = zz[:common_h, :common_w]
        dsm_crop = dsm_crop[:common_h, :common_w]
        dom_crop = dom_crop[:, :common_h, :common_w]
        ys = ys[:common_h]
        xs = xs[:common_w]

    if zz.shape[0] != len(ys) or zz.shape[1] != len(xs):
        zz_fixed = np.full((len(ys), len(xs)), fill_value=area_minZ, dtype=np.float64)
        r = min(zz.shape[0], zz_fixed.shape[0])
        c = min(zz.shape[1], zz_fixed.shape[1])
        zz_fixed[:r, :c] = zz[:r, :c]
        zz = zz_fixed

    crop_offset = (row_min, col_min)
    center_dsm_index_relative = (center_dsm_index[1] - col_min, center_dsm_index[0] - row_min)
    rotated_rect_vertices_relative = []
    for row_v, col_v in rect_indices:
        rotated_rect_vertices_relative.append([col_v - col_min, row_v - row_min])

    polygon_mask = create_polygon_mask((dom_crop.shape[1], dom_crop.shape[2]), rotated_rect_vertices_relative)
    valid_dsm_mask = (_valid_dsm_mask(zz, dsm_nodata).astype(np.uint8)) * 255
    combined_mask = np.where((polygon_mask > 0) & (valid_dsm_mask > 0), 255, 0).astype(np.uint8)

    zz_for_grid = zz.copy()
    zz_for_grid[combined_mask == 0] = area_minZ
    points_3d = build_ecef_grid_from_geo(xs, ys, zz_for_grid)
    points_3d[combined_mask == 0] = 0.0

    dsm_indices_xy = [(col - col_min, row - row_min) for row, col in dsm_indices]
    trapezoid_polygon = [(col, row) for row, col in dsm_indices]
    actual_rect = (col_min, row_min, col_max, row_max)
    trapezoid_area = calculate_polygon_rect_intersection_area(trapezoid_polygon, actual_rect)
    ref_total_area = max((col_max - col_min) * (row_max - row_min), 1)
    actual_ratio_percent = (trapezoid_area / ref_total_area) * 100.0

    return {
        "dom_crop": dom_crop,
        "point_cloud_crop": points_3d.astype(np.float32),
        "dsm_indices": np.array([(col, row) for row, col in dsm_indices], dtype=np.int32),
        "dsm_indices_relative": np.array(dsm_indices_xy, dtype=np.int32),
        "world_points": np.array(world_points, dtype=np.float64),
        "center_world": np.array(center_xyz, dtype=np.float64),
        "center_dsm_index": np.array((center_dsm_index[1], center_dsm_index[0]), dtype=np.int32),
        "center_dsm_index_relative": np.array(center_dsm_index_relative, dtype=np.int32),
        "crop_offset": crop_offset,
        "rotated_rect_vertices_relative": np.array(rotated_rect_vertices_relative, dtype=np.int32),
        "crop_mask": combined_mask,
        "min_shape": min(col_max - col_min, row_max - row_min),
        "actual_ratio_percent": actual_ratio_percent,
    }


def generate_ref_map(
    dsm_path: str,
    dom_path: str,
    pose_data,
    K: np.ndarray,
    image_width: int,
    image_height: int,
    output_dir: str | None = None,
    name: str = "1",
    crop_padding: int = 0,
    num_samples: int = 12000,
    t_near: float = 1.0,
    t_far: float = 20000.0,
    save_bbox: bool = False,
    save_debug_vis: bool = False,
    save_full_dom_points: bool = False,
    print_corner_intrinsics: bool = False,
):
    del num_samples, t_near, t_far

    K = normalize_K(K)

    dsm_array, dsm_transform, _, dsm_nodata, dom_array, _, _ = read_dsm_dom(dsm_path, dom_path)
    area_minZ = _compute_area_min_z(dsm_array, dsm_nodata)
    pose_w2c, K, pose_meta = transform_colmap_pose_intrinsic(pose_data, K)

    width = int(image_width)
    height = int(image_height)
    image_points = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
    center_point = (width // 2, height // 2)

    if print_corner_intrinsics:
        print("[corner_intrinsics] K_used_for_corners =")
        print(np.array2string(K, precision=6, suppress_small=False))
        print(f"[corner_intrinsics] image_width={width}, image_height={height}")
        print(f"[corner_intrinsics] image_points={image_points}")
        print(f"[corner_intrinsics] center_point={center_point}")
        print(
            "[corner_intrinsics] pose="
            f" lon={pose_meta['lon']}, lat={pose_meta['lat']}, alt={pose_meta['alt']},"
            f" roll={pose_meta['roll']}, pitch={pose_meta['pitch']}, yaw={pose_meta['yaw']}"
        )

    result = crop_dsm_dom_angle(
        dom_array,
        dsm_array,
        dsm_transform,
        dsm_nodata,
        K,
        pose_w2c,
        image_points,
        center_point,
        area_minZ,
        crop_padding=crop_padding,
    )

    ratio_img_path = None
    ratio_npy_path = None
    ratio_mask_path = None
    overlay_path = None
    full_dom_points_path = None

    if output_dir is not None and (save_bbox or save_debug_vis or save_full_dom_points):
        os.makedirs(output_dir, exist_ok=True)

    dom_crop_masked = apply_white_mask_to_dom_crop(result["dom_crop"], result["crop_mask"])
    ratio_img_uint8 = np.clip(np.transpose(dom_crop_masked, (1, 2, 0)), 0, 255).astype(np.uint8)

    if output_dir is not None and save_bbox:
        ratio_img_path = os.path.join(output_dir, f"{name}_bbox_dom.jpg")
        ratio_npy_path = os.path.join(output_dir, f"{name}_bbox_dom.npy")
        ratio_mask_path = os.path.join(output_dir, f"{name}_bbox_dom_mask.png")
        Image.fromarray(ratio_img_uint8).save(ratio_img_path)
        np.save(ratio_npy_path, result["point_cloud_crop"])
        cv2.imwrite(ratio_mask_path, result["crop_mask"])

    if output_dir is not None and save_debug_vis:
        overlay_path = os.path.join(output_dir, f"{name}_crop_points_on_dom.jpg")
        save_crop_points_on_dom(
            result["dom_crop"],
            result["dsm_indices_relative"],
            result["rotated_rect_vertices_relative"],
            overlay_path,
            center_index=result["center_dsm_index_relative"],
        )

    if output_dir is not None and save_full_dom_points:
        full_dom_points_path = os.path.join(output_dir, f"{name}_points_on_full_dom.jpg")
        save_points_on_full_dom(
            dom_array,
            result["dsm_indices"],
            full_dom_points_path,
            center_index=result["center_dsm_index"],
        )

    result["bbox_dom_rgb"] = ratio_img_uint8
    result["bbox_dom_path"] = ratio_img_path
    result["bbox_npy_path"] = ratio_npy_path
    result["bbox_mask_path"] = ratio_mask_path
    result["debug_overlay_path"] = overlay_path
    result["full_dom_points_path"] = full_dom_points_path
    result["area_minZ"] = area_minZ
    result["pose_w2c"] = pose_w2c
    result["pose_meta"] = pose_meta
    return result


def crop_dsm_dom_point(
    dsm_path: str,
    dom_array: np.ndarray,
    dsm_array: np.ndarray,
    dsm_transform,
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
    area_min_z=None,
    pose_data=None,
    pose_meta=None,
    dom_path: str | None = None,
):
    del dsm_path
    del output_dir
    del geotransform
    del min_valid_dsm_height
    del corner_candidate_uvs
    del dom_path
    del num_samples
    del t_near
    del t_far

    from .pipeline import rotate_all_with_mask

    if pose_meta is None and pose_data is not None:
        _, _, pose_meta = transform_colmap_pose_intrinsic(pose_data, K)

    area_minZ = float(area_min_z) if area_min_z is not None else _compute_area_min_z(dsm_array, dsm_nodata)
    width = int(image_width)
    height = int(image_height)
    image_points = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
    center_point = (width // 2, height // 2)

    result = crop_dsm_dom_angle(
        dom_array,
        dsm_array,
        dsm_transform,
        dsm_nodata,
        normalize_K(K),
        pose_w2c,
        image_points,
        center_point,
        area_minZ,
        crop_padding=crop_padding,
    )

    bbox_img = apply_white_mask_to_dom_crop(result["dom_crop"], result["crop_mask"])
    bbox_img = np.clip(np.transpose(bbox_img, (1, 2, 0)), 0, 255).astype(np.uint8)
    bbox_points = result["point_cloud_crop"].astype(np.float32)
    bbox_mask = result["crop_mask"]

    yaw_angle = 0.0
    if pose_meta is not None:
        yaw_angle = float(pose_meta["yaw_processed"])

    final_img, final_npy, final_mask = rotate_all_with_mask(
        bbox_img,
        bbox_points,
        yaw_angle,
        mask=bbox_mask,
        crop_padding=0,
    )
    if final_img is None or final_npy is None or final_mask is None:
        final_img = bbox_img
        final_npy = bbox_points
        final_mask = bbox_mask

    return {
        "dom_warp": final_img,
        "xyz_ecef": final_npy.astype(np.float32),
        "crop_mask": final_mask,
        "bbox_dom_rgb": bbox_img,
        "bbox_points": bbox_points,
        "bbox_mask": bbox_mask,
        "pose_meta": pose_meta if pose_meta is not None else {},
        "saved_paths": {},
    }
