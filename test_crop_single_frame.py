#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from pixloc.crop.proj2map import geo_coords_to_dsm_index
from pixloc.crop.ray_casting import TargetLocation
from pixloc.crop.transform_colmap import transform_colmap_pose_intrinsic
from pixloc.crop.utils import read_DSM_config


DEFAULT_DOM = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3_ortho_merge.tif"
DEFAULT_DSM = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3_DSM_merge.tif"
DEFAULT_OUT = "/media/amax/AE0E2AFD0E2ABE69/outputs_ortholoc/eloftr/crop_test_single"
DEFAULT_POSE = [112.99082, 28.292335, 67.383, 0.0, 68.1, -135.4]


def save_image_rgb(path, image):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] == 3:
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(path), image)


def draw_polygon_preview(dom_data, dsm_indices, out_path, max_side=2400):
    dom_vis = np.transpose(dom_data, (1, 2, 0)).copy()
    h, w = dom_vis.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1.0:
        dom_vis = cv2.resize(
            dom_vis,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    pts = np.asarray([[c * scale, r * scale] for r, c in dsm_indices], dtype=np.int32)
    cv2.polylines(dom_vis, [pts.reshape((-1, 1, 2))], True, (255, 0, 0), 3)
    for idx, pt in enumerate(pts):
        cv2.circle(dom_vis, tuple(pt), 5, (255, 255, 0), -1)
        cv2.putText(dom_vis, str(idx), tuple(pt + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    save_image_rgb(out_path, dom_vis)


def build_crop_debug(
    dsm_path,
    pose_data,
    ref_npy_path,
    map_data_pack,
    crop_padding=2,
    num_sample=4000,
):
    geotransform, area, area_min_z, dsm_data, dsm_transform, dom_data = map_data_pack
    pose_w2c, k_w2c, _, _ = transform_colmap_pose_intrinsic(pose_data)
    width, height = k_w2c[0, 2] * 2, k_w2c[1, 2] * 2
    image_points = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]

    locator = TargetLocation({"ray_casting": {}}, use_dsm=False)
    t0 = time.perf_counter()
    world_points = locator.predict_points_alt(
        dsm_path,
        pose_data,
        ref_npy_path,
        geotransform,
        k_w2c,
        area,
        area_min_z,
        num_sample=num_sample,
        object_pixel_coords_list=image_points,
    )
    raycast_s = time.perf_counter() - t0

    dsm_indices = [
        geo_coords_to_dsm_index(float(xyz[0]), float(xyz[1]), dsm_transform)
        for xyz in world_points
    ]

    rows, cols = zip(*dsm_indices)
    dsm_h, dsm_w = dsm_data.shape
    row_min = max(min(rows) - crop_padding, 0)
    row_max = min(max(rows) + crop_padding, dsm_h)
    col_min = max(min(cols) - crop_padding, 0)
    col_max = min(max(cols) + crop_padding, dsm_w)

    dsm_poly_local = np.array(
        [[c - col_min, r - row_min] for r, c in dsm_indices], dtype=np.float32
    )
    width_top = float(np.linalg.norm(dsm_poly_local[1] - dsm_poly_local[0]))
    width_bottom = float(np.linalg.norm(dsm_poly_local[2] - dsm_poly_local[3]))
    height_left = float(np.linalg.norm(dsm_poly_local[3] - dsm_poly_local[0]))
    height_right = float(np.linalg.norm(dsm_poly_local[2] - dsm_poly_local[1]))
    out_w = max(int(round(max(width_top, width_bottom))), 1)
    out_h = max(int(round(max(height_left, height_right))), 1)

    return {
        "pose_data": [float(x) for x in pose_data],
        "K": np.asarray(k_w2c, dtype=float).tolist(),
        "image_points": [[float(x), float(y)] for x, y in image_points],
        "world_points": np.asarray(world_points, dtype=float).tolist(),
        "dsm_indices_row_col": [[int(r), int(c)] for r, c in dsm_indices],
        "dsm_shape": [int(dsm_h), int(dsm_w)],
        "crop_bounds": {
            "row_min": int(row_min),
            "row_max": int(row_max),
            "col_min": int(col_min),
            "col_max": int(col_max),
        },
        "dsm_poly_local": dsm_poly_local.tolist(),
        "edge_lengths": {
            "width_top": width_top,
            "width_bottom": width_bottom,
            "height_left": height_left,
            "height_right": height_right,
        },
        "expected_output_size": {"width": int(out_w), "height": int(out_h)},
        "raycast_s": float(raycast_s),
    }


def save_scaled_crop_preview(map_data_pack, debug, out_path, max_side=2400):
    _, _, _, dsm_data, dsm_transform, dom_data = map_data_pack
    bounds = debug["crop_bounds"]
    row_min = bounds["row_min"]
    row_max = bounds["row_max"]
    col_min = bounds["col_min"]
    col_max = bounds["col_max"]
    dom_crop = dom_data[:, row_min:row_max, col_min:col_max]
    if dom_crop.size == 0:
        raise ValueError("empty dom crop")

    out_w = debug["expected_output_size"]["width"]
    out_h = debug["expected_output_size"]["height"]
    scale = min(1.0, float(max_side) / max(out_w, out_h))
    preview_w = max(1, int(round(out_w * scale)))
    preview_h = max(1, int(round(out_h * scale)))
    src = np.asarray(debug["dsm_poly_local"], dtype=np.float32)
    dst = np.array(
        [[0, 0], [preview_w - 1, 0], [preview_w - 1, preview_h - 1], [0, preview_h - 1]],
        dtype=np.float32,
    )
    h_mat = cv2.getPerspectiveTransform(src, dst)
    dom_hwc = np.transpose(dom_crop, (1, 2, 0))
    preview = cv2.warpPerspective(dom_hwc, h_mat, (preview_w, preview_h), flags=cv2.INTER_LINEAR)
    save_image_rgb(out_path, preview)
    return {"preview_width": preview_w, "preview_height": preview_h, "scale": scale}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dom", default=DEFAULT_DOM)
    parser.add_argument("--dsm", default=DEFAULT_DSM)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--pose", nargs=6, type=float, default=DEFAULT_POSE)
    parser.add_argument("--preview-max-side", type=int, default=2400)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_npy_path = os.path.splitext(args.dsm)[0] + ".npy"
    print(f"pose lon lat alt roll pitch yaw: {args.pose}", flush=True)
    print(f"loading map: {args.dsm}", flush=True)
    t0 = time.perf_counter()
    map_data_pack = read_DSM_config(args.dsm, args.dom, ref_npy_path)
    print(f"map loaded: {time.perf_counter() - t0:.3f}s", flush=True)

    debug = build_crop_debug(args.dsm, args.pose, ref_npy_path, map_data_pack)
    preview_info = save_scaled_crop_preview(
        map_data_pack,
        debug,
        out_dir / "crop_preview.png",
        max_side=args.preview_max_side,
    )
    draw_polygon_preview(
        map_data_pack[5],
        debug["dsm_indices_row_col"],
        out_dir / "polygon_on_dom.png",
        max_side=args.preview_max_side,
    )
    debug["preview"] = preview_info

    with open(out_dir / "crop_debug.json", "w", encoding="utf-8") as f:
        json.dump(debug, f, indent=2)

    print("expected_output_size:", debug["expected_output_size"], flush=True)
    print("crop_bounds:", debug["crop_bounds"], flush=True)
    print("preview:", preview_info, flush=True)
    print(f"saved: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
