#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform
except ImportError:  # pragma: no cover
    rasterio = None
    CRS = None
    transform = None


COORD_TRANSFORM = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert poses.json to txt: image_name lon lat alt roll pitch yaw"
    )
    parser.add_argument("--input", required=True, help="输入 poses.json")
    parser.add_argument("--output", default=None, help="输出 txt；默认与 input 同目录同名")
    parser.add_argument("--ref_tif", default=None, help="可选 GeoTIFF，用于读取 CRS")
    parser.add_argument(
        "--pitch_mode",
        choices=["nadir_zero", "down_positive", "dji"],
        default="nadir_zero",
        help="默认按 json_to_utm.py 的定义输出；down_positive/dji 仅对 pitch 做附加偏移",
    )
    parser.add_argument("--lonlat_precision", type=int, default=8)
    parser.add_argument("--metric_precision", type=int, default=3)
    return parser.parse_args()


def resolve_crs(data: dict, ref_tif: str | None):
    tif_path = ref_tif or data.get("dom_path") or data.get("dsm_path")
    if tif_path is None or rasterio is None:
        return None
    with rasterio.open(os.path.expanduser(tif_path)) as ds:
        return ds.crs


def world_xy_to_wgs84(x: float, y: float, crs) -> tuple[float, float]:
    if crs is None or CRS is None or transform is None:
        return x, y
    target = CRS.from_epsg(4326)
    if crs == target:
        return x, y
    lon, lat = transform(crs, target, [x], [y])
    return float(lon[0]), float(lat[0])


def pose_w2c_to_pose_c2w(pose_w2c: np.ndarray) -> np.ndarray:
    pose_w2c = np.asarray(pose_w2c, dtype=np.float64)
    if pose_w2c.shape == (4, 4):
        pose_w2c = pose_w2c[:3, :]
    if pose_w2c.shape != (3, 4):
        raise ValueError(f"pose_w2c must be 3x4 or 4x4, got {pose_w2c.shape}")
    rotation = pose_w2c[:3, :3]
    translation = pose_w2c[:3, 3]
    rotation_inv = rotation.T
    translation_inv = -rotation_inv @ translation
    pose_c2w = np.eye(4, dtype=np.float64)
    pose_c2w[:3, :3] = rotation_inv
    pose_c2w[:3, 3] = translation_inv
    return pose_c2w


def validate_pose_w2c(pose_w2c: np.ndarray) -> bool:
    pose_w2c = np.asarray(pose_w2c, dtype=np.float64)
    if pose_w2c.shape == (4, 4):
        pose_w2c = pose_w2c[:3, :]
    if pose_w2c.shape != (3, 4):
        return False
    if not np.isfinite(pose_w2c).all():
        return False
    rotation = pose_w2c[:3, :3]
    det = np.linalg.det(rotation)
    return np.isfinite(det) and abs(det) > 1e-8


def pose_c2w_to_rpy(pose_c2w: np.ndarray, pitch_mode: str) -> tuple[float, float, float]:
    pose_c2w_transformed = pose_c2w @ COORD_TRANSFORM
    rotation_c2w = pose_c2w_transformed[:3, :3]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        euler_xyz = Rotation.from_matrix(rotation_c2w).as_euler("xyz", degrees=True)

    # Match json_to_utm.py output order:
    # write ... roll pitch yaw as [euler_y, euler_x, euler_z]
    roll = float(euler_xyz[1])
    pitch = float(euler_xyz[0])
    yaw = float(euler_xyz[2])

    if pitch_mode == "down_positive":
        pitch = pitch + 90.0
    elif pitch_mode == "dji":
        pitch = pitch - 90.0
    return roll, pitch, yaw


def image_name_from_entry(entry: dict) -> str:
    image_path = entry.get("image_path")
    if image_path:
        return os.path.basename(image_path)
    sample_id = entry.get("sample_id")
    if sample_id:
        return str(sample_id)
    raise KeyError("Entry must contain image_path or sample_id")


def main() -> int:
    args = parse_args()
    input_path = os.path.expanduser(args.input)
    output_path = os.path.expanduser(args.output) if args.output else os.path.splitext(input_path)[0] + ".txt"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    poses = data.get("poses")
    if not isinstance(poses, list):
        raise ValueError(f"{input_path} does not contain a 'poses' list")

    crs = resolve_crs(data, args.ref_tif)
    lines: list[str] = []
    invalid_lines: list[str] = []

    for entry in poses:
        image_name = image_name_from_entry(entry)
        pose_w2c = np.asarray(entry.get("pose_w2c"), dtype=np.float64)
        if not validate_pose_w2c(pose_w2c):
            invalid_lines.append(image_name)
            continue
        pose_c2w = pose_w2c_to_pose_c2w(pose_w2c)
        x, y, z = pose_c2w[:3, 3]
        lon, lat = world_xy_to_wgs84(float(x), float(y), crs)
        roll, pitch, yaw = pose_c2w_to_rpy(pose_c2w, pitch_mode=args.pitch_mode)
        lines.append(
            f"{image_name} "
            f"{lon:.{args.lonlat_precision}f} "
            f"{lat:.{args.lonlat_precision}f} "
            f"{float(z):.{args.metric_precision}f} "
            f"{roll:.{args.metric_precision}f} "
            f"{pitch:.{args.metric_precision}f} "
            f"{yaw:.{args.metric_precision}f}"
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    if invalid_lines:
        invalid_output = os.path.splitext(output_path)[0] + "_invalid.txt"
        with open(invalid_output, "w", encoding="utf-8") as f:
            f.write("\n".join(invalid_lines) + "\n")

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
