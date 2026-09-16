import os
from pathlib import Path

import numpy as np

# =========================
# 直接运行配置（按需修改）
# =========================
DSM_PATH = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/HKairport/dsm_wgs84.tif"
DOM_PATH = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/HKairport/dom_wgs84.tif"
OUTPUT_DIR = "/home/amax/Documents/code/PiLoT/PiLoT/crop/test"
NAME = "sample"

# query pose: lon, lat, alt, roll, pitch, yaw
LON = 114.0427091112508
LAT = 22.416065856758376
ALT = 141.50411236009015
ROLL = 4.541406000587585
PITCH = 1.5399178705785226
YAW = 103.11220858726509

# 直接把 K 写死在代码里，避免再读 json 出现尺度/格式问题
K = np.array([
    [735.5326, 0.0, 586.1788],
    [0.0, 735.5326, 523.1838],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

DEBUG = True
CROP_VIS = True
USE_BBOX = True
TEST_RATIO = None  # 只有 USE_BBOX=False 时才会用到
SAVE_TO_DISK = True  # True: 保存 jpg/npy/mask；False: 只返回内存数组

# =========================
# 兼容导入
# =========================
try:
    from proj2map_test_wgs84 import generate_ref_map
except ImportError:
    from proj2map_test import generate_ref_map

try:
    from transform_colmap_wgs84 import transform_colmap_pose_intrinsic
except ImportError:
    from transform_colmap import transform_colmap_pose_intrinsic

from utils import read_DSM_config


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_k(k: np.ndarray) -> np.ndarray:
    k = np.array(k, dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError(f"K shape must be (3,3), got {k.shape}")
    if abs(k[2, 2]) < 1e-12:
        raise ValueError(f"Invalid K[2,2]={k[2,2]}")
    k = k / k[2, 2]
    k[2, :] = [0.0, 0.0, 1.0]
    return k


def main() -> None:
    print("=" * 100)
    print("Direct crop test")
    print("=" * 100)
    print(f"DSM_PATH   : {DSM_PATH}")
    print(f"DOM_PATH   : {DOM_PATH}")
    print(f"OUTPUT_DIR : {OUTPUT_DIR}")
    print(f"NAME       : {NAME}")
    print("pose       :")
    print(f"  lon={LON}")
    print(f"  lat={LAT}")
    print(f"  alt={ALT}")
    print(f"  roll={ROLL}")
    print(f"  pitch={PITCH}")
    print(f"  yaw={YAW}")
    print(f"USE_BBOX   : {USE_BBOX}")
    print(f"TEST_RATIO : {TEST_RATIO}")
    print(f"SAVE_TO_DISK: {SAVE_TO_DISK}")
    print("=" * 100)

    if not os.path.exists(DSM_PATH):
        raise FileNotFoundError(f"DSM 不存在: {DSM_PATH}")
    if not os.path.exists(DOM_PATH):
        raise FileNotFoundError(f"DOM 不存在: {DOM_PATH}")

    ref_rgb_path = os.path.join(OUTPUT_DIR, "ref_rgb")
    ref_depth_path = os.path.join(OUTPUT_DIR, "ref_depth")
    crop_vis_save_path = os.path.join(OUTPUT_DIR, "crop_vis")
    cache_npy_path = os.path.join(OUTPUT_DIR, "dsm_cache.npy")

    ensure_dir(OUTPUT_DIR)
    ensure_dir(ref_rgb_path)
    ensure_dir(ref_depth_path)
    if CROP_VIS:
        ensure_dir(crop_vis_save_path)
    else:
        crop_vis_save_path = None

    print("\n[1/4] 读取 DSM / DOM ...")
    geotransform, area, area_minZ, dsm_data, dsm_transform, dom_data = read_DSM_config(
        DSM_PATH, DOM_PATH, cache_npy_path
    )
    print(f"DSM shape     : {dsm_data.shape}")
    print(f"DOM shape     : {dom_data.shape}")
    print(f"area_minZ     : {area_minZ}")
    print(f"GeoTransform  : {geotransform}")

    print("\n[2/4] 构造 pose_w2c / intrinsic ...")
    pose_data = [LON, LAT, ALT, ROLL, PITCH, YAW]
    pose_w2c, intrinsic_default, q_intrinsics_info, osg_dict = transform_colmap_pose_intrinsic(pose_data)
    print("pose_w2c =")
    print(pose_w2c)

    K_used = normalize_k(K)
    print("K (hard-coded) =")
    print(K_used)
    print(f"derived image size from K: {int(round(K_used[0, 2] * 2))} x {int(round(K_used[1, 2] * 2))}")
    print(f"default intrinsic from transform_colmap =\n{intrinsic_default}")

    print("\n[3/4] 生成 crop ...")
    save_rgb_path = ref_rgb_path if SAVE_TO_DISK else None
    save_depth_path = ref_depth_path if SAVE_TO_DISK else None

    data = generate_ref_map(
        name=NAME,
        query_intrinsics=intrinsic_default,
        query_poses=pose_w2c,
        area_minZ=area_minZ,
        dsm_data=dsm_data,
        dsm_transform=dsm_transform,
        dom_data=dom_data,
        ref_rgb_path=save_rgb_path,
        ref_depth_path=save_depth_path,
        debug=True,
        crop_vis_save_path=crop_vis_save_path,
        test_ratio=TEST_RATIO,
        use_bbox=USE_BBOX,
        save_to_disk=True,
    )

    dom_crop = data['dom_crop']
    point_cloud_crop = data['point_cloud_crop']
    crop_mask = data['crop_mask']

    print("\n[4/4] 内存结果检查 ...")
    print("generate_ref_map return keys =", list(data.keys()))
    print("dom_crop shape:", dom_crop.shape, "dtype:", dom_crop.dtype)
    print("point_cloud_crop shape:", point_cloud_crop.shape, "dtype:", point_cloud_crop.dtype)
    print(f"point_cloud finite_ratio: {np.isfinite(point_cloud_crop).mean():.6f}")

    if crop_mask is not None:
        print("crop_mask shape:", crop_mask.shape, "dtype:", crop_mask.dtype)
        print(f"crop_mask valid_ratio: {(crop_mask > 0).mean():.6f}")
    else:
        print("crop_mask: None")

    print("rotated_rect_vertices_relative =", data.get("rotated_rect_vertices_relative"))
    print("crop_offset =", data.get("crop_offset"))
    print("actual_ratio_percent =", data.get("actual_ratio_percent"))
    print("saved_paths =", data.get("saved_paths"))

    if SAVE_TO_DISK:
        print("\n[5/5] 磁盘输出检查 ...")
        saved_paths = data.get("saved_paths", {})
        for key in ["rgb", "depth", "mask"]:
            p = saved_paths.get(key)
            print(f"{key}: {p} -> {os.path.exists(p) if p else False}")
            if p and p.endswith(".npy") and os.path.exists(p):
                arr = np.load(p)
                print(f"  npy shape={arr.shape}, dtype={arr.dtype}, finite_ratio={np.isfinite(arr).mean():.6f}")

    print("\n完成。")


if __name__ == "__main__":
    main()
