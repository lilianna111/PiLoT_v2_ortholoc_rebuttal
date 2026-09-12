import os
import sys

import numpy as np

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from crop import run_test_0310_style_pipeline


# 1. 基础配置
query_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/images/interval1_HKairport01/interval1"
ref_save_dir = "/home/amax/Documents/datasets/crop/DJI_20251217153719_0002_V"
ref_DOM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3_ortho_merge.tif"
ref_DSM_path = "/media/amax/AE0E2AFD0E2ABE69/datasets/DSM/0.3/0.3/DSM0.3_DSM_merge.tif"

# pose_data: [lon, lat, alt, roll, pitch, yaw]
pose_data = [112.999054, 28.290497, 157.509, 0.0, 37.1, -42.4]
name = "1"

# 2. 相机内参
K = np.array([
    [1350.0, 0.0, 957.85],
    [0.0, 1350.0, 537.55],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
image_width = 1920
image_height = 1080
SAVE_OUTPUTS = False
SAVE_DEBUG_VIS = True


def main() -> None:
    if query_path:
        print(f"query_path = {query_path}")
    print("开始执行 WGS84 裁剪流程...")

    result = run_test_0310_style_pipeline(
        dsm_path=ref_DSM_path,
        dom_path=ref_DOM_path,
        pose_data=pose_data,
        K=K,
        image_width=image_width,
        image_height=image_height,
        output_dir=ref_save_dir,
        name=name,
        save_outputs=SAVE_OUTPUTS,
        save_debug_vis=SAVE_DEBUG_VIS,
    )

    print("✅ 成功生成 WGS84 对齐数据：")
    print("  bbox_img:", result["bbox_img"].shape)
    print("  bbox_points:", result["bbox_points"].shape)
    print("  bbox_mask:", result["bbox_mask"].shape)
    print("  final_img:", result["final_img"].shape)
    print("  final_points:", result["final_points"].shape)
    print("  final_mask:", result["final_mask"].shape)
    if result.get("saved_paths"):
        for key, path in result["saved_paths"].items():
            print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
