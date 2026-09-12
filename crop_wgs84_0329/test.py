import argparse
import json
import os
import sys
import time

import numpy as np

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from crop import normalize_K, run_test_0310_style_pipeline


def load_k_and_size(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError("K_json must be a dict with keys: K, image_width, image_height")

    K = normalize_K(np.array(obj["K"], dtype=np.float64))
    image_width = int(obj["image_width"])
    image_height = int(obj["image_height"])
    return K, image_width, image_height


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsm_path", required=True)
    parser.add_argument("--dom_path", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--name", default="1")
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--alt", type=float, required=True)
    parser.add_argument("--roll", type=float, required=True)
    parser.add_argument("--pitch", type=float, required=True)
    parser.add_argument("--yaw", type=float, required=True)
    parser.add_argument("--K_json", default=os.path.join(CUR_DIR, "K.json"))
    parser.add_argument("--crop_padding", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=12000)
    parser.add_argument("--t_near", type=float, default=1.0)
    parser.add_argument("--t_far", type=float, default=20000.0)
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--save_debug_vis", action="store_true")
    parser.add_argument("--save_full_dom_points", action="store_true")
    parser.add_argument("--print_corner_intrinsics", action="store_true")
    args = parser.parse_args()

    K, image_width, image_height = load_k_and_size(args.K_json)
    pose_data = [args.lon, args.lat, args.alt, args.roll, args.pitch, args.yaw]

    t0 = time.perf_counter()
    result = run_test_0310_style_pipeline(
        dsm_path=args.dsm_path,
        dom_path=args.dom_path,
        pose_data=pose_data,
        K=K,
        image_width=image_width,
        image_height=image_height,
        output_dir=args.output_dir,
        name=args.name,
        crop_padding=args.crop_padding,
        num_samples=args.num_samples,
        t_near=args.t_near,
        t_far=args.t_far,
        save_outputs=args.save_outputs,
        save_debug_vis=args.save_debug_vis,
        save_full_dom_points=args.save_full_dom_points,
        print_corner_intrinsics=args.print_corner_intrinsics,
    )
    print(f"\n[TIMING] crop_wgs84/test.py 总耗时: {time.perf_counter() - t0:.3f}s")
    print("bbox_img shape =", result["bbox_img"].shape)
    print("bbox_points shape =", result["bbox_points"].shape)
    print("bbox_mask shape =", result["bbox_mask"].shape)
    print("final_img shape =", result["final_img"].shape)
    print("final_points shape =", result["final_points"].shape)
    print("final_mask shape =", result["final_mask"].shape)
    print("saved_paths =", result.get("saved_paths"))


if __name__ == "__main__":
    main()
