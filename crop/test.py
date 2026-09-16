import argparse
import json
import os
import sys
import time

import numpy as np

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from crop import generate_ref_map, normalize_K


def load_k_and_size(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        K = np.array(obj["K"], dtype=np.float64)
        image_width = int(obj["image_width"])
        image_height = int(obj["image_height"])
    else:
        K = np.array(obj, dtype=np.float64)
        raise ValueError("K_json must be a dict with keys: K, image_width, image_height")

    K = normalize_K(K)
    return K, image_width, image_height


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsm_path", required=True)
    parser.add_argument("--dom_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", default="sample")
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
    args = parser.parse_args()

    K, image_width, image_height = load_k_and_size(args.K_json)
    pose_data = [args.lon, args.lat, args.alt, args.roll, args.pitch, args.yaw]

    t0 = time.perf_counter()
    result = generate_ref_map(
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
    )
    print(f"\n[TIMING] test.py 总耗时: {time.perf_counter() - t0:.3f}s")

    # print("\n[DONE]")
    # print("dom_path =", result["dom_path"])
    # print("dsm_path =", result["dsm_path"])
    # print("debug_overlay_path =", result.get("debug_overlay_path"))
    # print("hit_lonlat =\n", result["hit_lonlat"])
    # print("dsm_indices =\n", result["dsm_indices"])


if __name__ == "__main__":
    main()
