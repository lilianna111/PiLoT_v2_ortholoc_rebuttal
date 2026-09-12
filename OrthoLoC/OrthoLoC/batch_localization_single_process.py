#!/usr/bin/env python3
"""
单进程批量定位：DOP / DSM / Matcher 只加载一次，循环处理一组查询图，
最后把所有结果汇总到一个 poses.json 中。
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import tempfile

import torch
from loguru import logger

from ortholoc import utils
from ortholoc.image_matching.MatcherIMCUI import MatcherIMCUI, MATCHER_ZOO
from ortholoc.scripts.run_localization import run_localization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img_dir", required=True, help="查询图目录")
    parser.add_argument("--img_glob", default="*.png", help="相对 img_dir 的 glob，默认 *.png")
    parser.add_argument("--dom", required=True, help="DOM / DOP GeoTIFF")
    parser.add_argument("--dsm", required=True, help="DSM GeoTIFF")
    parser.add_argument("--intrinsics", required=True, help="相机内参 json")
    parser.add_argument("--out", required=True, help="输出目录，保存 poses.json")
    parser.add_argument("--matcher", default="Mast3R", choices=list(MATCHER_ZOO.keys()))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--angles", nargs="+", type=int, default=[0], help="旋转候选角，默认 0")
    parser.add_argument("--min_conf", type=float, default=0.5)
    parser.add_argument("--reprojection_error", type=float, default=5.0)
    parser.add_argument("--pnp_mode", default="poselib", choices=["cv2", "poselib", "pycolmap"])
    parser.add_argument("--no_figures", action="store_true", help="不保存每张 query-vs-dom 匹配图")
    parser.add_argument("--figures_dir", default="figures", help="结果图目录名，默认 figures")
    parser.add_argument("--plot_max_pts", type=int, default=1000, help="每张图最多可视化多少匹配点")
    parser.add_argument("--continue_on_error", action="store_true", help="单张失败时跳过继续")
    return parser.parse_args()


def natural_sort_key(path: str) -> list[int | str]:
    stem = os.path.splitext(os.path.basename(path))[0]
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", stem)]


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA 不可用，请检查 PyTorch / 驱动，或改用 --device cpu")
        return 1

    pattern = os.path.join(os.path.expanduser(args.img_dir), args.img_glob)
    images = sorted(glob.glob(pattern), key=natural_sort_key)
    if not images:
        logger.error("未找到图像: {}", pattern)
        return 1

    out_root = os.path.expanduser(args.out)
    os.makedirs(out_root, exist_ok=True)
    poses_json_path = os.path.join(out_root, "poses.json")
    figures_root = os.path.join(out_root, args.figures_dir)
    if not args.no_figures:
        os.makedirs(figures_root, exist_ok=True)

    logger.info("加载 DOP / DSM（各一次）…")
    image_dop = utils.io.load_dop_tif(os.path.expanduser(args.dom))
    dsm = utils.io.load_dsm_tif(os.path.expanduser(args.dsm))

    logger.info("加载匹配模型 {}（一次）…", args.matcher)
    angles_f = [float(angle) for angle in args.angles]
    matcher = MatcherIMCUI(name=args.matcher, device=args.device, angles=angles_f)
    logger.info("匹配设备: {}", matcher.device)

    failed: list[str] = []
    aggregated_poses: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="ortholoc_batch_") as tmp_dir:
        for img_path in images:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            sample_tmp_dir = os.path.join(tmp_dir, stem)
            os.makedirs(sample_tmp_dir, exist_ok=True)
            logger.info("---- {} ----", img_path)
            try:
                run_localization(
                    matcher_name=args.matcher,
                    matcher=matcher,
                    cached_dop=image_dop,
                    cached_dsm=dsm,
                    img_path=os.path.expanduser(img_path),
                    dop_path=os.path.expanduser(args.dom),
                    dsm_path=os.path.expanduser(args.dsm),
                    intrinsics_path=os.path.expanduser(args.intrinsics),
                    output_dir=sample_tmp_dir,
                    device=args.device,
                    angles=angles_f,
                    min_conf=args.min_conf,
                    reprojection_error=args.reprojection_error,
                    pnp_mode=args.pnp_mode,
                    plot_max_pts=args.plot_max_pts,
                    save_figures=not args.no_figures,
                    crop_match_plot_to_dom_pts=True,
                )
                camera_params_path = os.path.join(sample_tmp_dir, "camera_params.json")
                if not os.path.exists(camera_params_path):
                    raise RuntimeError("未生成 camera_params.json")
                camera_params = utils.io.load_json(camera_params_path)
                if not args.no_figures:
                    fig_src = os.path.join(sample_tmp_dir, f"{stem}_{args.matcher}_matches.png")
                    if os.path.exists(fig_src):
                        shutil.move(fig_src, os.path.join(figures_root, os.path.basename(fig_src)))
                aggregated_poses.append({
                    "sample_id": stem,
                    "image_path": os.path.abspath(os.path.expanduser(img_path)),
                    "pose_w2c": camera_params.get("pose_w2c"),
                    "intrinsics": camera_params.get("intrinsics"),
                })
            except Exception as exc:
                if args.continue_on_error:
                    logger.exception("失败（已跳过）: {} — {}", img_path, exc)
                    failed.append(os.path.basename(img_path))
                else:
                    raise
            finally:
                shutil.rmtree(sample_tmp_dir, ignore_errors=True)

    utils.io.save_json(
        poses_json_path,
        {
            "dom_path": os.path.abspath(os.path.expanduser(args.dom)),
            "dsm_path": os.path.abspath(os.path.expanduser(args.dsm)),
            "intrinsics_path": os.path.abspath(os.path.expanduser(args.intrinsics)),
            "matcher": args.matcher,
            "device": args.device,
            "angles": angles_f,
            "poses": aggregated_poses,
            "failed_images": failed,
        },
    )
    logger.info("已写入汇总结果: {}", poses_json_path)
    if failed:
        logger.warning("共 {} 张失败: {}", len(failed), failed)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
