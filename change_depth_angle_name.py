from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


# ===================== 直接运行时的“写死配置” =====================
SRC_PATH = "/home/ps/Documents/liuxy24/PiLoT_v55/poses/depth/USA_seq2@8@foggy@intensity2@500.txt"
OUT_DIR = "/mnt/data1/UserData/liuxy/Mapscape/Test/depth_pose/"
# SRC_PATH = "/home/ps/Documents/liuxy24/PiLoT_v55/poses/USA_seq2@8@foggy@intensity2@500.txt"
# OUT_DIR = "/mnt/data1/UserData/liuxy/Mapscape/Test/angle/"
TARGETS = [
    "sunny",
    "rainy",
    "foggy",
    "night@intensity",
    "foggy@intensity2",
    "foggy@intensity3",
    "foggy@intensity1",
    "night@intensity1",
    "night@intensity2",
    "cloudy",
    "night@intensity3",
    "sunny@screen16",
    "sunny@screen8",
    "sunset",
    "snowy",
]
# ============================================================


def _split_prefix_suffix(filename: str) -> tuple[str, str]:
    parts = filename.split("@")
    if len(parts) < 3:
        raise ValueError(f"文件名拆分失败: {filename!r}")
    prefix = "@".join(parts[:2])
    suffix = "@" + parts[-1]
    return prefix, suffix


def _normalize_target(target: str) -> str:
    t = target.strip()
    if not t:
        raise ValueError("target 不能为空")
    return t if t.startswith("@") else "@" + t


def batch_copy_rename(
    src_path: Path,
    targets: list[str],
    out_dir: Path | None = None,
) -> list[Path]:
    src_path = src_path.expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"源文件不存在: {src_path}")

    out_dir = (out_dir or src_path.parent).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix, suffix = _split_prefix_suffix(src_path.name)

    written: list[Path] = []
    for target in targets:
        mid = _normalize_target(target)
        dst_path = out_dir / f"{prefix}{mid}{suffix}"

        # 使用 copyfile：只拷贝字节流，不修改权限位或时间戳；目标已存在则覆盖
        shutil.copyfile(src_path, dst_path)
        written.append(dst_path)

    return written


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out-dir", default="")
    p.add_argument("--targets", nargs="+", required=True)
    return p.parse_args()


def main() -> None:
    if len(sys.argv) == 1:
        base_dir = Path(__file__).resolve().parents[1]
        src_path = Path(SRC_PATH)
        if not src_path.is_absolute():
            src_path = base_dir / src_path
        out_dir = (base_dir / OUT_DIR) if OUT_DIR else None
        targets = list(TARGETS)
    else:
        args = _parse_args()
        src_path = Path(args.src)
        out_dir = Path(args.out_dir) if args.out_dir else None
        targets = list(args.targets)

    written = batch_copy_rename(
        src_path=src_path,
        targets=targets,
        out_dir=out_dir,
    )
    for p in written:
        print(f"Created: {p}")


if __name__ == "__main__":
    main()