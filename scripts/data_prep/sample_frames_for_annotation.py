#!/usr/bin/env python3
"""Create a balanced first annotation batch from extracted frames."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


EXTS = {".jpg", ".jpeg", ".png"}


def list_frames(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS])


def copy_sample(files: list[Path], target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        # Keep source video name in filename to avoid collisions.
        stem = f"{src.parent.name}__{src.stem}"
        dst = target_dir / f"{stem}{src.suffix.lower()}"
        shutil.copy2(src, dst)
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample frames for first annotation batch")
    parser.add_argument("--frames-root", type=Path, default=Path("data/frames"))
    parser.add_argument("--out", type=Path, default=Path("data/annotations/cvat_exports/seed_batch"))
    parser.add_argument("--total", type=int, default=450)
    parser.add_argument("--day-ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    if args.clear and args.out.exists():
        shutil.rmtree(args.out)

    day_frames = list_frames(args.frames_root / "day")
    dusk_frames = list_frames(args.frames_root / "dusk")

    if not day_frames and not dusk_frames:
        raise RuntimeError("No extracted frames found. Run extract_frames.py first.")

    random.seed(args.seed)

    target_day = int(args.total * args.day_ratio)
    target_dusk = args.total - target_day

    day_pick = random.sample(day_frames, min(target_day, len(day_frames))) if day_frames else []
    dusk_pick = random.sample(dusk_frames, min(target_dusk, len(dusk_frames))) if dusk_frames else []

    copied_day = copy_sample(day_pick, args.out / "day")
    copied_dusk = copy_sample(dusk_pick, args.out / "dusk")

    print(f"Sample created at: {args.out}")
    print(f"day={copied_day}, dusk={copied_dusk}, total={copied_day + copied_dusk}")


if __name__ == "__main__":
    main()
