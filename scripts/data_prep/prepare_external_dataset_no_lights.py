#!/usr/bin/env python3
"""Prepare external YOLO dataset by removing traffic lights and remapping classes.

Source dataset example: data/external/car
Expected source split layout:
  train/images, train/labels
  valid/images, valid/labels
  test/images,  test/labels
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

TARGET_CLASS_ORDER = [
    "Speed Limit 10",
    "Speed Limit 20",
    "Speed Limit 30",
    "Speed Limit 40",
    "Speed Limit 50",
    "Speed Limit 60",
    "Speed Limit 70",
    "Speed Limit 80",
    "Speed Limit 90",
    "Speed Limit 100",
    "Speed Limit 110",
    "Speed Limit 120",
    "Stop",
]

TARGET_CLASS_NAMES = [
    "speed_limit_10",
    "speed_limit_20",
    "speed_limit_30",
    "speed_limit_40",
    "speed_limit_50",
    "speed_limit_60",
    "speed_limit_70",
    "speed_limit_80",
    "speed_limit_90",
    "speed_limit_100",
    "speed_limit_110",
    "speed_limit_120",
    "stop",
]


def load_source_name_to_id(source_yaml: Path) -> dict[str, int]:
    with source_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    if isinstance(names, list):
        return {name: idx for idx, name in enumerate(names)}
    if isinstance(names, dict):
        return {name: int(idx) for idx, name in names.items()}
    raise ValueError("Unsupported 'names' format in source data.yaml")


def transform_label_file(
    src_label: Path,
    dst_label: Path,
    source_name_to_id: dict[str, int],
) -> tuple[int, int]:
    # Build id remap: source_id -> target_id
    source_id_to_target_id: dict[int, int] = {}
    for target_id, source_name in enumerate(TARGET_CLASS_ORDER):
        if source_name not in source_name_to_id:
            raise ValueError(f"Missing class '{source_name}' in source dataset")
        source_id_to_target_id[source_name_to_id[source_name]] = target_id

    kept = 0
    removed = 0
    out_lines: list[str] = []

    if src_label.exists():
        lines = src_label.read_text(encoding="utf-8").strip().splitlines()
    else:
        lines = []

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        source_id = int(float(parts[0]))
        if source_id not in source_id_to_target_id:
            removed += 1
            continue

        target_id = source_id_to_target_id[source_id]
        out_lines.append(" ".join([str(target_id)] + parts[1:]))
        kept += 1

    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return kept, removed


def process_split(
    src_root: Path,
    dst_root: Path,
    split_src: str,
    split_dst: str,
    source_name_to_id: dict[str, int],
) -> tuple[int, int, int]:
    src_img_dir = src_root / split_src / "images"
    src_lbl_dir = src_root / split_src / "labels"

    dst_img_dir = dst_root / split_dst / "images"
    dst_lbl_dir = dst_root / split_dst / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in src_img_dir.glob("*.*") if p.is_file()])

    images = 0
    total_kept = 0
    total_removed = 0

    for img in image_files:
        images += 1
        dst_img = dst_img_dir / img.name
        shutil.copy2(img, dst_img)

        src_lbl = src_lbl_dir / f"{img.stem}.txt"
        dst_lbl = dst_lbl_dir / f"{img.stem}.txt"
        kept, removed = transform_label_file(src_lbl, dst_lbl, source_name_to_id)
        total_kept += kept
        total_removed += removed

    return images, total_kept, total_removed


def write_target_data_yaml(dst_root: Path) -> None:
    out = {
        "path": str(dst_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(TARGET_CLASS_NAMES)},
    }
    with (dst_root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove traffic lights and remap external dataset")
    parser.add_argument("--src", type=Path, default=Path("data/external/car"))
    parser.add_argument("--dst", type=Path, default=Path("data/external/car_no_lights"))
    parser.add_argument("--src-yaml", type=Path, default=None)
    args = parser.parse_args()

    src_yaml = args.src_yaml or (args.src / "data.yaml")
    if not src_yaml.exists():
        raise FileNotFoundError(f"Source data.yaml not found: {src_yaml}")

    source_name_to_id = load_source_name_to_id(src_yaml)

    split_map = [("train", "train"), ("valid", "val"), ("test", "test")]

    total_images = 0
    total_kept = 0
    total_removed = 0

    for split_src, split_dst in split_map:
        images, kept, removed = process_split(
            args.src,
            args.dst,
            split_src,
            split_dst,
            source_name_to_id,
        )
        print(f"[{split_src}] images={images}, kept_boxes={kept}, removed_boxes={removed}")
        total_images += images
        total_kept += kept
        total_removed += removed

    write_target_data_yaml(args.dst)

    print("Done.")
    print(f"Total images: {total_images}")
    print(f"Total kept boxes: {total_kept}")
    print(f"Total removed boxes (traffic lights / unsupported): {total_removed}")
    print(f"Output: {args.dst}")


if __name__ == "__main__":
    main()
