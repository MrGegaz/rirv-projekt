#!/usr/bin/env python3
"""Quick sanity check for YOLO dataset layout and label IDs."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def collect_images(img_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO dataset")
    parser.add_argument("--data", type=Path, required=True, help="Path to data.yaml")
    args = parser.parse_args()

    with args.data.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["path"])
    if not root.is_absolute():
        root = (args.data.parent / root).resolve()

    names = cfg["names"]
    if isinstance(names, dict):
        class_count = len(names)
    else:
        class_count = len(names)

    for split in ["train", "val", "test"]:
        if split not in cfg:
            continue

        img_dir = (root / cfg[split]).resolve()
        lbl_dir = Path(str(img_dir).replace("/images", "/labels"))

        images = collect_images(img_dir)
        missing = 0
        invalid_ids = 0

        for img in images:
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                missing += 1
                continue

            for line in lbl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                cls_id = int(float(line.split()[0]))
                if cls_id < 0 or cls_id >= class_count:
                    invalid_ids += 1

        print(
            f"[{split}] images={len(images)} missing_labels={missing} "
            f"invalid_class_ids={invalid_ids}"
        )


if __name__ == "__main__":
    main()
