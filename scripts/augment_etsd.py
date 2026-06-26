#!/usr/bin/env python3
"""Generate synthetic YOLO detection samples from ETSD classification crops.

For each chosen ETSD class, paste a random crop onto a random background at
a random scale/position with light augmentation, then emit YOLO bounding-box
annotations. Output goes to data/external/etsd_synthetic/{train,val}/{images,labels}.

This bridges ETSD's classification format to our detection pipeline without
the model overfitting to "sign always fills frame".
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# ETSD class id (Vienna Convention numbering) → our new YOLO class id (13-19).
ETSD_TO_NEW_CLS = {
    35: 13,  # Give way
    37: 14,  # Priority road
    41: 15,  # No entry
    51: 16,  # No left turn
    52: 17,  # No right turn
    16: 18,  # Pedestrian crossing warning
    87: 19,  # Pass right (D1a)
}
NEW_CLASS_NAMES = {
    13: "give_way",
    14: "priority_road",
    15: "no_entry",
    16: "no_left_turn",
    17: "no_right_turn",
    18: "pedestrian_crossing",
    19: "pass_right",
}


def gather_backgrounds(pool_dir: Path, max_count: int, rng: random.Random) -> list[Path]:
    """Pick training images that have empty/missing label files (true backgrounds)."""
    if not pool_dir.exists():
        return []
    labels_dir = pool_dir.parent / "labels"
    out: list[Path] = []
    for img in pool_dir.glob("*.jpg"):
        lbl = labels_dir / f"{img.stem}.txt"
        if not lbl.exists() or lbl.stat().st_size == 0:
            out.append(img)
    if len(out) > max_count:
        out = rng.sample(out, max_count)
    return out


def procedural_background(rng: random.Random, size: int) -> np.ndarray:
    """Synthesize a sky/asphalt gradient with mild noise — a passable driving bg."""
    top = np.array([rng.randint(120, 220), rng.randint(120, 200), rng.randint(110, 200)])
    bottom = np.array([rng.randint(40, 110), rng.randint(40, 110), rng.randint(40, 100)])
    bg = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        t = y / size
        bg[y, :] = (top * (1 - t) + bottom * t).astype(np.uint8)
    noise = (np.random.random((size, size, 1)) * 20).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return bg


def random_background(rng: random.Random, size: int, real_pool: list[Path]) -> np.ndarray:
    if real_pool and rng.random() < 0.7:
        src = rng.choice(real_pool)
        img = cv2.imread(str(src))
        if img is not None:
            h, w = img.shape[:2]
            if h > size and w > size:
                x = rng.randint(0, w - size)
                y = rng.randint(0, h - size)
                return img[y:y + size, x:x + size].copy()
            return cv2.resize(img, (size, size))
    return procedural_background(rng, size)


def augment_sign(rng: random.Random, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]

    angle = rng.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if rng.random() < 0.7:
        max_shift = max(1, int(min(w, h) * 0.15))
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [rng.randint(0, max_shift), rng.randint(0, max_shift)],
            [w - rng.randint(0, max_shift), rng.randint(0, max_shift)],
            [w - rng.randint(0, max_shift), h - rng.randint(0, max_shift)],
            [rng.randint(0, max_shift), h - rng.randint(0, max_shift)],
        ])
        M_persp = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M_persp, (w, h), borderMode=cv2.BORDER_REPLICATE)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.randint(-8, 8)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.7, 1.3) + rng.randint(-25, 25), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if rng.random() < 0.4:
        ksize = rng.choice([5, 7, 9, 11])
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        rad = np.deg2rad(rng.uniform(0, 180))
        dx, dy = np.cos(rad), np.sin(rad)
        center = ksize // 2
        for i in range(ksize):
            offset = i - center
            x = int(round(center + offset * dx))
            y = int(round(center + offset * dy))
            if 0 <= x < ksize and 0 <= y < ksize:
                kernel[y, x] = 1.0
        kernel /= max(1.0, kernel.sum())
        img = cv2.filter2D(img, -1, kernel)

    return img


def alpha_paste(bg: np.ndarray, fg: np.ndarray, x: int, y: int) -> np.ndarray:
    """Paste fg onto bg at (x, y) with a small gaussian-feathered edge."""
    fh, fw = fg.shape[:2]
    mask = np.ones((fh, fw), dtype=np.float32)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)[..., None]
    bg_h, bg_w = bg.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + fw, bg_w), min(y + fh, bg_h)
    if x2 <= x1 or y2 <= y1:
        return bg
    fx1, fy1 = x1 - x, y1 - y
    fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)
    bg_roi = bg[y1:y2, x1:x2].astype(np.float32)
    fg_roi = fg[fy1:fy2, fx1:fx2].astype(np.float32)
    m_roi = mask[fy1:fy2, fx1:fx2]
    bg[y1:y2, x1:x2] = ((1 - m_roi) * bg_roi + m_roi * fg_roi).astype(np.uint8)
    return bg


def generate_sample(rng: random.Random, sign_path: Path,
                    backgrounds: list[Path], canvas_size: int):
    sign = cv2.imread(str(sign_path))
    if sign is None:
        return None, None
    target_w = rng.randint(20, 180)
    h, w = sign.shape[:2]
    scale = target_w / w
    target_h = max(1, int(h * scale))
    sign = cv2.resize(sign, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    sign = augment_sign(rng, sign)

    bg = random_background(rng, canvas_size, backgrounds)
    max_x = canvas_size - target_w
    max_y = int(canvas_size * 0.65) - target_h   # bias to upper 2/3
    if max_x < 1 or max_y < 1:
        return None, None
    x = rng.randint(0, max_x)
    y = rng.randint(0, max_y)
    img = alpha_paste(bg, sign, x, y)

    if rng.random() < 0.4:
        quality = rng.randint(50, 90)
        ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    xc = (x + target_w / 2) / canvas_size
    yc = (y + target_h / 2) / canvas_size
    bw = target_w / canvas_size
    bh = target_h / canvas_size
    return img, (xc, yc, bw, bh)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic detection samples from ETSD classifier crops")
    p.add_argument("--etsd-root", type=Path,
                   default=REPO_ROOT / "data" / "external" / "European Traffic Sign Dataset")
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "data" / "external" / "etsd_synthetic")
    p.add_argument("--background-pool", type=Path,
                   default=REPO_ROOT / "data" / "external" / "car_no_lights" / "train" / "images")
    p.add_argument("--samples-per-class", type=int, default=500)
    p.add_argument("--val-samples-per-class", type=int, default=80)
    p.add_argument("--background-only-frac", type=float, default=0.15,
                   help="Fraction of total samples to add as pure-background hard negatives.")
    p.add_argument("--canvas-size", type=int, default=640)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--montage", action="store_true", help="Save a 12-image sanity montage")
    args = p.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    for split in ("train", "val"):
        (args.output / split / "images").mkdir(parents=True, exist_ok=True)
        (args.output / split / "labels").mkdir(parents=True, exist_ok=True)

    backgrounds = gather_backgrounds(args.background_pool, 200, rng)
    print(f"Background pool: {len(backgrounds)} real bg images (+ procedural fallback)")

    montage_samples: list[np.ndarray] = []

    for etsd_cls, new_cls in ETSD_TO_NEW_CLS.items():
        folder = args.etsd_root / "Training" / f"{etsd_cls:03d}"
        sources = sorted(folder.glob("*.ppm")) if folder.exists() else []
        if not sources:
            print(f"  WARN: no images for ETSD class {etsd_cls:03d}")
            continue

        name = NEW_CLASS_NAMES[new_cls]
        for split, n in (("train", args.samples_per_class), ("val", args.val_samples_per_class)):
            generated = 0
            for i in range(n):
                src = rng.choice(sources)
                img, bbox = generate_sample(rng, src, backgrounds, args.canvas_size)
                if img is None:
                    continue
                stem = f"etsd_{etsd_cls:03d}_{split}_{i:04d}"
                img_path = args.output / split / "images" / f"{stem}.jpg"
                lbl_path = args.output / split / "labels" / f"{stem}.txt"
                cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                lbl_path.write_text(
                    f"{new_cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                )
                generated += 1

                if args.montage and split == "train" and len(montage_samples) < 12 and i % 80 == 0:
                    annotated = img.copy()
                    h, w = annotated.shape[:2]
                    x1 = int((bbox[0] - bbox[2] / 2) * w)
                    y1 = int((bbox[1] - bbox[3] / 2) * h)
                    x2 = int((bbox[0] + bbox[2] / 2) * w)
                    y2 = int((bbox[1] + bbox[3] / 2) * h)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
                    cv2.putText(annotated, name, (x1, max(y1 - 5, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)
                    montage_samples.append(annotated)

            print(f"  ETSD {etsd_cls:03d} -> {new_cls} ({name}, {split}): {generated} samples")

    # Hard-negative background samples: pure backgrounds with empty labels.
    # Teaches the detector that random street scenery (billboards, storefronts)
    # is not a sign even if it has sign-like colors.
    total_signs = args.samples_per_class * len(ETSD_TO_NEW_CLS)
    for split, frac in (("train", args.background_only_frac), ("val", args.background_only_frac * 0.5)):
        n = int(total_signs * frac)
        for i in range(n):
            bg = random_background(rng, args.canvas_size, backgrounds)
            stem = f"bg_only_{split}_{i:05d}"
            cv2.imwrite(str(args.output / split / "images" / f"{stem}.jpg"),
                        bg, [cv2.IMWRITE_JPEG_QUALITY, 90])
            (args.output / split / "labels" / f"{stem}.txt").write_text("")
        print(f"  Hard negatives ({split}): {n} background-only samples")

    if args.montage and montage_samples:
        cs = args.canvas_size
        gap = 10
        canvas = np.full((3 * cs + 4 * gap, 4 * cs + 5 * gap, 3), 30, dtype=np.uint8)
        for i, img in enumerate(montage_samples[:12]):
            r, c = divmod(i, 4)
            x = gap + c * (cs + gap)
            y = gap + r * (cs + gap)
            canvas[y:y + cs, x:x + cs] = img
        out_path = args.output / "synthetic_montage.jpg"
        cv2.imwrite(str(out_path), canvas)
        print(f"Sanity montage: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
