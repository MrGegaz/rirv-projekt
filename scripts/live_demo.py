#!/usr/bin/env python3
"""Live OpenCV demo for traffic-sign detection.

Reads a video, runs YOLO inference (FP16 on CUDA), draws boxes and a rolling
FPS counter, displays in an OpenCV window. Quit with 'q'.
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "trained" / "weights" / "best.pt"
DEFAULT_VIDEO = REPO_ROOT / "data" / "raw" / "day_video" / "DayDrive1.mp4"

BOX_COLOR = (0, 220, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 120, 0)
FPS_COLOR = (0, 255, 255)


def draw_detections(frame, boxes, names) -> None:
    for box in boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{names[cls_id]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), TEXT_BG, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live traffic-sign detection demo")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Auto-exit after N seconds (useful for benchmarking).")
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}. Run scripts/train.py first.")
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    model = YOLO(str(args.weights))
    names = model.names

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    window = "RiRV — traffic signs (q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    times = deque(maxlen=30)
    start = time.perf_counter()
    last = start
    frames = 0

    while True:
        if args.max_seconds is not None and (time.perf_counter() - start) >= args.max_seconds:
            break
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            device=0,
            half=True,
            verbose=False,
        )
        draw_detections(frame, results[0].boxes, names)

        now = time.perf_counter()
        times.append(now - last)
        last = now
        fps = len(times) / sum(times) if times else 0.0

        cv2.putText(frame, f"{fps:5.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, FPS_COLOR, 2, cv2.LINE_AA)

        cv2.imshow(window, frame)
        frames += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    elapsed = time.perf_counter() - start
    cap.release()
    cv2.destroyAllWindows()

    if frames > 0 and elapsed > 0:
        print(f"Processed {frames} frames in {elapsed:.2f}s — avg {frames / elapsed:.1f} FPS")


if __name__ == "__main__":
    main()
