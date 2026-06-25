#!/usr/bin/env python3
"""Live OpenCV demo for traffic-sign detection.

Reads a video, runs YOLO inference (FP16 on CUDA) with ByteTrack temporal
smoothing, draws stabilized class boxes and a rolling FPS counter in an
OpenCV window. Quit with 'q'.
"""

import argparse
import time
from collections import Counter, deque
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


class TrackHistory:
    """Rolling per-track classification buffer.

    For each persistent track id we keep the last `window` raw class predictions
    and return the majority class only once it appears at least `min_hits` times.
    This suppresses single-frame flickers (e.g. STOP misread as speed_limit at
    distance) and transient side false-positives.
    """

    def __init__(self, window: int, min_hits: int) -> None:
        self.window = window
        self.min_hits = min_hits
        self.tracks: dict[int, deque] = {}

    def update(self, track_id: int, cls: int):
        buf = self.tracks.get(track_id)
        if buf is None:
            buf = deque(maxlen=self.window)
            self.tracks[track_id] = buf
        buf.append(cls)
        top_cls, top_count = Counter(buf).most_common(1)[0]
        return top_cls if top_count >= self.min_hits else None


def draw_box(frame, x1: int, y1: int, x2: int, y2: int, label: str) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), TEXT_BG, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live traffic-sign detection demo")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Auto-exit after N seconds (useful for benchmarking).")
    parser.add_argument("--target-fps", type=float, default=60.0,
                        help="Cap playback at this FPS. Use 0 for uncapped.")
    parser.add_argument("--track-window", type=int, default=5,
                        help="Rolling window of recent frames per track.")
    parser.add_argument("--track-min-hits", type=int, default=3,
                        help="Min same-class hits within window to confirm a detection.")
    parser.add_argument("--no-tracking", action="store_true",
                        help="Disable ByteTrack smoothing (raw per-frame predictions).")
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

    # Warm up CUDA kernels so the first real frames hit the FPS target immediately.
    ok, warmup_frame = cap.read()
    if ok:
        for _ in range(5):
            model.predict(warmup_frame, imgsz=args.imgsz, conf=args.conf,
                          device=0, half=True, verbose=False)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    use_tracker = not args.no_tracking
    history = TrackHistory(args.track_window, args.track_min_hits)
    frame_period = 1.0 / args.target_fps if args.target_fps > 0 else 0.0
    times = deque(maxlen=30)
    start = time.perf_counter()
    last = start
    frames = 0

    while True:
        loop_start = time.perf_counter()
        if args.max_seconds is not None and (loop_start - start) >= args.max_seconds:
            break
        ok, frame = cap.read()
        if not ok:
            break

        if use_tracker:
            results = model.track(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=0,
                half=True,
                verbose=False,
                persist=True,
                tracker="bytetrack.yaml",
            )
        else:
            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=0,
                half=True,
                verbose=False,
            )

        for box in results[0].boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            conf = float(box.conf[0])
            raw_cls = int(box.cls[0])
            if use_tracker:
                if box.id is None:
                    continue
                confirmed = history.update(int(box.id[0]), raw_cls)
                if confirmed is None:
                    continue
                cls = confirmed
            else:
                cls = raw_cls
            draw_box(frame, x1, y1, x2, y2, f"{names[cls]} {conf:.2f}")

        now = time.perf_counter()
        times.append(now - last)
        last = now
        fps = len(times) / sum(times) if times else 0.0

        cv2.putText(frame, f"{fps:5.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, FPS_COLOR, 2, cv2.LINE_AA)

        cv2.imshow(window, frame)
        frames += 1

        if frame_period > 0:
            # Subtract a small constant for cv2.waitKey(1) overhead (~3 ms on Qt/X11).
            remaining = frame_period - (time.perf_counter() - loop_start) - 0.003
            if remaining > 0:
                time.sleep(remaining)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    elapsed = time.perf_counter() - start
    cap.release()
    cv2.destroyAllWindows()

    if frames > 0 and elapsed > 0:
        print(f"Processed {frames} frames in {elapsed:.2f}s — avg {frames / elapsed:.1f} FPS")


if __name__ == "__main__":
    main()
