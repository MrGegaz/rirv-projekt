#!/usr/bin/env python3
"""Live OpenCV demo for traffic-sign detection.

Reads a video, runs YOLO inference (FP16 on CUDA) with ByteTrack temporal
smoothing, draws stabilized class boxes, a per-sign distance estimate, and a
rolling FPS counter in an OpenCV window. Quit with 'q'.
"""

import argparse
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "trained" / "weights" / "best.pt"
DEFAULT_VIDEO = REPO_ROOT / "data" / "raw" / "day_video" / "DayDrive1.mp4"

BOX_COLOR = (0, 220, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 120, 0)
FPS_COLOR = (0, 255, 255)

# Pinhole-model defaults for distance estimation. Adjust via CLI for other
# cameras: focal_px ≈ (img_width / 2) / tan(FOV / 2), e.g. 1280-wide @ 60° → 1109.
DEFAULT_SIGN_HEIGHT_M = 0.60
DEFAULT_FOCAL_LENGTH_PX = 600

# Approach-speed estimation: rolling window of (t, dist) samples per track,
# slope of linear regression → m/s → km/h. Wider window = smoother but laggier.
DEFAULT_SPEED_WINDOW = 12
DEFAULT_MIN_SPEED_SAMPLES = 5


class TrackHistory:
    """Rolling per-track classification buffer.

    For each persistent track id we keep the last `window` raw class predictions
    and return the majority class only once it appears at least `min_hits` times.
    This suppresses single-frame flickers (e.g. STOP misread as speed_limit at
    distance) and transient side false-positives.
    """

    def __init__(self, window: int, min_hits: int,
                 speed_window: int = DEFAULT_SPEED_WINDOW,
                 min_speed_samples: int = DEFAULT_MIN_SPEED_SAMPLES) -> None:
        self.window = window
        self.min_hits = min_hits
        self.speed_window = speed_window
        self.min_speed_samples = min_speed_samples
        self.tracks: dict[int, deque] = {}
        self.dist_samples: dict[int, deque] = {}

    def update(self, track_id: int, cls: int):
        buf = self.tracks.get(track_id)
        if buf is None:
            buf = deque(maxlen=self.window)
            self.tracks[track_id] = buf
        buf.append(cls)
        top_cls, top_count = Counter(buf).most_common(1)[0]
        return top_cls if top_count >= self.min_hits else None

    def update_distance(self, track_id: int, t: float, dist_m: float) -> None:
        buf = self.dist_samples.get(track_id)
        if buf is None:
            buf = deque(maxlen=self.speed_window)
            self.dist_samples[track_id] = buf
        buf.append((t, dist_m))

    def speed_kmh(self, track_id: int):
        buf = self.dist_samples.get(track_id)
        if buf is None or len(buf) < self.min_speed_samples:
            return None
        ts = np.fromiter((t for t, _ in buf), dtype=float, count=len(buf))
        ds = np.fromiter((d for _, d in buf), dtype=float, count=len(buf))
        slope_mps = np.polyfit(ts, ds, 1)[0]   # negative slope = approaching
        approach_mps = -slope_mps
        if approach_mps <= 0:
            return None
        return approach_mps * 3.6


def estimate_distance_m(bbox_height_px: int, focal_px: float, real_height_m: float) -> float:
    if bbox_height_px <= 0:
        return 0.0
    return (real_height_m * focal_px) / bbox_height_px


def draw_box(frame, x1: int, y1: int, x2: int, y2: int, label: str) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

    lines = label.split("\n")
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    max_w = max(w for w, _ in sizes)
    line_h = max(h for _, h in sizes) + 4
    total_h = line_h * len(lines)

    cv2.rectangle(frame, (x1, y1 - total_h), (x1 + max_w + 4, y1), TEXT_BG, -1)
    for i, line in enumerate(lines):
        y = y1 - total_h + line_h * (i + 1) - 4
        cv2.putText(frame, line, (x1 + 2, y), font, scale, TEXT_COLOR, thickness, cv2.LINE_AA)


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
    parser.add_argument("--sign-height", type=float, default=DEFAULT_SIGN_HEIGHT_M,
                        help="Assumed physical sign height in meters for distance estimate.")
    parser.add_argument("--focal-length", type=float, default=DEFAULT_FOCAL_LENGTH_PX,
                        help="Camera focal length in pixels (rough default for 1280-wide dashcam).")
    parser.add_argument("--no-distance", action="store_true",
                        help="Hide the distance estimate appended to each label.")
    parser.add_argument("--speed-window", type=int, default=DEFAULT_SPEED_WINDOW,
                        help="Rolling window of recent (time, distance) samples per track for speed.")
    parser.add_argument("--no-speed", action="store_true",
                        help="Hide the approach-speed (km/h) estimate.")
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
    history = TrackHistory(args.track_window, args.track_min_hits,
                           speed_window=args.speed_window)
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
                track_id = int(box.id[0])
                confirmed = history.update(track_id, raw_cls)
                if confirmed is None:
                    continue
                cls = confirmed
            else:
                track_id = None
                cls = raw_cls

            extras = []
            if not args.no_distance:
                dist_m = estimate_distance_m(y2 - y1, args.focal_length, args.sign_height)
                if dist_m > 0:
                    extras.append(f"dist: {round(dist_m)}m")
                    if track_id is not None and not args.no_speed:
                        history.update_distance(track_id, loop_start, dist_m)
                        spd = history.speed_kmh(track_id)
                        if spd is not None:
                            extras.append(f"spd: {round(spd)} km/h")

            label = f"{names[cls]} {conf:.2f}"
            if extras:
                label += "\n" + " ".join(extras)
            draw_box(frame, x1, y1, x2, y2, label)

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
