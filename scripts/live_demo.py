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

# Hold last confirmed render for N frames after detection loss (jitter mitigation).
GHOST_MAX_AGE = 20
# Bbox smoothing: smoothed = α * current + (1 - α) * previous.
BBOX_EMA_ALPHA = 0.7

# Class-specific confidence overrides. Higher = suppress (less false positives);
# lower = encourage (catch more weak detections). Classes not listed use args.conf.
PER_CLASS_CONF = {
    "pass_right": 0.60,      # suppress pass_straight (not in taxonomy) misreads
    "priority_road": 0.30,   # known weak class — lower bar
}


class TrackHistory:
    """Rolling per-track classification buffer with EMA smoothing and ghost memory.

    For each persistent track id we keep the last `window` raw class predictions
    and return the majority class only once it appears at least `min_hits` times.
    We also retain the last confirmed render (`last_seen`) so callers can re-draw
    a fading box for a short while after detection loss, and EMA-smoothed bbox
    coordinates (`last_bbox`) to suppress per-frame twitching.
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
        self.last_bbox: dict[int, tuple[float, float, float, float]] = {}
        self.last_seen: dict[int, tuple[int, tuple[int, int, int, int], int, float, str]] = {}

    def update(self, track_id: int, cls: int):
        buf = self.tracks.get(track_id)
        if buf is None:
            buf = deque(maxlen=self.window)
            self.tracks[track_id] = buf
        buf.append(cls)
        top_cls, top_count = Counter(buf).most_common(1)[0]
        return top_cls if top_count >= self.min_hits else None

    def smooth_bbox(self, track_id: int, bbox: tuple[float, float, float, float]):
        prev = self.last_bbox.get(track_id)
        if prev is None:
            smoothed = bbox
        else:
            α = BBOX_EMA_ALPHA
            smoothed = tuple(α * c + (1.0 - α) * p for c, p in zip(bbox, prev))
        self.last_bbox[track_id] = smoothed
        return smoothed

    def remember(self, track_id: int, frame_idx: int,
                 bbox: tuple[int, int, int, int], cls: int, conf: float, label: str) -> None:
        self.last_seen[track_id] = (frame_idx, bbox, cls, conf, label)

    def ghost_tracks(self, frame_idx: int, max_age: int = GHOST_MAX_AGE):
        """Yield (track_id, age, bbox, cls, conf, label) for tracks last seen 1..max_age frames ago."""
        stale = []
        for tid, (last_idx, bbox, cls, conf, label) in self.last_seen.items():
            age = frame_idx - last_idx
            if age <= 0:
                continue
            if age > max_age:
                stale.append(tid)
                continue
            yield tid, age, bbox, cls, conf, label
        for tid in stale:
            del self.last_seen[tid]

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


class DetectionLog:
    """Rolling log of recently confirmed sign classes for the on-screen sidebar.

    Deduplicates by class name within a `gap_frames` window so a sign seen
    repeatedly doesn't flood the list. Entries persist after the sign leaves
    the frame — a readability backup when labels flicker too fast to read.
    """

    def __init__(self, maxlen: int = 5, gap_frames: int = 30) -> None:
        self.entries: deque = deque(maxlen=maxlen)
        self.gap_frames = gap_frames

    def add(self, cls_name: str, dist_m: float | None, spd_kmh: float | None, frame_idx: int) -> None:
        for existing in self.entries:
            if existing[0] == cls_name and frame_idx - existing[3] < self.gap_frames:
                return
        self.entries.append((cls_name, dist_m, spd_kmh, frame_idx))

    def render(self, frame, anchor_x: int, anchor_y: int) -> None:
        if not self.entries:
            return
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        line_h = 20
        title = "Recent:"
        lines = [title]
        for cls_name, dist_m, spd_kmh, _ in reversed(self.entries):
            parts = [cls_name]
            if dist_m is not None:
                parts.append(f"{round(dist_m)}m")
            if spd_kmh is not None:
                parts.append(f"{round(spd_kmh)}km/h")
            lines.append(" ".join(parts))
        widths = [cv2.getTextSize(l, font, scale, thickness)[0][0] for l in lines]
        max_w = max(widths)
        box_x1 = anchor_x - max_w - 8
        box_y1 = anchor_y - 4
        box_x2 = anchor_x + 4
        box_y2 = anchor_y + line_h * len(lines) + 4
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), TEXT_BG, -1)
        cv2.putText(frame, lines[0], (box_x1 + 4, anchor_y + line_h - 4),
                    font, scale, FPS_COLOR, thickness, cv2.LINE_AA)
        for i, line in enumerate(lines[1:], start=1):
            cv2.putText(frame, line, (box_x1 + 4, anchor_y + line_h * (i + 1) - 4),
                        font, scale, TEXT_COLOR, thickness, cv2.LINE_AA)


def estimate_distance_m(bbox_height_px: int, focal_px: float, real_height_m: float) -> float:
    if bbox_height_px <= 0:
        return 0.0
    return (real_height_m * focal_px) / bbox_height_px


def draw_box(frame, x1: int, y1: int, x2: int, y2: int, label: str, alpha: float = 1.0) -> None:
    box_color = tuple(int(c * alpha) for c in BOX_COLOR)
    text_color = tuple(int(c * alpha) for c in TEXT_COLOR)
    text_bg = tuple(int(c * alpha) for c in TEXT_BG)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    lines = label.split("\n")
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    max_w = max(w for w, _ in sizes)
    line_h = max(h for _, h in sizes) + 4
    total_h = line_h * len(lines)

    cv2.rectangle(frame, (x1, y1 - total_h), (x1 + max_w + 4, y1), text_bg, -1)
    for i, line in enumerate(lines):
        y = y1 - total_h + line_h * (i + 1) - 4
        cv2.putText(frame, line, (x1 + 2, y), font, scale, text_color, thickness, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live traffic-sign detection demo")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Auto-exit after N seconds (useful for benchmarking).")
    parser.add_argument("--target-fps", type=float, default=-1.0,
                        help="Cap playback at this FPS. -1 (default) auto-matches source video FPS; "
                             "0 disables the cap entirely.")
    parser.add_argument("--no-opengl", action="store_true",
                        help="Disable OpenGL window backend (fallback to CPU rendering).")
    parser.add_argument("--track-window", type=int, default=7,
                        help="Rolling window of recent frames per track.")
    parser.add_argument("--track-min-hits", type=int, default=4,
                        help="Min same-class hits within window to confirm a detection.")
    parser.add_argument("--min-bbox-px", type=int, default=18,
                        help="Reject detections whose shorter side is below N px (noise zone).")
    parser.add_argument("--max-aspect-ratio", type=float, default=1.8,
                        help="Reject detections with aspect ratio above N "
                             "(trained signs are ≈ square; reklame are wide rectangles).")
    parser.add_argument("--iou", type=float, default=0.50,
                        help="NMS IoU threshold (lower lets nearby different-class boxes coexist).")
    parser.add_argument("--no-sidebar", action="store_true",
                        help="Hide the recent-detections sidebar (top-right).")
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
    parser.add_argument("--speed-scale", type=float, default=1.1,
                        help="Multiplier applied to displayed km/h. Default 1.1 compensates for "
                             "the slight underestimation from a one-size-fits-all sign-height "
                             "assumption (0.60 m) vs real sign mix.")
    parser.add_argument("--no-speed", action="store_true",
                        help="Hide the approach-speed (km/h) estimate.")
    parser.add_argument("--roi-bottom-margin", type=float, default=0.30,
                        help="Reject detections whose center falls in the bottom N fraction "
                             "of the frame (road/hood region). 0 disables.")
    parser.add_argument("--roi-side-margin", type=float, default=0.0,
                        help="Reject detections whose center is within N fraction of the left "
                             "or right edge (extreme periphery). 0 disables.")
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

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if args.target_fps < 0:
        # Auto: match source so playback runs at real-time speed regardless of monitor.
        effective_target_fps = source_fps if source_fps and source_fps > 0 else 30.0
        print(f"Auto target FPS: {effective_target_fps:.1f} (source video)")
    else:
        effective_target_fps = args.target_fps

    window = "RiRV — traffic signs (q to quit)"
    window_flags = cv2.WINDOW_NORMAL
    if not args.no_opengl:
        # OpenGL backend offloads scaling/blit to GPU — eliminates the ~15ms
        # fullscreen upscale tax that otherwise drops FPS below source. Falls
        # back silently if the OpenCV build wasn't compiled with OpenGL.
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_OPENGL)
        except cv2.error:
            cv2.namedWindow(window, window_flags)
            print("OpenGL window not supported in this OpenCV build — using CPU render.")
    else:
        cv2.namedWindow(window, window_flags)

    # Warm up CUDA kernels so the first real frames hit the FPS target immediately.
    ok, warmup_frame = cap.read()
    if ok:
        for _ in range(5):
            model.predict(warmup_frame, imgsz=args.imgsz, conf=args.conf,
                          iou=args.iou, device=0, half=True, verbose=False)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    use_tracker = not args.no_tracking
    history = TrackHistory(args.track_window, args.track_min_hits,
                           speed_window=args.speed_window)
    detection_log = DetectionLog()
    times = deque(maxlen=30)
    start = time.perf_counter()
    last = start
    frames = 0
    frame_idx = 0
    frames_skipped = 0
    playback_start: float | None = None
    playback_frame_anchor = 0
    playback_speed = 1.0
    paused = False
    SPEED_STEP = 1.25
    SPEED_MIN = 0.25
    SPEED_MAX = 4.0
    print("Controls: [space] pause/resume  [+/-] speed  [r] reset speed  [q] quit")

    def reset_playback_clock(at_frame: int) -> tuple[float, int]:
        return time.perf_counter(), at_frame

    while True:
        loop_start = time.perf_counter()
        if args.max_seconds is not None and (loop_start - start) >= args.max_seconds:
            break

        # Real-time sync: drop source frames if we're behind playback clock.
        # Keeps playback at chosen speed even when render can't hit it
        # (e.g. fullscreen on a slow display path). Uses cheap cap.grab()
        # which advances the stream without decoding.
        if playback_start is not None and effective_target_fps > 0:
            elapsed = loop_start - playback_start
            expected_idx = playback_frame_anchor + int(elapsed * effective_target_fps * playback_speed)
            while frame_idx < expected_idx:
                if not cap.grab():
                    break
                frame_idx += 1
                frames_skipped += 1

        ok, frame = cap.read()
        if not ok:
            break
        if playback_start is None:
            playback_start, playback_frame_anchor = reset_playback_clock(frame_idx)

        if use_tracker:
            results = model.track(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
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
                iou=args.iou,
                device=0,
                half=True,
                verbose=False,
            )

        frame_h, frame_w = frame.shape[:2]
        roi_y_max = frame_h * (1.0 - args.roi_bottom_margin)
        roi_x_min = frame_w * args.roi_side_margin
        roi_x_max = frame_w * (1.0 - args.roi_side_margin)

        seen_track_ids: set[int] = set()
        for box in results[0].boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if cy >= roi_y_max or cx < roi_x_min or cx >= roi_x_max:
                continue
            bw, bh = x2 - x1, y2 - y1
            if min(bw, bh) < args.min_bbox_px:
                continue
            long_side, short_side = max(bw, bh), max(1, min(bw, bh))
            if long_side / short_side > args.max_aspect_ratio:
                continue
            conf = float(box.conf[0])
            raw_cls = int(box.cls[0])
            cls_name = names[raw_cls]
            per_class_threshold = PER_CLASS_CONF.get(cls_name)
            if per_class_threshold is not None and conf < per_class_threshold:
                continue
            if use_tracker:
                if box.id is None:
                    continue
                track_id = int(box.id[0])
                confirmed = history.update(track_id, raw_cls)
                if confirmed is None:
                    continue
                cls = confirmed
                sx1, sy1, sx2, sy2 = (int(round(v)) for v in
                                       history.smooth_bbox(track_id, (x1, y1, x2, y2)))
                seen_track_ids.add(track_id)
            else:
                track_id = None
                cls = raw_cls
                sx1, sy1, sx2, sy2 = x1, y1, x2, y2

            dist_m_val = None
            spd_val = None
            extras = []
            if not args.no_distance:
                dist_m = estimate_distance_m(sy2 - sy1, args.focal_length, args.sign_height)
                if dist_m > 0:
                    dist_m_val = dist_m
                    extras.append(f"dist: {round(dist_m)}m")
                    if track_id is not None and not args.no_speed:
                        history.update_distance(track_id, loop_start, dist_m)
                        spd = history.speed_kmh(track_id)
                        if spd is not None:
                            spd *= args.speed_scale
                            spd_val = spd
                            extras.append(f"spd: {round(spd)} km/h")

            label = f"{names[cls]} {conf:.2f}"
            if extras:
                label += "\n" + " ".join(extras)
            draw_box(frame, sx1, sy1, sx2, sy2, label)

            if track_id is not None:
                history.remember(track_id, frame_idx, (sx1, sy1, sx2, sy2), cls, conf, label)
            detection_log.add(names[cls], dist_m_val, spd_val, frame_idx)

        # Ghost render: tracks confirmed earlier but not seen this frame fade out.
        if use_tracker:
            for tid, age, bbox, cls, conf, label in history.ghost_tracks(frame_idx):
                if tid in seen_track_ids:
                    continue
                fade = 1.0 - (age - 1) / max(1, GHOST_MAX_AGE) * 0.7
                draw_box(frame, *bbox, label, alpha=max(0.3, fade))

        now = time.perf_counter()
        times.append(now - last)
        last = now
        fps = len(times) / sum(times) if times else 0.0

        cv2.putText(frame, f"{fps:5.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, FPS_COLOR, 2, cv2.LINE_AA)

        if playback_start is not None and effective_target_fps > 0:
            wall_elapsed = time.perf_counter() - playback_start
            frames_played = max(1, frame_idx - playback_frame_anchor)
            playback_ratio = (frames_played / (effective_target_fps * playback_speed)) / max(1e-6, wall_elapsed)
            cv2.putText(frame, f"playback: {playback_ratio:4.2f}x", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, FPS_COLOR, 2, cv2.LINE_AA)

        if abs(playback_speed - 1.0) > 0.01:
            cv2.putText(frame, f"speed: {playback_speed:.2f}x", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

        if not args.no_sidebar:
            detection_log.render(frame, anchor_x=frame_w - 10, anchor_y=10)

        cv2.imshow(window, frame)
        frames += 1
        frame_idx += 1

        if playback_start is not None and effective_target_fps > 0:
            # Sleep only if we're AHEAD of the playback clock. Subtract a small
            # constant for cv2.waitKey(1) overhead (~3 ms on Qt/X11).
            frames_since_anchor = frame_idx - playback_frame_anchor
            next_frame_time = playback_start + frames_since_anchor / (effective_target_fps * playback_speed)
            to_sleep = next_frame_time - time.perf_counter() - 0.003
            if to_sleep > 0:
                time.sleep(to_sleep)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = True
            paused_frame = frame.copy()
            cv2.putText(paused_frame, "PAUSED", (frame.shape[1] // 2 - 110, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.imshow(window, paused_frame)
            quit_during_pause = False
            while paused:
                k = cv2.waitKey(50) & 0xFF
                if k == ord(" "):
                    paused = False
                    playback_start, playback_frame_anchor = reset_playback_clock(frame_idx)
                elif k == ord("q"):
                    quit_during_pause = True
                    paused = False
            if quit_during_pause:
                break
        elif key in (ord("+"), ord("=")):
            playback_speed = min(SPEED_MAX, playback_speed * SPEED_STEP)
            playback_start, playback_frame_anchor = reset_playback_clock(frame_idx)
        elif key in (ord("-"), ord("_")):
            playback_speed = max(SPEED_MIN, playback_speed / SPEED_STEP)
            playback_start, playback_frame_anchor = reset_playback_clock(frame_idx)
        elif key == ord("r"):
            playback_speed = 1.0
            playback_start, playback_frame_anchor = reset_playback_clock(frame_idx)

    elapsed = time.perf_counter() - start
    cap.release()
    cv2.destroyAllWindows()

    if frames > 0 and elapsed > 0:
        skip_pct = (frames_skipped / max(1, frames + frames_skipped)) * 100
        print(f"Processed {frames} frames in {elapsed:.2f}s — avg {frames / elapsed:.1f} FPS "
              f"(skipped {frames_skipped} source frames, {skip_pct:.1f}%)")


if __name__ == "__main__":
    main()
