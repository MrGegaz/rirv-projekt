#!/usr/bin/env python3
"""Extract frames from day/dusk videos with ffmpeg.

Output structure:
  data/frames/day/<video_stem>/frame_000001.jpg
  data/frames/dusk/<video_stem>/frame_000001.jpg
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n{process.stderr.strip()}"
        )


def extract_video(video_path: Path, out_dir: Path, fps: float, overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for old in out_dir.glob("frame_*.jpg"):
            old.unlink()

    output_pattern = str(out_dir / "frame_%06d.jpg")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        output_pattern,
    ]
    run_cmd(cmd)


def collect_videos(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*.mp4") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from project videos")
    parser.add_argument("--day-dir", type=Path, default=Path("data/raw/day_video"))
    parser.add_argument("--dusk-dir", type=Path, default=Path("data/raw/dusk_video"))
    parser.add_argument("--out-root", type=Path, default=Path("data/frames"))
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not available in PATH")

    day_videos = collect_videos(args.day_dir)
    dusk_videos = collect_videos(args.dusk_dir)

    if not day_videos and not dusk_videos:
        raise RuntimeError("No videos found in day/dusk input directories")

    for video in day_videos:
        out_dir = args.out_root / "day" / video.stem
        print(f"[day] extracting {video.name} -> {out_dir}")
        extract_video(video, out_dir, args.fps, args.overwrite)

    for video in dusk_videos:
        out_dir = args.out_root / "dusk" / video.stem
        print(f"[dusk] extracting {video.name} -> {out_dir}")
        extract_video(video, out_dir, args.fps, args.overwrite)

    print("Frame extraction completed.")


if __name__ == "__main__":
    main()
