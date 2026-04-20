#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-models/checkpoints/yolo26_baseline/weights/best.pt}"
SOURCE="${2:-data/raw/day_video/DayDrive1.mp4}"
IMGSZ="${3:-640}"
CONF="${4:-0.25}"

yolo detect predict \
  model="$MODEL" \
  source="$SOURCE" \
  imgsz="$IMGSZ" \
  conf="$CONF" \
  device=0 \
  save=True \
  save_txt=False \
  project=demo/output \
  name=prediction
