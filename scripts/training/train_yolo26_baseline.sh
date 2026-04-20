#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-yolo26n.pt}"
DATA="${2:-config/data_local.yaml}"
EPOCHS="${3:-80}"
IMGSZ="${4:-640}"
BATCH="${5:-8}"

# RTX 3050 6GB-friendly baseline defaults.
yolo detect train \
  model="$MODEL" \
  data="$DATA" \
  epochs="$EPOCHS" \
  imgsz="$IMGSZ" \
  batch="$BATCH" \
  device=0 \
  workers=4 \
  cos_lr=True \
  patience=20 \
  amp=True \
  project=models/checkpoints \
  name=yolo26_baseline
