#!/usr/bin/env python3
"""Train YOLO11s on the external traffic-sign dataset (car_no_lights, 13 classes).

Output: models/trained/weights/best.pt
"""

from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_CFG = REPO_ROOT / "config" / "data.yaml"


def main() -> None:
    model = YOLO("yolo11s.pt")
    model.train(
        data=str(DATA_CFG),
        epochs=80,
        imgsz=640,
        batch=16,
        device=0,
        amp=True,
        cos_lr=True,
        patience=20,
        workers=4,
        project=str(REPO_ROOT / "models"),
        name="trained",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
