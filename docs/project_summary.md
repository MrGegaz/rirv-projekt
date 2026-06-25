# Project Summary — Analiza prometnih znakova

**Kolegij:** Računalni i robotski vid (RiRV)
**Institucija:** Veleučilište u Bjelovaru, 6. semestar

## Cilj
Detekcija i klasifikacija prometnih znakova iz video snimaka vožnje, demonstrirano preko **live OpenCV prozora** sa stabilnom inferencijom ≥25 FPS.

## Pristup
- **Model:** YOLO11s (Ultralytics)
- **Hardware target:** NVIDIA RTX 3050 Laptop 6 GB (FP16 inference)
- **Trening dataset:** [Kaggle car detection](https://www.kaggle.com/datasets/pkdarabi/cardetection/data), pripremljen u `data/external/car_no_lights/` (3530 train / 801 val / 638 test, semafori uklonjeni, remapan u 13 klasa)
- **Evaluacija:** profesorovi videi (`data/raw/day_video/`, `data/raw/dusk_video/`) — koriste se samo za live demo, **nikad nisu trening podaci**
- **Taksonomija:** 13 klasa — `speed_limit_{10,20,30,40,50,60,70,80,90,100,110,120}`, `stop`

## Pipeline

| Korak | Skripta | Opis |
|---|---|---|
| (1×) Priprema dataseta | [scripts/prepare_dataset.py](../scripts/prepare_dataset.py) | Filtrira semafore i remapira klase iz sirovog Kaggle exporta |
| Trening | [scripts/train.py](../scripts/train.py) | YOLO11s, 80 epoha, batch 16, imgsz 640, AMP, cos_lr |
| Live demo | [scripts/live_demo.py](../scripts/live_demo.py) | OpenCV prozor, FP16 inference, rolling FPS overlay |

## Hiperparametri (sažetak)
- model: `yolo11s.pt`
- epochs: 80, patience: 20
- batch: 16, imgsz: 640
- optimizer: auto (SGD/Adam selekcija od Ultralyticsa)
- amp: True, cos_lr: True
- device: CUDA 0

## Demo scenario za obranu
1. Pokrenuti `live_demo.py` na `DayDrive1.mp4` (dnevna scena).
2. Po želji pokrenuti na `NightDrive1.mp4` (sumrak) — kvalitativna usporedba.
3. FPS counter u kutu prikazuje stabilnost inference brzine.

## Tipične granice
- Sumrak ima slabiji kontrast → mogući propušteni mali znakovi (mitigacija: viši `--imgsz`).
- Class imbalance u datasetu → neki speed_limit razredi imaju manje primjera (potencijalna analiza za izvještaj).
- Ako FPS padne ispod targeta: TensorRT export (`yolo export model=… format=engine half=True`) tipično daje 2-3× boost.

## Reference
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [Kaggle dataset](https://www.kaggle.com/datasets/pkdarabi/cardetection/data)
- Historija razvoja: [docs/labs/lab02](labs/lab02_inicijalna_dokumentacija.md), [docs/labs/lab03](labs/lab03_methodology_traffic_sign_analysis.md)
