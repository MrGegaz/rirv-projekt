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
| Live demo | [scripts/live_demo.py](../scripts/live_demo.py) | OpenCV prozor, FP16 inference, ByteTrack temporal smoothing, distance overlay, rolling FPS overlay |

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

## Distance & speed estimation

Uz svaku detekciju live demo prikazuje grubu procjenu udaljenosti do znaka u
metrima i brzine približavanja u km/h.

**Distance** se računa pinhole-kamera formulom:

```
distance_m = (sign_height_m × focal_length_px) / bbox_height_px
```

Defaultne vrijednosti: `sign_height_m = 0.60` (standardna visina HR cestovnih
znakova) i `focal_length_px = 600` (kalibrirano na demo videima). Obje su
konfigurirane preko `--sign-height` i `--focal-length` flagova ako koristiš
drugu kameru ili znakove autocestaške veličine (~0.90 m).

**Approach speed** se računa kao slope linearne regresije nad zadnjih ~12
uzoraka `(timestamp, distance)` po track ID-u (`np.polyfit` daje `m/s`,
množimo sa 3.6 za km/h). Linearna regresija nad prozorom je puno robusnija
od `Δd / Δt` jer usrednjuje šum iz procjene udaljenosti. Negativan slope =
sign se približava → prikaže se km/h; pozitivan slope (sign već iza vozila)
→ broj se ne prikazuje. Veličina prozora je tunable s `--speed-window`.

Label se renderira u dva reda:
```
speed_limit_50 0.92
dist: 32m spd: 47 km/h
```

Sve je gruba procjena (~±30% na distance, ~±5 km/h na speed) jer nemamo
kalibraciju kamere — bitan je **trend** (broj se monotono smanjuje kako se
vozilo približava znaku; km/h prati subjektivnu brzinu vožnje), ne apsolutna
preciznost. Pojedinačni overlayi se sakrivaju s `--no-distance` ili `--no-speed`.

## Tipične granice
- Sumrak ima slabiji kontrast → mogući propušteni mali znakovi (mitigacija: viši `--imgsz`).
- Class imbalance u datasetu → neki speed_limit razredi imaju manje primjera (potencijalna analiza za izvještaj).
- Ako FPS padne ispod targeta: TensorRT export (`yolo export model=… format=engine half=True`) tipično daje 2-3× boost.

## Reference
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [Kaggle dataset](https://www.kaggle.com/datasets/pkdarabi/cardetection/data)
- Historija razvoja: [docs/labs/lab02](labs/lab02_inicijalna_dokumentacija.md), [docs/labs/lab03](labs/lab03_methodology_traffic_sign_analysis.md)
