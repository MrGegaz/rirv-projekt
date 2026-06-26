# Project Summary — Analiza prometnih znakova

**Kolegij:** Računalni i robotski vid (RiRV)
**Institucija:** Veleučilište u Bjelovaru, 6. semestar

## Cilj
Detekcija i klasifikacija prometnih znakova iz video snimaka vožnje, demonstrirano preko **live OpenCV prozora** sa stabilnom inferencijom ≥25 FPS.

## Pristup
- **Model:** YOLO11s (Ultralytics)
- **Hardware target:** NVIDIA RTX 3050 Laptop 6 GB (FP16 inference)
- **Trening dataset:** kombinacija dva izvora
  - **Realna detekcija:** [Kaggle car detection](https://www.kaggle.com/datasets/pkdarabi/cardetection/data), pripremljen u `data/external/car_no_lights/` (3530 train / 801 val / 638 test, semafori uklonjeni, remapan u 13 klasa)
  - **Synthetic augmentacija:** [European Traffic Sign Dataset](https://www.researchgate.net/publication/329307891_Classification_of_Traffic_Signs_The_European_Dataset) (Serna & Ruichek, 2018) — 7 dodatnih gradskih klasa generiranih preko paste-on-background sintetike (`scripts/augment_etsd.py`, 500 train + 80 val po klasi + 15% hard-negative backgrounds). Augmentacije: random scale 20-180 px, rotation ±15°, perspective warp (±15% corner shift), HSV color jitter, motion blur (5-11 px directional kernel), JPEG compression artifacts.
- **Evaluacija:** profesorovi videi (`data/raw/day_video/`, `data/raw/dusk_video/`) — koriste se samo za live demo, **nikad nisu trening podaci**
- **Taksonomija:** 20 klasa
  - 13 originalnih (speed_limits 10-120, stop)
  - 7 ETSD: give_way, priority_road, no_entry, no_left_turn, no_right_turn, pedestrian_crossing, pass_right

## Pipeline

| Korak | Skripta | Opis |
|---|---|---|
| (1×) Priprema Kaggle dataseta | [scripts/prepare_dataset.py](../scripts/prepare_dataset.py) | Filtrira semafore i remapira klase iz sirovog Kaggle exporta |
| (1×) Synthetic detection iz ETSD | [scripts/augment_etsd.py](../scripts/augment_etsd.py) | Paste-on-background generator za 7 dodatnih gradskih klasa iz ETSD klasifikacijskih cropova |
| Trening | [scripts/train.py](../scripts/train.py) | YOLO11s, 80 epoha, batch 16, imgsz 640, AMP, cos_lr |
| Live demo | [scripts/live_demo.py](../scripts/live_demo.py) | OpenCV prozor, FP16 inference, ByteTrack temporal smoothing, distance + approach speed overlay, ROI mask, label persistence, sidebar, frame-skip real-time sync, playback speed shortcuts |

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
preciznost. `--speed-scale 1.1` default kompenzira blagi underestimation zbog
miks veličina znakova vs naša jedna pretpostavka 0.60 m. Pojedinačni overlayi
se sakrivaju s `--no-distance` ili `--no-speed`.

## Inference stabilizacija (live demo)

Bez retraina, niz inference-side mehanizama poboljšava čitljivost i smanjuje
false positives:

- **ByteTrack temporal smoothing** (`--track-window 7 --track-min-hits 4`) — class
  label se prikazuje tek kad se ista klasa pojavi 4 puta u 7-frame prozoru po track ID-u.
  Eliminira single-frame flicker FP-ove (reklame koje samo briefly liče).
- **Label persistence** — confirmed label/bbox ostaje vidljiv 20 framova (~330ms @ 60 FPS)
  nakon detection loss-a, s fadeom 1.0 → 0.3. Čitljivost čak i kad detekcija na kratko padne.
- **Bbox EMA smoothing** (α=0.7) — box ne skače piksel-po-piksel jer YOLO predikcije variraju.
- **ROI mask** (`--roi-bottom-margin 0.30`) — donja 30% slike (cesta, haube) se ignorira.
- **Geometric filters** — `--min-bbox-px 18` reže šum, `--max-aspect-ratio 1.8` reže
  wide-rectangle reklame.
- **Per-class confidence overrides** — `pass_right` 0.60 (suppress out-of-taxonomy
  `pass_straight` misclassifications), `priority_road` 0.30 (catch more u known weak klasi).
- **Real-time sync s frame skip** — ako render ne stiže držati source FPS (npr. fullscreen),
  source frames se drop-aju umjesto da playback uspori. Video ostaje real-time 1.0x.
  On-screen `playback: X.XXx` indikator potvrđuje točan timing.
- **Recent detections sidebar** — top-right panel s zadnjih 5 unique detekcija
  (klasa + distance + km/h). Backup čitljivost kad label-i prolaze prebrzo.

## Tipične granice
- **Synthetic-to-real domain gap**: ETSD source crops su konzistentni frontal views;
  realni gradski znakovi mogu imati shadows, weather wear, mounting strukture, color
  shifts ovisne o kameri koje paste-on-background ne reproducira savršeno. Vidljivo
  npr. na nekim priority_road instancama gdje model "ne vidi" znak iako je čovjeku
  jasan — to je well-documented sim-to-real problem.
- **Out-of-taxonomy klase**: znakovi koji nisu među 20 trained klasa (npr. `pass_straight`)
  ponekad triggerju najbližu trained klasu (`pass_right`). Mitigirano per-class conf
  override-om, ali za real fix treba dodati klasu i retrain.
- **Sumrak** ima slabiji kontrast → mogući propušteni mali znakovi (mitigacija: viši `--imgsz`).
- Ako FPS padne ispod targeta: TensorRT export (`yolo export model=… format=engine half=True`) tipično daje 2-3× boost.

## Reference
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [Kaggle dataset](https://www.kaggle.com/datasets/pkdarabi/cardetection/data)
- Historija razvoja: [docs/labs/lab02](labs/lab02_inicijalna_dokumentacija.md), [docs/labs/lab03](labs/lab03_methodology_traffic_sign_analysis.md)
