# Tehnologije i komponente projekta

Pregled svih biblioteka, modela, dataseta i mehanizama korištenih u projektu
**Analiza prometnih znakova** (kolegij Računalni i robotski vid, Veleučilište
u Bjelovaru, 6. semestar). Cilj dokumenta: za svaku komponentu objasniti
**što je** i **zašto je odabrana**.

---

## 1. Python ekosustav

| Tehnologija | Verzija/izvor | Uloga |
|---|---|---|
| **Python 3.12** | system | Glavni jezik. Sve skripte u `scripts/`. |
| **PyTorch** | wheel s CUDA 12.1 support | Deep learning backend za YOLO inference i trening. FP16 (half precision) na GPU radi 2× brže od FP32 bez gubitka mAP-a. |
| **Ultralytics YOLO11** | `ultralytics>=8.3` | High-level wrapper oko YOLO11 modela. Pruža `YOLO()` API, ByteTrack integraciju, AMP trening, automatsku predikciju i export. |
| **OpenCV** | `opencv-python` (Qt backend) | Video I/O, frame drawing, OpenCV window za demo, image manipulation u augmentaciji. |
| **NumPy** | latest | Numerička osnova — bbox aritmetika, linearna regresija za speed estimation, sve transformations. |

Sve dependency-je su u `requirements.txt`. Trening i inference rade u virtual
environment-u (`.venv`).

---

## 2. Hardware target

| Komponenta | Vrijednost |
|---|---|
| GPU | NVIDIA RTX 3050 Laptop (6 GB VRAM) |
| Inference precision | FP16 (half=True) |
| Trening precision | FP16 s AMP (Automatic Mixed Precision) |
| Default inference imgsz | 960 (može do 1280-1600 za bolji recall) |
| Trening imgsz | 640 (memory-constrained na 6 GB s batch 16) |

---

## 3. Model: YOLO11s

| Parametar | Vrijednost |
|---|---|
| Arhitektura | YOLO11 (small varijant) |
| Parametri | 9.44 M |
| GFLOPs | 21.6 |
| Anchor-free | Da (YOLO11 ne koristi predefined anchors) |
| Klase | **20** |

**Zašto YOLO11s:**
- Najnoviji Ultralytics model (release 2024) s boljom mAP/FPS krivuljom od YOLOv8.
- "Small" varijanta je sweet spot za 6 GB GPU — `n` (nano) bi bio brži ali manje točan, `m` (medium) ne stane u VRAM s batch 16.
- Anchor-free arhitektura znači manje hiperparametara za tune-anje, robusnija na variable sign sizes.
- Inference 5.6 ms/img na našem GPU-u — sa source FPS 60 imamo ~3× headroom.

**Pretrained weights:** `yolo11s.pt` (auto-download od Ultralyticsa, treniran na COCO). Ovo je starting point — naš `models/trained/weights/best.pt` je rezultat 80 epoha fine-tuninga na našem datasetu.

---

## 4. Klase (20 ukupno)

### 13 originalnih (Kaggle car_no_lights, IDs 0-12)
`speed_limit_10`, `speed_limit_20`, `speed_limit_30`, `speed_limit_40`,
`speed_limit_50`, `speed_limit_60`, `speed_limit_70`, `speed_limit_80`,
`speed_limit_90`, `speed_limit_100`, `speed_limit_110`, `speed_limit_120`,
`stop`

### 7 ETSD-derivirane (synthetic augmentation, IDs 13-19)
`give_way`, `priority_road`, `no_entry`, `no_left_turn`, `no_right_turn`,
`pedestrian_crossing`, `pass_right`

---

## 5. Dataset izvori

### A. Kaggle car_no_lights (real detection data)
- **Izvor:** [Kaggle car detection dataset](https://www.kaggle.com/datasets/pkdarabi/cardetection/data)
- **Lokacija:** `data/external/car_no_lights/`
- **Veličina:** 3530 train / 801 val / 638 test
- **Format:** YOLO bbox annotations
- **Obrada:** `scripts/prepare_dataset.py` — filtrira semafore (nisu klase za naš projekt), remapira class IDs
- **Karakteristike:** realne fotografije iz vozila, raznolik scenery, prirodne occlusion-i

### B. European Traffic Sign Dataset (ETSD)
- **Izvor:** [Serna & Ruichek, 2018](https://www.researchgate.net/publication/329307891_Classification_of_Traffic_Signs_The_European_Dataset)
- **Lokacija:** `data/external/European Traffic Sign Dataset/`
- **Originalna veličina:** 164 klase, 60k+ slika
- **Format:** klasifikacijski crops (tight cropove znakova bez konteksta)
- **Naš subset:** 7 klasa relevantnih za gradsku vožnju (ETSD IDs 35, 37, 41, 51, 52, 16, 87)
- **Caveat:** ETSD je classification-only format — direktan trening YOLO-a na ovome bi naučio "znak uvijek puni frame", što ne generalizira. Rješenje: synthetic detection generation (vidi sekciju 6).

### C. Profesorovi demo videi
- **Lokacija:** `data/raw/day_video/*.mp4`, `data/raw/dusk_video/*.mp4`
- **Format:** 1280×720 @ 59.9 FPS, dashcam snimke gradske vožnje
- **Uloga:** **Samo za evaluaciju / live demo**, **NIKAD nisu trening podaci**. Ovo je tvrdo pravilo projekta.

---

## 6. Synthetic detection augmentation (`scripts/augment_etsd.py`)

Pretvara ETSD klasifikacijske cropove u YOLO detection format pomoću
**paste-on-background** tehnike.

### Pipeline

1. **Background pool** (`gather_backgrounds()`)
   - 200 random slika iz `car_no_lights/train/images/` s **praznim** label fajlovima (znači stvarno gradska scena bez znakova).
   - Fallback: `procedural_background()` generira gradijent neba + asfalt s noise-om.

2. **Sign augmentation** (`augment_sign()`)
   - **Rotation:** ±15°
   - **Perspective warp:** random corner shift ±15% — simulira skew kako voziš pored znaka
   - **HSV color jitter:** Hue ±8, Saturation 0.7-1.3×, Value 0.7-1.3× + offset
   - **Motion blur:** directional kernel 5-11 px s random angle 0-180° — simulira camera shake / vehicle motion
   - **JPEG compression:** re-encode na quality 50-90 — uvodi compression artifacts

3. **Composition** (`generate_sample()`)
   - **Scale:** target width random 20-180 px (širi raspon nego što ETSD originally daje)
   - **Position:** random u gornjih 65% slike (znakovi ne stoje na asfaltu)
   - **Alpha paste:** gaussian-feathered edge (smanjuje sharp boundary artifact)

4. **Hard negative backgrounds** (15% samples)
   - Pure background slike s **praznim** labels. Uče model "nije svaki dio scene znak", smanjuju false positives na reklamama/zgradama.

### Output
- `data/external/etsd_synthetic/{train,val}/{images,labels}/`
- 7 klasa × 500 train + 80 val = 3500 + 560 sign samples
- Plus ~525 train + 262 val hard-negative backgrounds
- Sanity montage: `synthetic_montage.jpg` (12 random sampled s bbox-ovima crtanim)

### Zašto ovaj pristup
- Direktan trening na klasifikacijskim cropovima bi učio "znak puni frame" → ne generalizira
- Manual anotacija novih realnih primjera = sati posla
- Synthetic paste s jakom augmentacijom = nekoliko minuta, model uči znakove **na različitim pozicijama, skalama i pod realnim distortion-ima**
- Hard negatives osiguravaju da model ne FP-a na sličnim shape patterns (round reklame, crveni patterns, etc.)

---

## 7. Trening pipeline (`scripts/train.py`)

```python
model = YOLO("yolo11s.pt")
model.train(
    data="config/data.yaml",
    epochs=80,
    imgsz=640,
    batch=16,
    device=0,
    amp=True,
    cos_lr=True,
    patience=20,
)
```

| Hiperparametar | Vrijednost | Zašto |
|---|---|---|
| `epochs` | 80 | Plateau viđen oko 60-70, 80 je safety margin |
| `imgsz` | 640 | Maksimum koji stane u 6 GB s batch 16 + AMP |
| `batch` | 16 | Memory limit |
| `optimizer` | auto (Ultralytics bira) → AdamW | Auto-selection radi dobro |
| `lr0` | auto | Cosine LR schedule s warmup |
| `cos_lr` | True | Cosine annealing — glatkiji convergence |
| `amp` | True | FP16 trening = 2× brže, isti rezultat |
| `patience` | 20 | Early stop ako 20 epoha bez improvement |
| Workers | 4 | DataLoader paralelizam |

### Sleep-prevention pri treniranju
Trening pokreta se pod `systemd-inhibit --what=sleep:idle --mode=block` da
laptop ne uđe u sleep tijekom dugih runs (prethodni run je hangao zbog CUDA
sleep crash-a).

### Final metrike (v2 hardened model)
- **mAP50:** 0.992
- **mAP50-95:** 0.929
- **Precision:** 0.992
- **Recall:** 0.974
- **Inference:** 5.6 ms/img na RTX 3050

---

## 8. Live demo pipeline (`scripts/live_demo.py`)

Glavna demonstracijska komponenta. Otvara OpenCV prozor, čita video frame-by-frame,
runa YOLO inference, primjenjuje stabilizacijske mehanizme, prikazuje detekcije
+ overlays.

### 8.1 Inference

```python
model = YOLO("models/trained/weights/best.pt")
results = model.track(
    frame,
    imgsz=960,
    conf=0.40,
    iou=0.50,
    device=0,
    half=True,
    persist=True,
    tracker="bytetrack.yaml",
)
```

- **`imgsz=960`** — više od trening 640 jer infer s većom rezolucijom često
  bolje detektira sitne znakove
- **`conf=0.40`** — globalna minimum confidence (per-class overrides postoje)
- **`iou=0.50`** — NMS IoU threshold; spušten s defaulta 0.7 da nearby
  different-class boxovi koegzistiraju (stacked signs na stupu)
- **`half=True`** — FP16 = 2× brže inference
- **`tracker="bytetrack.yaml"`** — Ultralytics-ov ByteTrack config za persistent
  track IDs između framova

### 8.2 ByteTrack temporal smoothing

ByteTrack je tracker koji svakoj detekciji dodjeljuje **persistent ID-jeve**
across frames. Za našu upotrebu:

- **Per-track classification buffer** (`TrackHistory.update()`) — drži zadnje
  N=7 raw class predictions po track ID-u
- **Confirmation policy** — class label se prikazuje tek kad se ista klasa
  pojavi ≥4 puta u prozoru od 7 framova
- **Učinak:** single-frame flicker FP-ovi (reklame koje briefly liče na znak)
  se filtriraju

### 8.3 Bbox EMA smoothing

```
smoothed = α × current + (1 - α) × previous   (α = 0.7)
```

Bez ovoga, YOLO bbox koordinate variraju 2-3 px po framu — vidljivi twitch
prozora. EMA glada bez vidljivog lag-a.

### 8.4 Label persistence (ghost tracks)

Najvažniji UX fix. Kad track izgubi detekciju, last confirmed label/bbox
ostaje vidljiv **20 framova** (~330ms @ 60 FPS) s **alpha fade** (1.0 → 0.3).

- Bez ovog: label flicker traje 50-100ms, ne stigne se pročitati
- S ovim: label ostaje vidljiv dovoljno dugo da bude human-readable

### 8.5 ROI mask

`--roi-bottom-margin 0.30` (default) — detekcije čiji center pada u donjih
30% slike se odbacuju. Tu su cesta, haube, ne znakovi.

`--roi-side-margin 0.0` (default) — bez horizontalne maske jer su valid
right-side curb signs.

### 8.6 Geometric FP filters

- **Min bbox size** (`--min-bbox-px 18`) — odbacuje tinydetekcije gdje je
  model najmanje siguran
- **Max aspect ratio** (`--max-aspect-ratio 1.8`) — odbacuje wide-rectangle
  detekcije (reklame); svi trenirani znakovi su ≈ kvadrat

### 8.7 Per-class confidence overrides

```python
PER_CLASS_CONF = {
    "pass_right": 0.60,      # suppress pass_straight (out-of-taxonomy) FP-ove
    "priority_road": 0.30,   # known weak klasa — lower bar za catch more
}
```

### 8.8 Distance estimation (pinhole camera model)

```
distance_m = (sign_height_m × focal_length_px) / bbox_height_px
```

| Parametar | Default | Tunable |
|---|---|---|
| `sign_height_m` | 0.60 m | `--sign-height` |
| `focal_length_px` | 600 | `--focal-length` (kalibrirano na demo video-ima) |

Gruba procjena ±30%, ali za demo svrhe dovoljno — pokazuje **trend** (broj
monotono pada kako se vozilo približava).

### 8.9 Approach speed estimation (linearna regresija)

Per track ID, drži rolling window od 12 zadnjih (timestamp, distance)
samples. Pomoću `numpy.polyfit` izračuna slope:

```python
slope_mps = np.polyfit(timestamps, distances, 1)[0]
approach_mps = -slope_mps        # negativan slope = sign se približava
speed_kmh = approach_mps * 3.6 * speed_scale
```

`--speed-scale 1.1` (default) kompenzira blagi underestimation zbog mix
veličina znakova (jedan `sign_height_m=0.60` ne odgovara svim klasama
jednako).

Linearna regresija je puno robusnija od `Δd/Δt` jer usrednjuje noise.

### 8.10 Real-time frame skip

Glavni fix za "video djeluje usporeno u fullscreen-u":

```python
elapsed = now - playback_start
expected_idx = anchor + int(elapsed × source_fps × playback_speed)
while frame_idx < expected_idx:
    cap.grab()  # advance bez decode (cheap)
    frame_idx += 1
    frames_skipped += 1
```

Auto-detect source FPS preko `cap.get(cv2.CAP_PROP_FPS)` (default je `-1` =
auto). On-screen `playback: X.XXx` indikator pokazuje stvarni real-time omjer.

### 8.11 Keyboard shortcuts

| Tipka | Akcija |
|---|---|
| `space` | pauza / resume |
| `+` / `=` | ubrzaj 1.25× (max 4×) |
| `-` / `_` | uspori 1.25× (min 0.25×) |
| `r` | reset speed na 1.0× |
| `q` | quit |

### 8.12 Recent detections sidebar

Top-right panel s zadnjih 5 unique confirmed detekcija. Format:

```
Recent:
priority_road 27m 51km/h
stop 14m 33km/h
...
```

Backup čitljivosti — i kad label flickne prebrzo, vidiš u sidebaru što je
upravo prošlo.

### 8.13 OpenGL window backend (opcionalno)

`cv2.WINDOW_OPENGL` flag pokušava GPU-accelerated rendering za fullscreen.
Ako cv2 build nije compiled s OpenGL → fallback na CPU + frame skip ga
kompenzira.

---

## 9. Datotečna struktura

```
RiRV/
├── config/
│   └── data.yaml                       # YOLO dataset config (20 klasa, list-format multi-root)
├── data/
│   ├── external/
│   │   ├── car_no_lights/              # Kaggle dataset (gitignored)
│   │   ├── etsd_synthetic/             # generated paste-on-background (gitignored)
│   │   └── European Traffic Sign Dataset/  # ETSD klasifikacijski crops (gitignored)
│   └── raw/
│       ├── day_video/                  # demo videi (gitignored, NE training)
│       └── dusk_video/                 # demo videi (gitignored, NE training)
├── docs/
│   ├── project_summary.md              # podrobni sažetak projekta
│   └── labs/                           # arhivirane lab dokumentacije lab02, lab03
├── logs/                               # trening logs (gitignored)
├── models/
│   ├── trained/
│   │   ├── weights/
│   │   │   ├── best.pt                 # finalni trained model (commited)
│   │   │   └── last.pt                 # last checkpoint (gitignored)
│   │   ├── results.csv                 # per-epoch metrike (gitignored)
│   │   └── ... (other Ultralytics outputs)
│   └── trained_v1_backup/              # rollback safety net (gitignored)
├── scripts/
│   ├── prepare_dataset.py              # (jednokratno) Kaggle preprocessing
│   ├── augment_etsd.py                 # (jednokratno) ETSD synthetic generation
│   ├── train.py                        # YOLO11s trening
│   └── live_demo.py                    # glavna demo aplikacija
├── README.md                           # quick start + flagovi
├── TEHNOLOGIJE.md                      # ovaj dokument
├── requirements.txt                    # Python dependencies
└── .gitignore
```

---

## 10. Tijek razvoja (kronološki)

1. **lab02-lab03** — inicijalna istraživanja, prvo prototipiranje (arhivirano u `docs/labs/`)
2. **Cleanup + pivot** (commit `98d98ab`) — uklonjen Streamlit dashboard, hybrid pipeline ideja
   odbačena, fokus na YOLO11s
3. **ByteTrack temporal smoothing** (commit `cbe0053`) — riješen flicker FP problem
4. **Distance estimation** (commit `f2f03eb`) — pinhole camera model overlay
5. **Approach speed estimation** (commit `3cb6a08`) — linearna regresija nad distance window
6. **20-class proširenje** (commit `c926e0d`) — dodano 7 ETSD klasa preko synthetic augmentation
7. **Hardened retrain + inference stabilizacija** (commit `a4edcde`) — perspective warp,
   motion blur, JPEG compression, hard negatives + label persistence, frame skip, keyboard
   shortcuts, sidebar

---

## 11. Iskrene limitacije

| Limitacija | Opis | Mitigacija |
|---|---|---|
| **Sim-to-real domain gap** | ETSD synthetic ima clean frontal crops; real signs imaju shadows, weather wear, mounting strukture, camera-specific color shifts. Vidljivo posebice na nekim priority_road instancama gdje model "ne vidi" znak iako je očit | Inference tweaks (per-class conf, label persistence). Stvarni fix bi tražio real labeled data + retrain. |
| **Out-of-taxonomy klase** | `pass_straight` nije među 20 treniranih klasa → model fallback-a na najsličniji (`pass_right`) | Per-class conf override (pass_right 0.60); za real fix dodati klasu i retrain |
| **Stacked signs na istom stupu** | Model često prikaže samo onaj s većim confidenceom | NMS iou=0.50 (lower threshold) pomaže ali ne rješava potpuno |
| **Dnevni uvjeti generalno teži od noćnih** | Više visual clutter-a (reklame, pedestrians, shadows) | Empirijski opažanje — night demo je solidnija demonstracija (paradoxalno, jer su znakovi optimizirani za noćnu retroreflektivnost) |
| **Distance/speed nisu kalibrirana** | ~±30% na distance, ~±5 km/h na speed | `--speed-scale` i `--focal-length` flagovi za fine-tune |
| **Sumrak** ima slabiji kontrast | Mogući propušteni mali znakovi | Mitigacija: `--imgsz 1280` |

---

## 12. Reference

- **Ultralytics YOLO11:** https://docs.ultralytics.com/models/yolo11/
- **ByteTrack paper:** Zhang et al., 2022, "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
- **European Traffic Sign Dataset:** Serna & Ruichek, 2018
- **Kaggle car_no_lights:** https://www.kaggle.com/datasets/pkdarabi/cardetection/data
- **Pinhole camera model:** standardna computer vision literatura (npr. Hartley & Zisserman, "Multiple View Geometry")
- **Historija razvoja:** `docs/labs/lab02_inicijalna_dokumentacija.md`, `docs/labs/lab03_methodology_traffic_sign_analysis.md`
