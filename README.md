# Analiza prometnih znakova (RiRV)

Detekcija prometnih znakova iz video zapisa vožnje pomoću **YOLO11s**.
Model je treniran na kombinaciji javnog Kaggle dataseta (`car_no_lights`, 13
klasa) i sintetičnih primjera generiranih iz **European Traffic Sign Dataset**
(7 dodatnih gradskih klasa). Profesorovi videi se koriste samo kao
**demonstracijski / evaluacijski materijal**.

## Klase (20 ukupno)
- Speed limits (13): `speed_limit_10/20/30/40/50/60/70/80/90/100/110/120`, `stop`
- ETSD dodatne (7): `give_way`, `priority_road`, `no_entry`, `no_left_turn`,
  `no_right_turn`, `pedestrian_crossing`, `pass_right`

## Brzi start

1. **Instalacija** (jednom):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   # torch + CUDA wheel posebno; vidi requirements.txt
   ```

2. **Trening** (opcionalno — preskoči ako već postoji `models/trained/weights/best.pt`):
   ```bash
   python scripts/train.py
   ```
   Ciljano za RTX 3050 6 GB: YOLO11s, batch 16, imgsz 640, AMP, 80 epoha.

3. **Live demo**:
   ```bash
   python scripts/live_demo.py --video data/raw/day_video/DayDrive1.mp4
   ```
   Otvara OpenCV prozor s detekcijama, ByteTrack temporal smoothingom, FPS
   counterom, real-time playback ratio indikatorom, procjenom udaljenosti
   svakog znaka u metrima, procjenom brzine približavanja u km/h, ROI maskom
   (donja 30% slike se ignorira), label persistencom (~330ms nakon detection
   loss-a) i sidebarom u gornjem desnom kutu s zadnjih 5 detekcija.

   **Keyboard shortcuts u runtime-u:**
   - `space` — pauza / resume
   - `+` / `=` — ubrzaj 1.25× (max 4×)
   - `-` / `_` — uspori 1.25× (min 0.25×)
   - `r` — reset playback speed na 1.0×
   - `q` — quit

   **Korisni CLI flagovi:**
   - `--imgsz 1280` — veći input → bolje detekcije sitnih znakova (FPS cost)
   - `--conf 0.30` — niži global confidence threshold
   - `--speed-scale 1.25` — calibrate prikazan km/h (default 1.1)
   - `--roi-bottom-margin 0.30` — fraction donje slike koja se ignorira
   - `--min-bbox-px 18` — reject sitne detekcije (noise zone)
   - `--max-aspect-ratio 1.8` — reject wide-rectangle FP-ove (reklame)
   - `--target-fps -1` — auto-match source FPS (default); 0 = uncapped
   - `--no-distance`, `--no-speed`, `--no-sidebar`, `--no-tracking`

## Struktura

```
config/data.yaml                 # YOLO dataset config (20 klasa)
data/external/car_no_lights/     # training data (Kaggle, semafori uklonjeni)
data/raw/{day_video,dusk_video}/ # demo videi
models/trained/weights/best.pt   # istrenirani model
scripts/train.py                 # trening
scripts/live_demo.py             # demo
scripts/prepare_dataset.py       # (jednokratno) preparacija Kaggle dataseta
scripts/augment_etsd.py          # (jednokratno) synthetic detection iz ETSD crops
data/external/etsd_synthetic/    # generated synthetic samples (gitignored)
docs/                            # project_summary + arhivirani lab dokumenti
```
