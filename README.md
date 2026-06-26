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
   counterom, procjenom udaljenosti svakog znaka u metrima i procjenom
   brzine približavanja u km/h. Quit: `q`. Korisni flagovi:
   `--no-distance` (skida distance overlay), `--no-speed` (skida km/h),
   `--no-tracking` (raw per-frame predikcije, bez km/h jer nema track ID-a),
   `--focal-length N` (kalibracija za drugu kameru).

## Struktura

```
config/data.yaml                 # YOLO dataset config (13 klasa)
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
