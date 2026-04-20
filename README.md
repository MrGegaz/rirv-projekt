# Analiza prometnih znakova (Računalni i robotski vid)

## Opis projekta
Ovaj projekt razvija sustav za **detekciju i analizu prometnih znakova** iz videozapisa vožnje. Fokus je na robusnom radu u dva uvjeta snimanja:
- vožnja po danu
- vožnja u laganom sumraku

Cilj je izraditi metodologiju i demonstraciju prikladnu za obranu projekta na kolegiju **Računalni i robotski vid**.

## Glavni ciljevi
- prepoznati i klasificirati prometne znakove iz video okvira
- usporediti performanse modela u dnevnim i sumračnim uvjetima
- analizirati tipične greške (promašene detekcije, lažni pozitivni rezultati)
- pripremiti jasan demo s mjerljivim rezultatima

## Planirani tehnički pristup
- izdvajanje frameova iz ulaznih videa
- anotacija znakova pomoću bounding box oznaka
- treniranje detekcijskog modela (Ultralytics YOLO26, uz YOLO11 kao stabilan fallback)
- usporedni baseline eksperiment (RT-DETR) za metodološku validaciju
- evaluacija po metrikama (npr. mAP, precision, recall)
- kvalitativna analiza rezultata na stvarnim scenama vožnje

## Ključne značajke projekta
- fokus na **realnim prometnim scenama** iz dobivenih videa
- zasebna evaluacija za **day vs. dusk** uvjete

## Struktura repozitorija
- `data/` - sirovi podaci, frameovi i anotacije
- `models/` - checkpoints i izvozni modeli
- `scripts/` - skripte za pripremu podataka, trening i inferencu
- `reports/` - tablice i figure rezultata
- `demo/` - ulazni i izlazni materijali za demonstraciju
- `docs/` - dodatna projektna dokumentacija

## Trenutni status
- inicijalni research dokument: `lab02_inicijalna_dokumentacija.md`
- metodologija za nastavak rada: `lab03_methodology_traffic_sign_analysis.md`
- postavljena osnovna struktura direktorija za daljnji razvoj

## Brzi start (inicijalna verzija)
1. Instalacija ovisnosti:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```
2. Izdvajanje frameova iz videa:
```bash
python scripts/data_prep/extract_frames.py --fps 2.0 --overwrite
```
3. Priprema vanjskog dataseta bez traffic lights:
```bash
python scripts/data_prep/prepare_external_dataset_no_lights.py \
  --src data/external/car \
  --dst data/external/car_no_lights
```
4. Anotacija prema smjernicama:
- `docs/annotation_guidelines.md`
5. (Opcionalno) Generiranje prvog balansiranog anotacijskog batcha:
```bash
python scripts/data_prep/sample_frames_for_annotation.py --total 450 --day-ratio 0.6 --clear
```
6. Trening baseline modela:
```bash
bash scripts/training/train_yolo26_baseline.sh
```
7. Fallback trening (ako treba konzervativniji model):
```bash
bash scripts/training/train_yolo11_fallback.sh
```
8. Predikcija na videu:
```bash
bash scripts/inference/predict_video.sh \
  models/checkpoints/yolo26_baseline/weights/best.pt \
  data/raw/day_video/DayDrive1.mp4
```

## Napomena
Projekt je u fazi aktivnog razvoja. Sadržaj i metrika rezultata će se nadopunjavati nakon anotacije podataka i prvih trening iteracija.
