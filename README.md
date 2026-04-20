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

## Napomena
Projekt je u fazi aktivnog razvoja. Sadržaj i metrika rezultata će se nadopunjavati nakon anotacije podataka i prvih trening iteracija.
