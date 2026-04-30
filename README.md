# OpenSky DVC cevovod

Ta projekt uporablja podatkovni cevovod z DVC za zajem, predobdelavo in validacijo podatkov iz OpenSky Network API. Prilagoditev sledi istemu principu kot primer z Arnes, vendar je parametrizirana za letalske `state vector` podatke in za lokalni zračni prostor, ki ga določimo z bounding box območjem.

## Pričakovani rezultat

Razvit je ponovljiv DVC cevovod, ki:

- zajame OpenSky posnetek v `data/raw`,
- iz vseh surovih posnetkov zgradi obdelane tabele v `data/processed`,
- pripravi kumulativno zgodovino letov `states_history.csv`,
- izvede validacijo in shrani poročilo v `reports/validation/opensky_validation.json`.

## Parametri cevovoda

V datoteki `params.yaml` so zbrani vsi ključni parametri:

```yaml
fetch:
  url: "https://opensky-network.org/api/states/all"
  output_dir: "data/raw"
  timeout_seconds: 30
  bbox:
    lamin: 42.5
    lomin: 15.5
    lamax: 46.5
    lomax: 19.8

preprocess:
  raw_dir: "data/raw"
  output_dir: "data/processed"
  format: "csv"
  history_filename: "states_history.csv"

validate:
  processed_dir: "data/processed"
  history_file: "data/processed/states_history.csv"
  report_path: "reports/validation/opensky_validation.json"
```

Pomembna OpenSky prilagoditev je `fetch.bbox`, s katerim omejimo zajem na izbran del zračnega prostora. To je smiselna zamenjava za parameter `station` iz Arnes primera.

## Faze DVC

Projekt uporablja tri faze:

1. `fetch`
   Prenese najnovejši OpenSky snapshot in ga shrani v `data/raw/states_<timestamp>.json`.

2. `preprocess`
   Prebere vse snapshot datoteke iz `data/raw`, vsako pretvori v tabelo `states_processed_<timestamp>.csv` in nato zgradi deterministično zgodovinsko tabelo `states_history.csv`.

3. `validate`
   Preveri obdelane podatke in shrani validacijsko poročilo v JSON obliki.

Datoteka `dvc.yaml` je prilagojena tako, da DVC spremlja:

- odvisnosti skript,
- vhodne mape z raw podatki,
- parametre iz `params.yaml`,
- izhode obdelave in metrike validacije.

## Zakaj je ta prilagoditev primerna za OpenSky

Za DVC je pomembno, da je posamezna faza ponovljiva. Zato `preprocess` ne dograjuje zgodovine na podlagi prejšnjega izhoda, ampak jo ob vsakem zagonu ponovno zgradi iz vseh datotek v `data/raw`. S tem je rezultat odvisen samo od vhodov in parametrov, kar je pravilnejše za DVC kot inkrementalno dodajanje vrstic v obstoječo izhodno datoteko.

Enakovredno zahtevi "da cevovod deluje za vsa merilna mesta" v OpenSky projektu pomeni:

- da zajamemo vse zrakoplove znotraj izbranega bounding box območja,
- da predobdelava deluje za vse zbrane raw snapshot datoteke,
- da validacija preveri celotno kumulativno zgodovino.

## Zagon

Za ponovni izračun cevovoda uporabi:

```bash
dvc repro
```

Za deljenje rezultatov po uspešni izvedbi:

```bash
git push
dvc push
```

Avtomatskega `git commit` in `git push` nisem vključil neposredno v `dvc.yaml`, ker sta to okoljsko odvisna koraka, ki poslabšata prenosljivost in ponovljivost cevovoda. Bolj varno je, da ostaneta ločena od same obdelave podatkov.
