# OpenSky DVC Pipeline

Ta projekt uporablja podatkovni cevovod z DVC za zajem, predobdelavo in validacijo podatkov iz OpenSky Network API. Prilagoditev sledi istemu principu kot primer z Arnes, vendar je parametrizirana za letalske `state vector` podatke in za lokalni zracni prostor, ki ga dolocimo z bounding box obmocjem.

## Pricakovani rezultat

Razvit je ponovljiv DVC cevovod, ki:

- zajame OpenSky posnetek v `data/raw`,
- iz vseh surovih posnetkov zgradi obdelane tabele v `data/processed`,
- pripravi kumulativno zgodovino letov `states_history.csv`,
- izvede validacijo in shrani porocilo v `reports/validation/opensky_validation.json`.

## Parametri cevovoda

V datoteki `params.yaml` so zbrani vsi kljucni parametri:

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

Pomembna OpenSky prilagoditev je `fetch.bbox`, s katerim omejimo zajem na izbran del zracnega prostora. To je smiselna zamenjava za parameter `station` iz Arnes primera.

## Faze DVC

Projekt uporablja tri faze:

1. `fetch`
   Prenese najnovejsi OpenSky snapshot in ga shrani v `data/raw/states_<timestamp>.json`.

2. `preprocess`
   Prebere vse snapshot datoteke iz `data/raw`, vsako pretvori v tabelo `states_processed_<timestamp>.csv` in nato zgradi deterministicno zgodovinsko tabelo `states_history.csv`.

3. `validate`
   Preveri obdelane podatke in shrani validacijsko porocilo v JSON obliki.

Datoteka `dvc.yaml` je prilagojena tako, da DVC spremlja:

- odvisnosti skript,
- vhodne mape z raw podatki,
- parametre iz `params.yaml`,
- izhode obdelave in metrike validacije.

## Zakaj je ta prilagoditev primerna za OpenSky

Za DVC je pomembno, da je posamezna faza ponovljiva. Zato `preprocess` ne dograjuje zgodovine na podlagi prejsnjega izhoda, ampak jo ob vsakem zagonu ponovno zgradi iz vseh datotek v `data/raw`. S tem je rezultat odvisen samo od vhodov in parametrov, kar je pravilnejse za DVC kot inkrementalno dodajanje vrstic v obstojeco izhodno datoteko.

Enakovredno zahtevi "da cevovod deluje za vsa merilna mesta" v OpenSky projektu pomeni:

- da zajamemo vse zrakoplove znotraj izbranega bounding box obmocja,
- da predobdelava deluje za vse zbrane raw snapshot datoteke,
- da validacija preveri celotno kumulativno zgodovino.

## GitHub Actions in DVC

V posodobljenem GitHub delovnem toku smo izvedli nekaj kljucnih sprememb, ki izboljsujejo preglednost, avtomatizacijo in zanesljivost procesiranja OpenSky podatkov. Po koraku za nastavitev Pythona je dodan locen korak za nastavitev Git konfiguracije, kar zagotavlja, da so vse nadaljnje Git operacije v GitHub Actions pravilno podpisane s podatki avtomatiziranega okolja.

Pomembna sprememba se nanasa tudi na upravljanje z DVC. Namesto rocnega poganjanja posameznih skript se uporablja ukaz `dvc repro`, ki samodejno izvede vse potrebne faze, definirane v `dvc.yaml`. To je za OpenSky projekt posebej koristno, ker se ob vsakem zagonu dosledno izvede celoten tok `fetch -> preprocess -> validate`, brez rocnega usklajevanja posameznih korakov.

Nastavitve za oddaljeni DVC vir so locene v poseben korak `DVC setup remote`, kjer se konfigurira DagsHub S3 endpoint ter poverilnice za dostop do DVC oddaljene shrambe. S tem je delovni tok bolj pregleden in lazje vzdrzevan.

V delu, kjer se izvede cevovod, workflow najprej poklice `dvc pull --allow-missing`. Ta prilagoditev je pomembna zato, ker validacijsko porocilo ob prvem zagonu se morda ne obstaja v DVC cache-u. Z uporabo `--allow-missing` workflow ne odpove po nepotrebnem, ampak nadaljuje do `dvc repro`, kjer se manjkajoci artefakti ponovno ustvarijo, nato pa se z `dvc push` shranijo v oddaljeni DVC repozitorij.

Na koncu se namesto rocnega dodajanja posameznih podatkovnih datotek v Git doda in potrdi datoteka `dvc.lock`, ki belezi trenutno stanje rezultatov cevovoda. Ce se v cevovodu ne zgodi nobena sprememba, bo korak `git commit` varno preskocen zaradi dodatka `|| true`. S tem smo tok poenostavili in zagotovili, da se celoten postopek pridobivanja, obdelave, validacije in shranjevanja OpenSky podatkov izvaja na ponovljiv in konsistenten nacin.

## GitHub workflow za OpenSky

Celoten GitHub Actions potek dela z uporabo DVC cevovodov:

```yaml
name: Fetch data on schedule

on:
  workflow_dispatch:
  schedule:
    - cron: "0 0 * * *"

permissions:
  contents: write

jobs:
  fetch_opensky:
    name: Fetch and pre-process OpenSky data
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.PAT_TOKEN || github.token }}
          fetch-depth: 0

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.14"

      - name: Install uv
        uses: astral-sh/setup-uv@v5

      - name: Install dependencies
        run: uv sync

      - name: Setup Git
        run: |
          git config --local user.email "actions@github.com"
          git config --local user.name "GitHub Actions"

      - name: DVC setup remote
        env:
          DAGSHUB_S3_ENDPOINT_URL: ${{ vars.DAGSHUB_S3_ENDPOINT_URL }}
          DAGSHUB_ACCESS_KEY_ID: ${{ secrets.DAGSHUB_ACCESS_KEY_ID }}
          DAGSHUB_SECRET_ACCESS_KEY: ${{ secrets.DAGSHUB_SECRET_ACCESS_KEY }}
        run: |
          if [ -n "$DAGSHUB_S3_ENDPOINT_URL" ]; then
            uv run dvc remote add -d origin s3://dvc --force
            uv run dvc remote modify origin endpointurl "$DAGSHUB_S3_ENDPOINT_URL"
            uv run dvc remote modify origin --local access_key_id "$DAGSHUB_ACCESS_KEY_ID"
            uv run dvc remote modify origin --local secret_access_key "$DAGSHUB_SECRET_ACCESS_KEY"
          else
            echo "DAGSHUB_S3_ENDPOINT_URL is not set. Skipping remote setup."
          fi

      - name: Run DVC pipeline
        env:
          OPENSKY_USERNAME: ${{ secrets.OPENSKY_USERNAME }}
          OPENSKY_PASSWORD: ${{ secrets.OPENSKY_PASSWORD }}
          OPENSKY_URL: ${{ vars.OPENSKY_URL }}
          DAGSHUB_S3_ENDPOINT_URL: ${{ vars.DAGSHUB_S3_ENDPOINT_URL }}
          DAGSHUB_ACCESS_KEY_ID: ${{ secrets.DAGSHUB_ACCESS_KEY_ID }}
          DAGSHUB_SECRET_ACCESS_KEY: ${{ secrets.DAGSHUB_SECRET_ACCESS_KEY }}
        run: |
          uv run dvc pull --allow-missing
          uv run dvc status
          uv run dvc repro
          uv run dvc push
          git add dvc.lock
          git commit -m "Update dvc.lock on `date` with GitHub Actions" || true

      - name: Push changes
        uses: ad-m/github-push-action@master
        with:
          github_token: ${{ secrets.PAT_TOKEN }}
```

## Zagon

Za ponovni izracun cevovoda uporabi:

```bash
dvc repro
```

Za deljenje rezultatov po uspesni izvedbi:

```bash
git push
dvc push
```

Avtomatskega `git commit` in `git push` nismo vkljucili neposredno v `dvc.yaml`, ker sta to okoljsko odvisna koraka, ki poslabsata prenosljivost in ponovljivost cevovoda. Bolj varno je, da ostaneta locena od same obdelave podatkov in se izvajata v GitHub Actions workflow-u.
