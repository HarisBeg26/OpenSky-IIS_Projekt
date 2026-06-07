# OpenSky DVC, Great Expectations and Evidently Pipeline

Ta projekt uporablja podatkovni cevovod z DVC za zajem, predobdelavo, validacijo in testiranje OpenSky podatkov. Poleg osnovnega DVC toka sta vkljuceni dve dodatni plasti preverjanja kakovosti:

- Great Expectations za validacijo oblike in pravilnosti obdelanih podatkov.
- Evidently za zaznavanje sprememb v porazdelitvi podatkov med referencnim in trenutnim OpenSky snapshotom.

## Hitra lokalna vzpostavitev

Projekt vsebuje vse skripte, konfiguracije in zaklepne datoteke za vzpostavitev
iz sveze kopije repozitorija. Potrebni veliki podatki, modeli in porocila so
verzionirani z DVC in se prenesejo iz zasebnega DagsHub remote-a.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
notepad .env
.\scripts\setup-local.ps1
.\scripts\run-local.ps1
```

Linux ali macOS:

```bash
cp .env.example .env
bash scripts/setup-local.sh
bash scripts/run-local.sh
```

Po zagonu je aplikacija na <http://127.0.0.1:8000>, API dokumentacija pa na
<http://127.0.0.1:8000/docs>.

Za prvi `dvc pull` sta v `.env` potrebna `DAGSHUB_ACCESS_KEY_ID` in
`DAGSHUB_SECRET_ACCESS_KEY`. Za zunanji HuggingFace model je potreben
`HF_TOKEN`. Nobena skrivnost ni shranjena v Git.

Celotna navodila, Docker Compose pot, preverjanje namestitve in odpravljanje
tezav so v [LOCAL_SETUP.md](LOCAL_SETUP.md).

## Pricakovani rezultat

Razvit je ponovljiv cevovod, ki:

- zajame OpenSky posnetke v `data/raw`,
- iz vseh surovih posnetkov zgradi obdelane tabele v `data/processed`,
- pripravi kumulativno zgodovino `states_history.csv`,
- izvede Great Expectations validacijo nad obdelanimi podatki,
- izvede Evidently drift test med referencnim in trenutnim snapshotom,
- nauci dva napovedna modela z nevronskimi mrezami,
- pripravi produkcijsko monitoring porocilo modelov,
- servira uporabniski in administratorski vmesnik prek FastAPI,
- zgradi Docker sliko za namestitev v produkcijo,
- shrani Great Expectations porocilo v `reports/validation/opensky_validation.json`,
- shrani Evidently HTML porocilo v `reports/evidently/opensky_data_drift_report.html`,
- shrani metrike ucenja v `reports/model_training/opensky_metrics.json`,
- ob uspesnem Evidently testu posodobi referencni snapshot v `data/reference/opensky`.

## Pomembna opomba o verzijah

Projekt za del validacije in testiranja uporablja Python 3.11. To je pomembno zaradi kompatibilnosti z `great-expectations==0.18.21`, medtem ko Evidently uporabljamo v razponu `>=0.7.1,<0.8.0`, kar je skladno z uradno dokumentacijo in PyPI objavami za vejo 0.7.

Uporabni viri:

- [Great Expectations 0.18.21 na PyPI](https://pypi.org/project/great-expectations/0.18.21/)
- [Evidently na PyPI](https://pypi.org/project/evidently/)
- [Evidently Report docs](https://docs.evidentlyai.com/docs/library/report)
- [Evidently Data Drift preset](https://docs.evidentlyai.com/metrics/preset_data_drift)
- [Evidently Data Summary preset](https://docs.evidentlyai.com/metrics/preset_data_summary)

## Parametri cevovoda

V `params.yaml` so zbrani vsi kljucni parametri:

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
  gx_dir: "gx"
  datasource_name: "opensky_processed"
  data_asset_name: "opensky_history"
  expectation_suite_name: "opensky_history_suite"
  checkpoint_name: "opensky_history_checkpoint"
  docs_site_name: "local_site"

test_data:
  processed_dir: "data/processed"
  reference_dir: "data/reference/opensky"
  reference_file: "states_reference.csv"
  report_html: "reports/evidently/opensky_data_drift_report.html"
  report_json: "reports/evidently/opensky_data_drift_summary.json"
  min_rows: 30
  drift_share: 0.7
  drop_columns:
    - "icao24"
    - "callsign"
    - "time_position"
    - "last_contact"
    - "sensors"
    - "squawk"
    - "snapshot_time"
    - "snapshot_time_utc"
    - "source_snapshot"
    - "event_time_utc"
    - "captured_at_utc"

train:
  history_file: "data/processed/states_history.csv"
  models_dir: "models/opensky"
  metrics_path: "reports/model_training/opensky_metrics.json"
  mlflow_tracking_uri: "https://dagshub.com/HarisBeg26/OpenSky-IIS_Projekt.mlflow"
  mlflow_experiment_name: "OpenSky-IIS_Projekt_train"
  test_size: 0.2
  max_sequences: 12000
  window_size: 2
  random_state: 42
  lstm_units: 64
  dense_units: 32
  dropout: 0.2
  epochs: 20
  batch_size: 64
  validation_split: 0.2
  patience: 4
  onnx_opset: 13
  feature_columns:
    - "longitude"
    - "latitude"
    - "baro_altitude"
    - "velocity"
    - "true_track"
    - "vertical_rate"
    - "geo_altitude"
    - "on_ground"
    - "spi"
    - "position_source"

monitoring:
  report_path: "reports/model_monitoring/production_model_monitoring.json"
  processed_dir: "data/processed"
  models_dir: "models/opensky"
  training_metrics: "reports/model_training/opensky_metrics.json"
  max_trajectory_mae: 5.0
  min_on_ground_accuracy: 0.85
  max_data_age_hours: 72
```

## Faze DVC

Projekt ima sest glavnih faz:

1. `fetch`
   Prenese najnovejsi OpenSky snapshot in ga shrani v `data/raw/states_<timestamp>.json`.

2. `preprocess`
   Prebere vse snapshot datoteke iz `data/raw`, vsako pretvori v tabelo `states_processed_<timestamp>.csv` in zgradi `states_history.csv`.

3. `validate`
   Pozene Great Expectations checkpoint nad `states_history.csv`, zgradi Data Docs in shrani JSON povzetek validacije.

4. `test_data`
   Pozene Evidently drift test nad zadnjim obdelanim snapshotom in referencnim snapshotom.

5. `train`
   Nauci dva napovedna modela nad `states_history.csv` in shrani modele ter metrike.

6. `monitor`
   Ustvari produkcijsko porocilo o stanju modelov, podatkov in metrik.

Vizualno je tok naslednji:

```text
            +-------+
            | fetch |
            +-------+
                 |
                 v
          +------------+
          | preprocess |
          +------------+
            /        \
           v          v
   +----------+   +-----------+
   | validate |   | test_data |
   +----------+   +-----------+
           \       /
            v     v
          +---------+
          |  train  |
          +---------+
               |
               v
          +---------+
          | monitor |
          +---------+
```

## Great Expectations za OpenSky

Great Expectations uporabljamo za strukturirano validacijo obdelane zgodovine letov `states_history.csv`. Preverjamo shemo podatkov, prisotnost kljucnih stolpcev, osnovne obsege vrednosti in konzistentnost casovnih ter identifikacijskih polj.

Skripta [gx/run_checkpoint.py](gx/run_checkpoint.py):

- inicializira ali ponovno uporabi Filesystem Data Context v mapi `gx`,
- konfigurira Pandas filesystem datasource za obdelano zgodovino letov,
- ustvari oziroma posodobi expectation suite in checkpoint,
- pozene checkpoint,
- zgradi Great Expectations Data Docs,
- shrani povzetek validacije v `reports/validation/opensky_validation.json`.

## Evidently za OpenSky

Evidently uporabljamo za preverjanje data drift med dvema OpenSky snapshotoma:

- `current` je najnovejsi obdelani snapshot `states_processed_*.csv` ali `states_processed_*.parquet`,
- `reference` je zadnji uspesno potrjen referencni snapshot v `data/reference/opensky/states_reference.csv`.

Ob prvem zagonu referencni snapshot se ne obstaja, zato se trenutni snapshot uporabi tudi kot referencni. Pri naslednjih zagonih se novi trenutni snapshot primerja s shranjenim referencnim snapshotom.

Ker nekateri OpenSky stolpci vsebujejo identifikatorje ali casovne oznake, ki skoraj vedno driftajo in niso koristni za primerjavo porazdelitev, jih pred testiranjem izpustimo. To so na primer `icao24`, `callsign`, `time_position`, `last_contact`, `snapshot_time`, `source_snapshot` in sorodna casovna polja.

Ker so posamezni OpenSky snapshoti v izbranem `bbox` obmocju lahko zelo majhni, je drift gate prilagojen na dva nacina:

- strogo fail/passed odlocanje vklopimo sele, ko imata tako `reference` kot `current` vsaj `test_data.min_rows` vrstic,
- pri dovolj velikem vzorcu stage ne gleda vseh pomoznih testov iz `DataSummaryPreset`, ampak dataset-level drift signal z `drift_share`.

Uporabljamo prag `test_data.drift_share: 0.7`, kar pomeni, da se dataset drift obravnava kot kriticen sele, ko drift zaznamo pri vec kot 70 % primerjanih stolpcev. To sledi tudi Evidently dokumentaciji, kjer je privzeti dataset-level drift prag 50 %, vendar ga lahko za konkreten primer prilagodimo.

Skripta [test_opensky_data.py](src/data/test_opensky_data.py) izvede:

- nalaganje trenutnega in referencnega snapshota,
- uskladitev skupnih stolpcev za primerjavo,
- odstranitev neprimernih ali praznih stolpcev,
- zagon `Report([DataSummaryPreset(), DataDriftPreset()], include_tests=True)`,
- shranjevanje HTML porocila,
- shranjevanje JSON povzetka rezultatov,
- posodobitev referencnega snapshota samo v primeru uspesnega testa.

To je OpenSky ekvivalent zahtevi, da mora skripta delovati za vsa merilna mesta. Pri nas to pomeni, da mora drift test delovati za vse zajete zrakoplove v vseh snapshotih znotraj izbranega `bbox` obmocja.

## DVC validate in test_data fazi

`validate` faza je definirana tako:

```yaml
validate:
  cmd: uv run python gx/run_checkpoint.py
  deps:
    - gx/run_checkpoint.py
    - src/data/validate_opensky_data.py
    - data/processed
    - params.yaml
  outs:
    - gx/uncommitted:
        persist: true
  metrics:
    - reports/validation/opensky_validation.json
```

`test_data` faza je definirana tako:

```yaml
test_data:
  cmd: uv run python src/data/test_opensky_data.py
  deps:
    - src/data/test_opensky_data.py
    - data/processed
    - params.yaml
  params:
    - test_data.processed_dir
    - test_data.reference_dir
    - test_data.reference_file
    - test_data.report_html
    - test_data.report_json
    - test_data.min_rows
    - test_data.drift_share
    - test_data.drop_columns
  outs:
    - data/reference:
        persist: true
    - reports/evidently:
        persist: true
```

S tem poskrbimo, da se drift test pozene le ob spremembah obdelanih OpenSky snapshotov ali konfiguracije testiranja. Referencni snapshot se ob uspesnem testu osvezi, Evidently porocila pa se ohranijo v `reports/evidently`.

## Ucenje napovednih modelov za OpenSky

Za OpenSky imamo dve razlicni napovedni nalogi, obe izvedeni z rekurentnimi nevronskimi mrezami v TensorFlow/Keras:

- `trajectory_lstm` uporablja LSTM regresijski model in napoveduje naslednjo pozicijo letala, torej `target_next_latitude` in `target_next_longitude`.
- `on_ground_lstm` uporablja LSTM klasifikacijski model in napoveduje, ali bo letalo v naslednjem stanju na tleh (`target_next_on_ground`).

Obe nalogi uporabljata zgodovinske OpenSky zapise, zdruzene po `icao24` in urejene po casu. Skripta [train.py](src/model/train.py) iz posameznih letal ustvari drseca zaporedja oblike "zadnjih N stanj -> naslednje stanje", nato izvede casovni train/test razcep. Modela sta shranjena kot `.keras` in `.onnx` datoteki v `models/opensky`, predprocesorji pa v `models/opensky/preprocessors.pkl`. Metrike so zapisane v `reports/model_training/opensky_metrics.json`.

`train` faza je definirana tako:

```yaml
train:
  cmd: uv run python main.py train
  deps:
    - main.py
    - src/model/preprocess.py
    - src/model/train.py
    - data/processed
    - gx/uncommitted
    - reports/validation/opensky_validation.json
    - data/reference
    - reports/evidently/opensky_data_drift_summary.json
    - params.yaml
  outs:
    - models/opensky:
        persist: true
  metrics:
    - reports/model_training/opensky_metrics.json
```

S tem trening stece po predobdelavi podatkov in se ponovno izvede ob spremembi podatkov, parametrov ali modelne kode. V celotnem GitHub Actions toku se pred treningom izvedejo tudi Great Expectations in Evidently preverjanja.

Ker trening uporablja TensorFlow, po posodobitvi odvisnosti najprej osvezi okolje z:

```bash
uv sync
```

Trening uporablja MLflow za sledenje eksperimentom na DagsHub:

- tracking URI je nastavljen v `train.mlflow_tracking_uri`,
- eksperiment je nastavljen v `train.mlflow_experiment_name`,
- `train.mlflow_mode: auto` lokalno uporabi mapo `mlruns`, v GitHub Actions pa DagsHub,
- vsak zagon shrani parametre, metrike in artefakte modelov,
- oba Keras modela se dodatno serializirata v ONNX format z `train.onnx_opset`,
- modeli so hkrati verzionirani z DVC kot izhod `models/opensky`.

Za GitHub Actions morata biti nastavljeni skrivnosti:

- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`

Pri DagsHub je `MLFLOW_TRACKING_USERNAME` obicajno uporabnisko ime, `MLFLOW_TRACKING_PASSWORD` pa DagsHub token.

Za lokalno ucenje poverilnice niso potrebne. Eksperimente lahko po treningu pregledas z:

```bash
uv run mlflow ui --backend-store-uri mlruns
```

Nato odpri `http://127.0.0.1:5000`.

## Produkcijsko nadzorovanje modelov

Skripta [monitor_models.py](src/monitoring/monitor_models.py) pripravi porocilo `reports/model_monitoring/production_model_monitoring.json`. V njem preverimo:

- ali so prisotni pricakovani produkcijski artefakti modelov,
- ali so metrike modelov znotraj pragov,
- ali obstaja svezi obdelani OpenSky snapshot,
- kaksno je skupno stanje modelov v produkciji (`ok`, `warn`, `fail`).

Monitoring lahko zazenes z:

```bash
uv run python main.py monitor
```

## Uporabniski in administratorski vmesnik

Uporabniski vmesnik je locena React aplikacija v `frontend/`, FastAPI pa ostane inteligentni API servis v [src/app/main.py](src/app/main.py).

FastAPI ponuja:

- `/api/intelligence/briefing` operativni briefing, prioritetno vrsto in model readiness z uporabnisko nastavljivimi pragi,
- `/api/predictions/{icao24}` napoved naslednje pozicije in verjetnosti `on_ground`, shadow primerjavo in opcijsko HuggingFace zero-shot oceno tveganja,
- `/api/flights` zadnje obdelane OpenSky zapise z razlago operativnega signala,
- `/api/admin/summary` zdruzen pregled validacije, drifta, metrik in modelnih artefaktov,
- `/api/admin/advanced` razsirjen administratorski pogled nad quality gates, MLflow sledenjem, registrom modelov in report linki,
- `/api/admin/model-registry/{model_key}/stage` lokalno upravljanje faz modela (`Candidate`, `Staging`, `Production`, `Archived`).

React UI prikaze inteligentno izkusnjo: uporabnik izbere zrakoplov, vidi razloge za opozorilo, priporocen ukrep, heuristicno projekcijo ter napoved iz naucenih modelov. Uporabnik lahko prilagodi prag nizke visine, hitrosti spuscanja, visoke hitrosti in minimalnega attention score; nastavitve se shranijo lokalno v brskalniku in takoj vplivajo na prioritetno vrsto.

Poleg dveh lastno naucenih nevronskih mrez projekt vkljucuje tudi obstojeci nauceni model `facebook/bart-large-mnli` iz HuggingFace za zero-shot klasifikacijo tekstovnega opisa tveganja leta. Integracija je privzeto vklopljena. Lokalno ustvari datoteko `.env` po vzoru `.env.example` in dodaj `HF_TOKEN` z dovoljenjem Inference Providers; pri Render Blueprint namestitvi se `HF_TOKEN` vnese kot skrivnost. Token se nikoli ne shrani v Git. Ce token manjka, osnovna aplikacija se vedno deluje, administratorska kartica pa jasno prikaze stanje `configuration_required`.

Razsirjena administratorska plosca zdruzuje:

- rezultate Great Expectations validacije in Evidently drift testiranja,
- ovrednotenje modelov v produkciji,
- MLflow experiment tracking in podatke o registriranih modelih,
- lokalni lifecycle nadzor modelov za prikaz migracije med fazami,
- shadow testing primerjavo med LSTM trajektorijo in kinematicnim baseline modelom,
- globalno permutacijsko pomembnost znacilk z baseline metrikami in opozorili za varno interpretacijo,
- povezave do HTML/JSON porocil, kadar so artefakti prisotni po `dvc pull`.

Prikaz razlagalnosti ni SHAP in ne prikazuje vzrocnosti ali smeri vpliva. Vsaka znacilka se premesa cez testne primere, nato pa se izmeri poslabsanje rezultata. Daljsi stolpec zato pomeni vecjo odvisnost modela od znacilke, ne pa boljse znacilke ali boljsega modela. Posebej mocna odvisnost trajektorije od `latitude` ali `longitude` je opozorilo za mozno geografsko pomnjenje in zahteva preverjanje na nevidenih zrakoplovih oziroma regijah.

Lokalni backend:

```bash
uv run uvicorn src.app.main:app --reload
```

Lokalni frontend:

```bash
cd frontend
npm install
npm run dev
```

Produkcijski build frontenda:

```bash
cd frontend
npm run build
```

Ko `frontend/dist` obstaja, ga FastAPI servira na `/`.

## Docker in namestitev v produkcijo

Produkcijska namestitev uporablja brezplacni hibridni arhitekturni vzorec:

- en Render Free `skywatch-api` servis zdruzuje React, FastAPI in ONNX Runtime,
- online napovedi se izvajajo kot model-as-dependency znotraj javnega API servisa,
- DVC faza `batch_predict` pripravi paketne napovedi kot dodatno odporno pot,
- loceni `skywatch-model-service` ostaja implementiran za lokalni Docker Compose prikaz vzorca Model as a Service.

S tem produkcija brez stroskov demonstrira online model-as-dependency, batch/offline napovedovanje in hibridno kombinacijo obeh pristopov. Loceni Model as a Service je dodatno dokazljiv lokalno. Shadow testing ostaja pristop testiranja modela in ni napacno prikazan kot arhitekturni vzorec namestitve.

Lokalni Docker zagon:

```bash
docker compose up --build
```

Po zagonu sta na voljo:

- uporabniski vmesnik in javni API na `http://127.0.0.1:8000`,
- zasebni modelni servis za lokalno preverjanje na `http://127.0.0.1:8001/health`.

### Render

Datoteka `render.yaml` definira en brezplacni spletni servis `skywatch-api` v regiji Frankfurt. Servis se lahko ob neaktivnosti ustavi, zato je prvi odziv po daljsem premoru lahko pocasnejsi.

1. V GitHub nastavitvah ustvari Personal Access Token z dovoljenjem `read:packages`.
2. V Render `Workspace Settings > Container Registry Credentials` dodaj GHCR poverilnico z imenom `github-container-registry`.
3. V Render izberi `New > Blueprint`, povezi repozitorij in uporabi korensko datoteko `render.yaml`.
4. Ob ustvarjanju servisa vnesi skrivnost `HF_TOKEN`.
5. V servisu `skywatch-api` kopiraj Deploy Hook URL.
6. V GitHub Actions secrets dodaj samo `RENDER_API_DEPLOY_HOOK_URL`.

GitHub Actions workflow `.github/workflows/docker.yml` po uspesnem podatkovnem cevovodu:

- prenese DVC artefakte,
- zgradi in objavi `api-latest` sliko v GHCR,
- sprozi namestitev brezplacnega javnega API servisa.

Render ne potrebuje `MODEL_SERVICE_TOKEN`, ker ONNX modela izvaja znotraj istega vsebnika. Loceni modelni servis in token se uporabljata samo pri lokalnem `docker compose up --build` preizkusu.

## GitHub Actions in DVC

GitHub Actions workflow uporablja Python 3.11, regenerira `uv.lock`, izvede `uv sync --locked`, nato pa z `dvc repro` pozene celoten tok:

`fetch -> preprocess -> validate + test_data -> train -> batch_predict -> monitor`

Po tem:

- `dvc push` shrani DVC artefakte,
- `git add dvc.lock uv.lock` pripravi zaklepne datoteke,
- metricni JSON-i za validacijo, trening in monitoring ostanejo DVC outputs/metrics in se ne dodajajo neposredno v Git,
- Great Expectations Data Docs se po uspesni validaciji lahko objavijo na Netlify.

Zajem OpenSky podatkov uporablja vec ponovnih poskusov in svezi DVC fallback. Fallback na obstojec raw snapshot je dovoljen samo, ce je snapshot mlajsi od `fetch.max_cached_age_hours`; s tem pipeline ne nadaljuje tiho na prestarih podatkih.

Aktualni workflow je v [fetch_data.yml](.github/workflows/fetch_data.yml).

## Zagon

Great Expectations validacijo lahko zazenes neposredno:

```bash
uv run python gx/run_checkpoint.py
```

Evidently drift test lahko zazenes neposredno:

```bash
uv run python src/data/test_opensky_data.py
```

Napovedne modele lahko naucis neposredno:

```bash
uv run python main.py train
```

Ali pa pozenes celoten cevovod:

```bash
uv run dvc repro
```

Po uspesni izvedbi lahko rezultate delis z:

```bash
git push
uv run dvc push
```
