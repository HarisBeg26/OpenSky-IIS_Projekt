# OpenSky DVC, Great Expectations and Evidently Pipeline

Ta projekt uporablja podatkovni cevovod z DVC za zajem, predobdelavo, validacijo in testiranje OpenSky podatkov. Poleg osnovnega DVC toka sta vkljuceni dve dodatni plasti preverjanja kakovosti:

- Great Expectations za validacijo oblike in pravilnosti obdelanih podatkov.
- Evidently za zaznavanje sprememb v porazdelitvi podatkov med referencnim in trenutnim OpenSky snapshotom.

## Pricakovani rezultat

Razvit je ponovljiv cevovod, ki:

- zajame OpenSky posnetke v `data/raw`,
- iz vseh surovih posnetkov zgradi obdelane tabele v `data/processed`,
- pripravi kumulativno zgodovino `states_history.csv`,
- izvede Great Expectations validacijo nad obdelanimi podatki,
- izvede Evidently drift test med referencnim in trenutnim snapshotom,
- shrani Great Expectations porocilo v `reports/validation/opensky_validation.json`,
- shrani Evidently HTML porocilo v `reports/evidently/opensky_data_drift_report.html`,
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
```

## Faze DVC

Projekt ima stiri glavne faze:

1. `fetch`
   Prenese najnovejsi OpenSky snapshot in ga shrani v `data/raw/states_<timestamp>.json`.

2. `preprocess`
   Prebere vse snapshot datoteke iz `data/raw`, vsako pretvori v tabelo `states_processed_<timestamp>.csv` in zgradi `states_history.csv`.

3. `validate`
   Pozene Great Expectations checkpoint nad `states_history.csv`, zgradi Data Docs in shrani JSON povzetek validacije.

4. `test_data`
   Pozene Evidently drift test nad zadnjim obdelanim snapshotom in referencnim snapshotom.

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
```

## Great Expectations za OpenSky

Great Expectations uporabljamo za strukturirano validacijo obdelane zgodovine letov `states_history.csv`. Preverjamo shemo podatkov, prisotnost kljucnih stolpcev, osnovne obsege vrednosti in konzistentnost casovnih ter identifikacijskih polj.

Skripta [gx/run_checkpoint.py](C:/Users/vunja/Desktop/Haris/Faks/Master/1.%20letnik/2.%20semester/IIS/Vaje/Projekt/OpenSky-IIS_Projekt/gx/run_checkpoint.py:1):

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

Skripta [test_opensky_data.py](C:/Users/vunja/Desktop/Haris/Faks/Master/1.%20letnik/2.%20semester/IIS/Vaje/Projekt/OpenSky-IIS_Projekt/src/data/test_opensky_data.py:1) izvede:

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
    - test_data.drop_columns
  outs:
    - data/reference:
        persist: true
    - reports/evidently:
        persist: true
```

S tem poskrbimo, da se drift test pozene le ob spremembah obdelanih OpenSky snapshotov ali konfiguracije testiranja. Referencni snapshot se ob uspesnem testu osvezi, Evidently porocila pa se ohranijo v `reports/evidently`.

## GitHub Actions in DVC

GitHub Actions workflow uporablja Python 3.11, regenerira `uv.lock`, izvede `uv sync --locked`, nato pa z `dvc repro` pozene celoten tok:

`fetch -> preprocess -> validate -> test_data`

Po tem:

- `dvc push` shrani DVC artefakte,
- `git add dvc.lock uv.lock` pripravi zaklepne datoteke,
- Great Expectations Data Docs se po uspesni validaciji lahko objavijo na Netlify.

Aktualni workflow je v [fetch_data.yml](C:/Users/vunja/Desktop/Haris/Faks/Master/1.%20letnik/2.%20semester/IIS/Vaje/Projekt/OpenSky-IIS_Projekt/.github/workflows/fetch_data.yml:1).

## Zagon

Great Expectations validacijo lahko zazenes neposredno:

```bash
uv run python gx/run_checkpoint.py
```

Evidently drift test lahko zazenes neposredno:

```bash
uv run python src/data/test_opensky_data.py
```

Ali pa pozenes celoten cevovod:

```bash
dvc repro
```

Po uspesni izvedbi lahko rezultate delis z:

```bash
git push
dvc push
```
