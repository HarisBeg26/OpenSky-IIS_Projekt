# OpenSky DVC and Great Expectations Pipeline

Ta projekt uporablja podatkovni cevovod z DVC za zajem, predobdelavo in validacijo podatkov iz OpenSky Network API. Poleg osnovnega DVC toka je validacija nadgrajena z uporabo knjiznice Great Expectations, da lahko po vsakem zagonu avtomatsko preverimo kakovost obdelanih letalskih podatkov in zgradimo porocila Data Docs.

## Pricakovani rezultat

Razvit je ponovljiv cevovod, ki:

- zajame OpenSky posnetke v `data/raw`,
- iz vseh surovih posnetkov zgradi obdelane tabele v `data/processed`,
- pripravi kumulativno zgodovino `states_history.csv`,
- izvede validacijo nad obdelanimi podatki z Great Expectations,
- shrani validacijsko porocilo v `reports/validation/opensky_validation.json`,
- generira HTML porocila v `gx/uncommitted/data_docs/local_site`.

## Pomembna opomba o verzijah

Za ta del naloge uporabljamo `great-expectations==0.18.21`. Ta veja dokumentacije in API-ja je vezana na starejso verzijo GX in po uradni dokumentaciji podpira Python 3.8 do 3.11. Zato je projekt za del validacije prilagojen na Python 3.11, `numpy<2.0` in `pandas<3.0`.

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
  gx_dir: "gx"
  datasource_name: "opensky_processed"
  data_asset_name: "opensky_history"
  expectation_suite_name: "opensky_history_suite"
  checkpoint_name: "opensky_history_checkpoint"
  docs_site_name: "local_site"
```

Pomembna OpenSky prilagoditev je `fetch.bbox`, s katerim omejimo zajem na izbran del zracnega prostora. To je smiselna zamenjava za parameter `station` iz primera za ozracje.

## Faze DVC

Projekt uporablja tri glavne faze:

1. `fetch`
   Prenese najnovejsi OpenSky snapshot in ga shrani v `data/raw/states_<timestamp>.json`.

2. `preprocess`
   Prebere vse snapshot datoteke iz `data/raw`, vsako pretvori v tabelo `states_processed_<timestamp>.csv` in nato zgradi deterministicno zgodovinsko tabelo `states_history.csv`.

3. `validate`
   Pozene Great Expectations checkpoint nad `states_history.csv`, zgradi Data Docs in shrani JSON povzetek validacije.

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
       |
       v
 +----------+
 | validate |
 +----------+
```

## Great Expectations za OpenSky

Great Expectations uporabljamo za strukturirano validacijo obdelanih OpenSky podatkov. Namesto podatkov o kakovosti zraka tu preverjamo zgodovino letov `states_history.csv`, ki vsebuje stolpce, kot so `icao24`, `last_contact`, `longitude`, `latitude`, `velocity`, `true_track`, `snapshot_time_utc` in `source_snapshot`.

Skripta [gx/run_checkpoint.py](C:/Users/vunja/Desktop/Haris/Faks/Master/1.%20letnik/2.%20semester/IIS/Vaje/Projekt/OpenSky-IIS_Projekt/gx/run_checkpoint.py:1) ob zagonu:

- inicializira ali ponovno uporabi Filesystem Data Context v mapi `gx`,
- konfigurira Pandas filesystem datasource za obdelano zgodovino letov,
- ustvari oziroma posodobi expectation suite `opensky_history_suite`,
- ustvari oziroma posodobi checkpoint `opensky_history_checkpoint`,
- pozene checkpoint,
- zgradi Great Expectations Data Docs,
- shrani povzetek validacije v `reports/validation/opensky_validation.json`.

Tako se validacija lahko uporablja lokalno, v DVC cevovodu in v GitHub Actions.

## Smiselna validacijska pravila za OpenSky

Pri OpenSky projektu so primerna naslednja pricakovanja:

- tabela vsebuje vsaj eno vrstico,
- stolpci se ujemajo s pricakovano shemo obdelanega OpenSky nabora,
- `icao24` ni prazen in se ujema s 6-mestnim hex zapisom,
- `source_snapshot` sledi vzorcu `states_YYYYMMDDTHHMMSSZ.json`,
- `latitude` je med `-90` in `90`,
- `longitude` je med `-180` in `180`,
- `velocity` ni negativna in ostaja v realnem operativnem razponu,
- `true_track` je med `0` in `360`,
- `position_source` je eden izmed vrednosti `0`, `1`, `2`,
- kljucni casovni in identifikacijski stolpci niso manjkajoci.

To je OpenSky ekvivalent zahtevi, da mora validacija delovati za vsa merilna mesta. Pri nas to pomeni, da mora validacija delovati za vse zajete zrakoplove v vseh snapshot datotekah znotraj izbranega `bbox` obmocja.

## DVC validate faza

Korak `validate` v `dvc.yaml` je definiran tako:

```yaml
validate:
  cmd: uv run python gx/run_checkpoint.py
  deps:
    - gx/run_checkpoint.py
    - src/data/validate_opensky_data.py
    - data/processed
    - params.yaml
  params:
    - validate.processed_dir
    - validate.history_file
    - validate.report_path
    - validate.gx_dir
    - validate.datasource_name
    - validate.data_asset_name
    - validate.expectation_suite_name
    - validate.checkpoint_name
    - validate.docs_site_name
  outs:
    - gx/uncommitted:
        persist: true
  metrics:
    - reports/validation/opensky_validation.json
```

S tem poskrbimo, da se Great Expectations validacija pozene samo, ko pride do sprememb v obdelanih OpenSky podatkih ali v konfiguraciji validacije. Data Docs in validation results se ohranijo v `gx/uncommitted`, povzetek validacije pa je viden kot DVC metric.

## GitHub Actions in DVC

V GitHub Actions workflow-u smo tok prilagodili tako, da podpira tudi Great Expectations:

- workflow uporablja Python 3.11, ki je kompatibilen z GX 0.18.21,
- najprej se izvede `uv lock --python 3.11` in nato `uv sync --python 3.11 --locked`,
- nato `dvc pull --allow-missing`, da prvi zagon ne pade zaradi se neobstojecih artefaktov,
- `dvc repro` izvede celoten tok `fetch -> preprocess -> validate`,
- `dvc push` shrani DVC artefakte,
- `git add dvc.lock uv.lock` pripravi spremembe zaklepnih datotek za commit,
- po uspesni validaciji se Data Docs iz `gx/uncommitted/data_docs/local_site` objavijo na Netlify.

Aktualni workflow je v [fetch_data.yml](C:/Users/vunja/Desktop/Haris/Faks/Master/1.%20letnik/2.%20semester/IIS/Vaje/Projekt/OpenSky-IIS_Projekt/.github/workflows/fetch_data.yml:1).

## Netlify objava

Za samodejno objavo Great Expectations porocil workflow uporablja Netlify CLI in produkcijski deploy v ze obstojece Netlify mesto. V GitHub Secrets morata biti nastavljena:

- `NETLIFY_AUTH_TOKEN`
- `NETLIFY_SITE_ID`

Workflow po validaciji objavi vsebino mape `gx/uncommitted/data_docs/local_site` z ukazom:

```bash
netlify deploy --dir=gx/uncommitted/data_docs/local_site --prod
```

To pomeni, da se po vsakem uspesnem scheduled ali rocno sprozenem zagonu osvezi javno dostopna HTML dokumentacija Great Expectations.

## Zagon validacije

Validacijo lahko zazenes neposredno:

```bash
uv run python gx/run_checkpoint.py
```

Lahko pa pozenes celoten cevovod:

```bash
dvc repro
```

Po uspesni izvedbi lahko rezultate delis z:

```bash
git push
dvc push
```

## Naslednji korak

Logicen naslednji korak je objava generiranih Great Expectations Data Docs na Netlify, da bo porocilo dostopno tudi kot javna staticna stran po vsakem zagonu CI/CD toka.
