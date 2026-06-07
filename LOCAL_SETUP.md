# Lokalna vzpostavitev projekta SkyWatch

Ta dokument opisuje preverjen postopek od sveze kopije repozitorija do delujoce
aplikacije. Priporocena je produkcijsko podobna pot, kjer FastAPI servira tudi
zgrajeni React uporabniski vmesnik na enem naslovu.

## 1. Predpogoji

Namesti:

- Git,
- Python 3.11.x,
- [uv](https://docs.astral.sh/uv/getting-started/installation/),
- Node.js 22 LTS ali novejsi,
- Docker Desktop samo, ce zelis Docker Compose zagon.

Za dostop do verzioniranih podatkov in modelov potrebujes dostop do zasebnega
DagsHub repozitorija ter njegov S3 access key in secret key.

## 2. Sveza lokalna namestitev

Kloniraj repozitorij in odpri njegov korenski direktorij:

```powershell
git clone https://github.com/HarisBeg26/OpenSky-IIS_Projekt.git
Set-Location OpenSky-IIS_Projekt
```

### Windows PowerShell

```powershell
Copy-Item .env.example .env
notepad .env
.\scripts\setup-local.ps1
.\scripts\run-local.ps1
```

### Linux ali macOS

```bash
cp .env.example .env
nano .env
bash scripts/setup-local.sh
bash scripts/run-local.sh
```

V `.env` za prvi `dvc pull` nastavi vsaj:

```dotenv
DAGSHUB_ACCESS_KEY_ID=...
DAGSHUB_SECRET_ACCESS_KEY=...
```

Za delovanje zunanjega pretrained modela nastavi:

```dotenv
HF_TOKEN=hf_...
```

OpenSky uporabnisko ime in geslo sta opcijska. Brez njiju lahko javni API
deluje z omejitvami. MLflow poverilnice niso potrebne za lokalni zagon, ker se
lokalni eksperimenti shranjujejo v `mlruns/`.

Skripta izvede:

1. namestitev zaklenjenih Python odvisnosti,
2. lokalno konfiguracijo zasebnega DVC remote-a,
3. `dvc pull` podatkov, modelov in porocil,
4. `npm ci` in produkcijsko gradnjo React aplikacije,
5. preverjanje vseh obveznih lokalnih artefaktov.

Po zagonu odpri:

- aplikacija: <http://127.0.0.1:8000>,
- OpenAPI dokumentacija: <http://127.0.0.1:8000/docs>,
- health check: <http://127.0.0.1:8000/api/health>,
- administratorski API: <http://127.0.0.1:8000/api/admin/advanced>.

## 3. Docker Compose

Najprej izvedi setup, ker Docker sliki potrebujeta DVC modele, podatke in
porocila v gradbenem kontekstu:

```powershell
.\scripts\setup-local.ps1 -SkipFrontendBuild
docker compose up --build
```

Docker Compose zazene:

- `api` na <http://127.0.0.1:8000>,
- zasebni ONNX model service na <http://127.0.0.1:8001/health>.

Vrednosti `HF_TOKEN`, `HF_MODEL_ID` in `MODEL_SERVICE_TOKEN` se preberejo iz
lokalne `.env` datoteke. `.env` je izkljucen iz Git in Docker build konteksta.
Prav tako sta iz Docker slike izkljucena `.dvc/config.local` in lokalni
`mlruns/`, zato DagsHub in MLflow poverilnice niso vgrajene v sliko.

Ustavitev:

```powershell
docker compose down
```

## 4. Razvojni nacin z dvema procesoma

Backend:

```powershell
uv run uvicorn src.app.main:app --reload --port 8000
```

Frontend v drugem terminalu:

```powershell
Set-Location frontend
npm run dev
```

Nato odpri <http://127.0.0.1:5173>. Vite posreduje `/api` zahteve na FastAPI.

## 5. Preverjanje namestitve

```powershell
uv run python scripts/verify_local_setup.py --require-frontend
docker compose config --quiet
```

Python preverjanje ne izpise vrednosti skrivnosti. `HF_TOKEN` in DagsHub
poverilnice so prikazane samo kot `PASS` ali `WARN`. Uporabi `docker compose
config --quiet`, saj lahko navaden `docker compose config` izpise interpolirane
vrednosti iz `.env`.

## 6. Ponovitev podatkovnega in modelnega cevovoda

Celoten cevovod lahko traja dlje, ker ponovno zajame podatke in nauci modela:

```powershell
uv run dvc repro
```

Posamezne faze:

```powershell
uv run dvc repro preprocess
uv run dvc repro validate
uv run dvc repro test_data
uv run dvc repro train
uv run dvc repro batch_predict
uv run dvc repro monitor
```

Lokalni MLflow pregled po treningu:

```powershell
uv run mlflow ui --backend-store-uri mlruns --port 5000
```

## 7. Pogoste tezave

### `HF token required`

Dodaj `HF_TOKEN=hf_...` v korensko `.env` in ponovno zazeni FastAPI ali Docker
Compose proces.

### DVC vrne `AccessDenied`

Preveri DagsHub S3 access key in secret key v `.env`, nato ponovno zazeni
setup skripto. Skrivnosti se shranijo samo v ignorirano `.dvc/config.local`.

### Manjkajo modeli ali porocila

```powershell
uv run dvc pull --allow-missing --force
uv run python scripts/verify_local_setup.py
```

### Vrata 8000 so zasedena

```powershell
.\scripts\run-local.ps1 -Port 8010
```

### Python 3.14 ali druga nezdruzljiva verzija

Projekt zahteva Python 3.11 zaradi TensorFlow in Great Expectations 0.18.21.
Setup skripta zato uporablja `uv sync --python 3.11 --locked`.
