#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SKIP_DVC_PULL="${SKIP_DVC_PULL:-false}"
SKIP_FRONTEND_BUILD="${SKIP_FRONTEND_BUILD:-false}"

command -v uv >/dev/null 2>&1 || {
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
}

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env. Add DagsHub and HuggingFace credentials before continuing."
  if [[ "$SKIP_DVC_PULL" != "true" ]]; then
    echo "Edit .env, add DagsHub credentials, and run this script again." >&2
    exit 1
  fi
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

uv sync --python 3.11 --locked

if [[ "$SKIP_DVC_PULL" != "true" ]]; then
  if [[ -n "${DAGSHUB_ACCESS_KEY_ID:-}" && -n "${DAGSHUB_SECRET_ACCESS_KEY:-}" ]]; then
    uv run dvc remote add -d origin s3://dvc --force
    uv run dvc remote modify origin endpointurl \
      "${DAGSHUB_S3_ENDPOINT_URL:-https://dagshub.com/HarisBeg26/OpenSky-IIS_Projekt.s3}"
    uv run dvc remote modify origin --local access_key_id "$DAGSHUB_ACCESS_KEY_ID"
    uv run dvc remote modify origin --local secret_access_key "$DAGSHUB_SECRET_ACCESS_KEY"
  elif [[ ! -f .dvc/config.local ]]; then
    echo "DagsHub credentials are required for the first DVC pull." >&2
    exit 1
  fi
  uv run dvc pull --allow-missing --force
fi

VERIFY_ARGS=()
if [[ "$SKIP_FRONTEND_BUILD" != "true" ]]; then
  command -v npm >/dev/null 2>&1 || {
    echo "Node.js 22 LTS or newer is required: https://nodejs.org/" >&2
    exit 1
  }
  (
    cd frontend
    npm ci
    npm run build
  )
  VERIFY_ARGS+=(--require-frontend)
fi

uv run python scripts/verify_local_setup.py "${VERIFY_ARGS[@]}"
echo "Setup complete. Run ./scripts/run-local.sh and open http://127.0.0.1:8000"
