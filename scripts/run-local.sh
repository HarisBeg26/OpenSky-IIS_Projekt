#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PORT="${PORT:-8000}"

if [[ ! -f frontend/dist/index.html ]]; then
  (
    cd frontend
    npm ci
    npm run build
  )
fi

uv run python scripts/verify_local_setup.py --require-frontend
exec uv run uvicorn src.app.main:app --host 127.0.0.1 --port "$PORT"
