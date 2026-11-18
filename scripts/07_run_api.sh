#!/usr/bin/env bash
# Laptop 1 — model + API + ground truth
set -euo pipefail
cd "$(dirname "$0")/.."

export PATHSNAP_DATA_ROOT="${PATHSNAP_DATA_ROOT:-./data}"
export PATHSNAP_ARTIFACT_DIR="${PATHSNAP_ARTIFACT_DIR:-./artifacts}"
PORT="${PATHSNAP_API_PORT:-8000}"

echo "PathSnap API -> http://0.0.0.0:${PORT}"
echo "data=${PATHSNAP_DATA_ROOT} artifacts=${PATHSNAP_ARTIFACT_DIR}"
exec python3 -m uvicorn backend.app:app --host 0.0.0.0 --port "${PORT}" --log-level info
