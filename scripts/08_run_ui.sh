#!/usr/bin/env bash
# Laptop 2 — thin evaluator UI
set -euo pipefail
cd "$(dirname "$0")/.."

export PATHSNAP_API_URL="${PATHSNAP_API_URL:-http://localhost:8000}"
echo "PathSnap UI -> API at ${PATHSNAP_API_URL}"
exec python3 -m streamlit run ui/app.py --server.address 0.0.0.0 --server.port "${PATHSNAP_UI_PORT:-8501}"
