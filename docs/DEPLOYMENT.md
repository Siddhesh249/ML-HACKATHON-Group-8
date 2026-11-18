# Two-laptop deployment runbook (Laptop 1 = API, Laptop 2 = UI)

This is the exact procedure for the final evaluation. Rehearse it before the
demo and keep the API terminal visible: the evaluator must see the request
arrive and the score be computed server-side.

## 1. Roles

| Laptop | Runs | Must contain | Must not expose |
|---|---|---|---|
| 1 | FastAPI, model artifact, HMM, road networks, reference routes | repo, `data/evaluator_test/`, `artifacts/` | ground truth to the UI |
| 2 | Streamlit UI only | repo, `ui/`, `requirements.txt` | model, `.route`, data loaders |

## 2. Pre-flight (15 minutes before)

### Laptop 1

```bash
cd ML-HACKATHON-Group-8
source .venv/bin/activate
pip install -r requirements.txt

# The ten reserved evaluator trips must be here:
ls data/evaluator_test | wc -l          # expect 10
ls artifacts/model.joblib artifacts/hmm_config.json

python -m pytest -q                     # optional, 30 s
PATHSNAP_DATA_ROOT=./data python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --log-level info
```

Leave this terminal open and visible.

### Laptop 2

```bash
cd ML-HACKATHON-Group-8
source .venv/bin/activate
pip install -r requirements.txt
export PATHSNAP_API_URL=http://<LAPTOP1_IP>:8000
streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501
```

### Network

1. Both laptops on the same Wi-Fi or phone hotspot. Disable client isolation if
   the hotspot has it.
2. Find Laptop 1's IP: `hostname -I` (Linux) or `ipconfig` (Windows).
3. Test from Laptop 2:
   ```bash
   curl http://<LAPTOP1_IP>:8000/health
   ```
   Expect `{"status":"ok","ready":true,...}`.
4. If the firewall blocks the port, allow TCP 8000 on Laptop 1:
   - Linux (ufw): `sudo ufw allow 8000/tcp`
   - Windows: allow Python/uvicorn on the private network.

## 3. Demo flow

1. Evaluator opens `http://<LAPTOP2_IP>:8501` (or `localhost:8501` on Laptop 2).
2. UI shows the ten evaluation trips (`/trips` response; partition label visible).
3. Evaluator selects a trip and presses **Run prediction**.
4. On Laptop 1 the terminal shows:
   ```
   HTTP POST /predict status=200 ms=... rid=...
   PREDICT trip=00000000 points=... candidates=... mode=hmm top1=... latency_ms=... model=...
   ```
5. UI shows metric cards, the predicted route on a map, stage timings and the raw JSON.
6. Repeat for a second trip to prove it is not a single hardcoded example.

## 4. Ground-truth isolation checklist

- [ ] `GET /trips` returns IDs, counts and bboxes only — no route, no labels.
- [ ] `POST /predict` returns `metrics`, `predicted_route` and observed points —
      no `reference_route`, no per-observation truth.
- [ ] The prediction path opens `.track`, `.nodes`, `.arcs` only.
      `backend/service.py` calls `load_trip(..., with_route=False)`.
- [ ] Scoring happens after prediction in `pathsnap/scoring.py`.
- [ ] No `.route` file is present on Laptop 2.
- [ ] No hardcoded trip outputs; the API log proves live computation.

## 5. API reference

| Endpoint | Method | Notes |
|---|---|---|
| `/health` | GET | ready flag, model name/version, trips count, partition |
| `/version` | GET | API version, git commit, Python, model version |
| `/trips` | GET | ten evaluation trips, no ground truth |
| `/predict` | POST | `{"trip_id":"00000000","mode":"hmm"}` |

Response shape (abridged):

```json
{
  "trip_id": "00000000",
  "model_version": "…",
  "mode": "hmm",
  "metrics": {
    "point_top1": 0.0,
    "candidate_miss_rate": 0.0,
    "strict_continuity": 0.0,
    "lcs_f1": 0.0,
    "arc_f1": 0.0
  },
  "timings_ms": {"candidates_ms": 0, "features_ms": 0, "model_ms": 0, "decode_ms": 0, "scoring_ms": 0, "total_ms": 0},
  "predicted_route": [[8.59, 49.84]],
  "observed_points": [[8.59, 49.84]]
}
```

All numeric values above are placeholders; the live response contains measured
values. Do not quote these placeholders in the report.

## 6. Optional API key

Set `PATHSNAP_API_KEY` on Laptop 1 before starting. The UI must send
`X-API-Key`; use Streamlit secrets or an environment variable. Never commit the
key. For a closed demo network the key is optional, but the code path is tested.

## 7. Logging and evidence

The API logs at INFO:

```
2026-11-18 ... | HTTP POST /predict status=200 ms=1234.5 rid=1a2b3c4d client=...
2026-11-18 ... | PREDICT trip=00000000 points=1500 candidates=22500 mode=hmm top1=... latency_ms=... model=...
```

Capture:

1. Screenshot of the API terminal **after** the evaluator's request.
2. Screenshot of the UI showing the same trip ID, metrics and map.
3. Both into `results/screenshots/` and the report §7.

`GET /version` prints the git commit and model version so the evaluator can
verify the running code matches the repository.

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot reach ...` in the UI | wrong IP/port, firewall, different network | `curl /health` from Laptop 2; fix the URL; open the port |
| `/health` says `ready=false` | missing `artifacts/model.joblib` | run `make export`; check the path |
| `503 Model is not ready` | artifact failed validation (feature mismatch) | re-export with the current `config.FEATURE_NAMES` |
| `404 Trip not found` | evaluator trips not under `data/evaluator_test/` | place them there; set `PATHSNAP_EVALUATION_PARTITION` |
| `422` | trip ID is not eight digits | use the ID shown in the UI |
| Very slow first request | model/network warm-up, large trip | keep the process warm; show stage timings; retry once |
| UI shows the fallback partition | no evaluator trips present | label it clearly; this is for local demos only |
| CORS error | UI origin not allowed | set `PATHSNAP_CORS_ORIGINS` (or `*` for the demo) |

## 9. Contingencies

- **API crashes:** restart it; check the terminal traceback; never fake a result.
- **Network dies:** restart the hotspot; if it cannot be restored, run the UI on
  Laptop 1 via `localhost` and explain the local fallback. The two-laptop design
  is demonstrated by the logs and prior screenshots.
- **A trip fails:** capture the error, show the log, retry once. Report the
  affected trip honestly.
- **Model artifact missing:** stop the demo, run `make export`, verify
  `/health`, then continue.

## 10. Local smoke test (single laptop)

```bash
make test
make api &            # or scripts/07_run_api.sh
curl http://localhost:8000/health
curl http://localhost:8000/trips
curl -X POST http://localhost:8000/predict \
     -H 'Content-Type: application/json' \
     -d '{"trip_id":"00000000","mode":"hmm"}'
PATHSNAP_API_URL=http://localhost:8000 scripts/08_run_ui.sh
```

This is exactly the path the two-laptop demo uses, minus the network hop.
