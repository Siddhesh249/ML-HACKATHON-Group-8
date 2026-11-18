# 7 — API, UI and deployment

Owner: **D**. This section is filled with real screenshots and log excerpts
after the integration run. Placeholders below are not results.

## 7.1 Architecture

Laptop 1 hosts the model artifact, candidate/feature pipeline, HMM/Viterbi
decoder, road networks and the server-side reference data. Laptop 2 runs a thin
Streamlit client that talks to the FastAPI service over HTTP. The UI never
receives the reference route or per-observation labels.

```
Laptop 2 (Streamlit)  ──HTTP──►  Laptop 1 (FastAPI + model + HMM + ground truth)
   trip list, metrics, map         /health /version /trips /predict
```

## 7.2 API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | readiness, model name/version, trip count, partition |
| `/version` | GET | API version, git commit, Python, model version |
| `/trips` | GET | the ten evaluation trips (metadata only) |
| `/predict` | POST | run the full pipeline and score server-side |

Request:

```json
{"trip_id": "00000000", "mode": "hmm"}
```

## 7.3 Ground-truth isolation

- `Predictor._load_for_prediction()` calls `load_trip(..., with_route=False)`.
- `pathsnap/scoring.py` opens `.route` only after the arc sequence exists.
- The response contains aggregate metrics and predicted geometry, not the
  reference route or per-observation labels.
- `tests/test_leakage.py` enforces this.

## 7.4 Evidence of live inference

**API terminal log** (replace with the actual captured log):

```
TBD — paste the real log line from Laptop 1, including:
HTTP POST /predict status=200 ms=... rid=...
PREDICT trip=... points=... candidates=... mode=hmm top1=... latency_ms=... model=...
```

**UI screenshot**:

```
TBD — results/screenshots/ui_prediction_<trip>.png
```

**Per-trip results**:

```
TBD — results/integration/run_log.csv
```

## 7.5 Stage timings

| Stage | p50 (ms) | p95 (ms) | Source |
|---|---|---|---|
| Candidates | TBD | TBD | `results/integration/run_log.csv` |
| Features | TBD | TBD | same |
| Model | TBD | TBD | same |
| Viterbi | TBD | TBD | same |
| Scoring | TBD | TBD | same |
| Total | TBD | TBD | same |

## 7.6 Deployment checklist

- [ ] Two laptops on the same network; `/health` reachable from Laptop 2.
- [ ] All ten evaluator trips listed in the UI.
- [ ] Prediction works for every trip, not only the demo trip.
- [ ] API logs visible and captured for the report.
- [ ] No hardcoded outputs; no `.route` on Laptop 2.
- [ ] Screenshots saved under `results/screenshots/`.
- [ ] The source in the repo matches the running code (git commit from `/version`).
