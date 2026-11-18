# 7 — API, UI and deployment

Owner: **D**. Evidence below was produced against the frozen model artifact
(`hist_gradient_boosting`, version `6c229c610200`) on the ten student-test trips.
Optuna ran after this run and the team kept the untuned model (gain was inside
fold noise and did not survive student-test), so these deployment numbers stand.
Only C's decoder tuning can still change the HMM columns.

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

`mode` is `hmm` (default) or `classifier` for the report comparison.

## 7.3 Ground-truth isolation

- `Predictor._load_for_prediction()` calls `load_trip(..., with_route=False)`.
- `pathsnap/scoring.py` opens `.route` only after the arc sequence exists.
- The response contains aggregate metrics and predicted geometry, not the
  reference route or per-observation labels. Verified: the response keys are
  `trip_id, model_version, model_name, mode, metrics, timings_ms,
  n_observations, n_matched, predicted_route, observed_points, route_arc_count`.
- `tests/test_leakage.py` and `tests/test_api.py` enforce this (39 tests pass).

## 7.4 Evidence of live inference

**API terminal log** (actual excerpt; full file
`results/screenshots/api_log_excerpt.txt`):

```
2026-11-18 11:36:03,455 INFO pathsnap.predictor | Loaded artifact model=hist_gradient_boosting version=6c229c610200 features=18
2026-11-18 11:36:03,455 INFO pathsnap.api | PathSnap API starting version=0.1.0 ready=True model=6c229c610200
2026-11-18 11:36:16,786 INFO pathsnap.predictor | PREDICT trip=00000052 points=262 candidates=2701 mode=hmm top1=0.9923664122137404 continuity=0.8902439024390244 latency_ms=224.8 model=6c229c610200
2026-11-18 11:36:16,787 INFO pathsnap.api | HTTP POST /predict status=200 ms=226.4 rid=9486adcf client=127.0.0.1
2026-11-18 11:41:56,525 INFO pathsnap.predictor | PREDICT trip=00000056 points=3958 candidates=43649 mode=hmm top1=0.7803571428571429 continuity=0.8589928057553957 latency_ms=4059.7 model=6c229c610200
2026-11-18 11:41:57,870 INFO pathsnap.api | HTTP POST /predict status=200 ms=1318.0 rid=050f072b client=127.0.0.1
```

Rendered terminal image: `results/screenshots/api_terminal_log.png`.

**UI screenshots**: `results/screenshots/ui_dark_hmm_trip00000000.png` (HMM),
`ui_dark_classifier_trip00000000.png` (classifier-only),
`ui_dark_empty_state.png` (landing state), and `ui_dark_api_offline.png`
(error state when Laptop 1 is down). The client uses a dark theme
(`.streamlit/config.toml` + `ui/app.py`) and labels the `student_test` fallback
explicitly.

**Per-trip results**: `results/integration/run_log.csv` (20 rows = 10 trips ×
2 decoders), `per_trip_comparison.csv`, `stage_timings.csv`, `summary.json`.

### Integration summary (10 student-test trips)

| Mode | Trips passed | Top-1 mean | Continuity mean | LCS F1 mean | Arc F1 mean | Total p50 | Total p95 |
|---|---|---|---|---|---|---|---|
| classifier | 10/10 | 0.8995 | 0.7547 | 0.7723 | 0.8890 | 208 ms | 531 ms |
| HMM (tuned, T25 v2) | 10/10 | 0.9297 | 0.9057 | 0.9335 | 0.9383 | 563 ms | 1669 ms |

- HMM improves strict route continuity on **10/10** trips.
- With the T25 v2 decoder config the HMM is **never worse on Top-1** (0/10 trips) and
  raises the mean from 0.900 to 0.930; LCS F1 rises from 0.772 to 0.933.
- Worst API latency in the committed run log (`results/integration/run_log.csv`):
  `00000056` (3,958 points) at **1.85 s** for HMM and **0.62 s** for classifier-only.
- Every response was server-scored against `.route`; the UI received no
  reference geometry or labels.

## 7.5 Stage timings

Median / p95 across the ten HMM-trip runs (`results/integration/stage_timings.csv`):

| Stage | p50 (ms) | p95 (ms) | Max (ms) |
|---|---|---|---|
| Load | 2.6 | 6.7 | 7.9 |
| Candidates | 75.0 | 138.3 | 138.8 |
| Features | 91.3 | 250.0 | 293.4 |
| Model | 25.5 | 72.2 | 78.7 |
| Viterbi | 349.3 | 1124.1 | 1205.9 |
| Scoring | 16.9 | 91.8 | 140.1 |
| **Total** | **562.7** | **1668.9** | **1847.7** |

Classifier-only total: p50 438.3 ms, p95 1179.0 ms. The HMM decode stage is the
dominant cost on large trips.

## 7.6 Deployment checklist

- [x] Local two-process loopback (API on 8000, UI on 8501) — `/health` ok,
      `/predict` ok, UI screenshot captured.
- [x] All ten student-test trips listed and predicted (10/10, both modes).
- [x] API logs visible and captured; model version and request IDs recorded.
- [x] Ground-truth isolation verified; no `.route` in the response.
- [x] No hardcoded outputs; every result came through the live pipeline.
- [ ] Two physical laptops on the same network; `/health` reachable from
      Laptop 2 (needs the human rehearsal).
- [ ] The ten reserved evaluator trips placed under `data/evaluator_test/`.
- [ ] Re-run the HMM columns after C's decoder tuning; the classifier columns are
      final for the frozen artifact.
- [ ] Screenshots copied into the final PDF.

Runbook: [`docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md).
