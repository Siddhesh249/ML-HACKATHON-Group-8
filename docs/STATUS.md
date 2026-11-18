# Project status — live state board

> **Read this second, after `AGENTS.md`.** This is the authoritative “what is
> true now”. Keep edits small and localised. Remove your claim when you finish.
> Newest handoff at the bottom of the file.

**Last updated:** 2026-11-18 13:45
**Updated by:** D (phase-1 close-out)
**Phase:** **Phase 1 complete** (T10–T21); Phase 2 begins at T22 (Optuna, decoder tuning, report, two-laptop, hardening).
**Current focus:** freeze Phase 1 evidence and tag `v0.1-phase1`; next is B's Optuna, C's decoder tuning, and D's physical rehearsal + final numbers.

---

## 1. Health

| Area | State | Evidence / note |
|---|---|---|
| Unit/contract tests | ✅ 39 passed | `make test`; `httpx` added to `requirements.txt` for Starlette `TestClient` |
| Package imports | ✅ | `make smoke` |
| Repo scaffold | ✅ | library, API, UI, scripts, docs, tests |
| EDA | ✅ T10 | 90 trips; `results/eda/inventory.csv`, `summary.json`, `quality_checks.csv`, `flags.csv`, figures; narrative `docs/DATA_NOTES.md` |
| Candidate generator | ✅ validated (T11) | sweep: production 60 m + fallbacks, K=15 → miss 0.092% train / 0.032% student-test; `results/eda/candidate_sweep.csv` |
| Route labels/alignment | ✅ validated (T12) | train 183,052/208,417 labelled; unreliable train `00000033`, `00000054`; `results/labels/coverage.csv`, 5 visual checks |
| Dataset cache | ✅ built (local) | 80 train trips, 1,732,459 rows; evidence `results/models/dataset_summary.json`, `cv_splits.csv` |
| Model comparison | ✅ Phase 1 | 8 families + baseline, GroupKFold(5): `results/models/model_comparison.csv` |
| Selected model | ✅ preliminary | HistGradientBoosting, untuned, CV Top-1 0.887: `results/models/selected_model.json` |
| Optuna study | ✅ T23/T24 | 40 trials, best CV Top-1 0.8872 vs 0.8868 untuned; **decision: keep untuned** → artifact unchanged — `results/optuna/`, `results/models/tuned_vs_untuned_decision.json` |
| Inference artifact | ✅ preliminary | `artifacts/model.joblib` v`6c229c610200`; `Predictor` ready=true, live predict verified on 1 student-test trip — **D can integrate** |
| FastAPI | ✅ live (local) | `/health` ready, `/trips` 10 trips, live `/predict` on small + large trips; log excerpt `results/screenshots/api_log_excerpt.txt` |
| Streamlit UI | ✅ dark theme (local) | screenshots `results/screenshots/ui_dark_hmm_trip00000000.png`, `ui_dark_classifier_trip00000000.png`, `ui_dark_empty_state.png`; fallback banner + route layer + legend |
| HMM decoder | ✅ Phase 1 complete | tests + real-trip benchmark (`results/hmm/transition_benchmark.csv`); tuning + classifier-vs-HMM are Phase 2 |
| Integration (10 trips) | ✅ 10/10 both modes | `results/integration/run_log.csv`, `summary.json`, `stage_timings.csv` |
| Two-laptop demo | 🟨 local loopback done | physical Laptop 2 rehearsal + evaluator trips pending |
| Viva notes | ✅ simplified (1000 → 350 lines) | `viva.md`: task/data/preprocessing/features, models, HMM, API, ledger, rapid-fire Q&A, demo; maintenance rules in §14 |

**A/B/C Phase-1 evidence and B’s preliminary artifact are committed.** D’s
integration numbers use the untuned artifact + default HMM config; refresh them
after B’s Optuna export and C’s decoder tuning. Quote only from `results/`.

---

## 2. Active claims

Add a row **before** editing an area. Areas follow `docs/ASSIGNMENTS.md` §1.

| Area / files | Agent | Branch | Since | Status |
|---|---|---|---|---|
| `backend/`, `ui/`, `docs/DEPLOYMENT.md`, integration | D | `feat/d-integration` | 2026-11-18 10:20 | active — local integration done, physical laptop pending |
| `pathsnap/io.py`, `geo.py`, `network.py`, `candidates.py`, `labels.py`, `visualize.py`, EDA, maps | A | `feat/a-data` | 2026-11-18 11:00 | released 11:30 (Phase 1 T10–T12, T21 done) |
| `pathsnap/features.py`, `dataset.py`, `models.py`, `metrics.py`, Optuna | B | `feat/b-models` | 2026-11-18 10:45 | released 11:10 (Phase 1 done; Optuna T23 next) |
| `pathsnap/hmm.py`, `scoring.py`, decoder tuning | C | — | — | Phase 1 T17/T18 closed by D cross-cover; tuning (T25–T27) is Phase 2 |

---

## 3. What works (scaffold)

- Trip discovery, validation and inventory code.
- Local metric projection, point-to-segment geometry, smoothing, bearings.
- KD-tree candidate generation over densely sampled arcs, radius fallbacks.
- Route-label alignment code: nearest route arc + confidence flags; monotonic
  progress is a diagnostic (D-009).
- 18-feature matrix, including graph-distance features via a corridor subgraph.
- Cached dataset builder.
- Grouped-CV harness and six model families (code; no results yet).
- HMM/Viterbi with emissions, network transitions, same-arc handling, gap resets.
- FastAPI contract: `/health`, `/version`, `/trips`, `/predict`; request IDs;
  structured prediction log; optional `X-API-Key`; trip-ID validation;
  ground-truth isolation (`with_route=False` on the prediction path).
- Streamlit UI: health, trips, prediction, metrics, Folium map, timings, raw JSON.
- Folium map builder with network/GPS/reference/prediction layers.
- 27 tests: geo, IO, candidates, labels, metrics, HMM, API, leakage.
- Cross-agent protocol: `AGENTS.md`, this file, `docs/AGENT_LOG.md`,
  `docs/DECISIONS.md`, `docs/GLOSSARY.md`, `scripts/agent_status.py`.

---

## 4. What is missing / blocked

| Issue | Impact | Owner | Next |
|---|---|---|---|
| HMM tuning not run | defaults improve continuity 10/10 but reduce Top-1 on 9/10 | C | tune `lam/sigma/beta` on train/student-test; rerun `05_eval_hmm.py` |
| No physical two-laptop rehearsal | final demo unproven on LAN/hotspot | D | `docs/DEPLOYMENT.md` §2–§4 |
| No evaluator trips locally | UI falls back to `student_test` | D | place data on Laptop 1; label fallback |

---

## 5. Dataset / artifact versions

| Item | Value | Source |
|---|---|---|
| Dataset cache | train, 80 trips, 208,417 obs, 1,732,459 rows, graph features on | `results/models/dataset_summary.json` (cache itself ignored) |
| Model artifact | `6c229c610200`, HistGradientBoosting, untuned, fit on 731,586 rows | `artifacts/model_card.json` |
| `artifacts/hmm_config.json` | **defaults** from `config.py` written by the exporter; C to replace with tuned values (T25) | `artifacts/hmm_config.json` |
| Feature contract | `v1`, 18 features | `pathsnap/config.py` |
| HMM defaults | beta 10, sigma 8, lambda 0.5, gap reset 30 s | `pathsnap/config.py` |

When B builds/exports, fill this table with the measured metadata and the source
path. Never quote a version from memory.

---

## 6. Next actions (owners)

| Priority | Task | Owner | Command / evidence |
|---|---|---|---|
| ✅ | Optuna + tuned-vs-untuned (done; artifact unchanged, D/C need **not** re-run for B) | B | `results/optuna/`, `results/models/tuned_vs_untuned.csv` |
| P1 | HMM tuning (Top-1 regression) | C | tune on train/student-test; `python scripts/05_eval_hmm.py`; include B's T17 follow-ups (same-arc backward tolerance τ, Dijkstra cap from config, λ = 0) — `docs/reviews/T17_hmm_review_B.md` |
| P1 | Two-laptop rehearsal | D | `docs/DEPLOYMENT.md`; `/health` from Laptop 2, screenshot |
| P2 | Refresh integration after tuned artifacts | D | `python scripts/run_eval.py --partition student_test` |
| P2 | Report §7 numbers refresh | D | `report/07_deployment.md`, `viva.md` ledger |
| P2 | Final hardening | D | secrets scan, public-repo check, tag |

Detailed packages: [`docs/ASSIGNMENTS.md`](ASSIGNMENTS.md).

---

## 7. Current handoff

**State:** A/B/C Phase-1 work is committed and green. D verified the live
FastAPI + Streamlit path locally on B’s preliminary artifact: 10/10 student-test
trips pass in both decoder modes, with stage timings and screenshots. The default
HMM config improves continuity on all 10 trips but reduces Top-1 on 9/10 — C
must tune before the final numbers are frozen.

**Last verified commits:** `c3dcbed` (main before this branch) + this branch.

**Reproduce the integration:**

```bash
# data mounted under ./data (train/ + student_test/)
make api
make ui             # separate terminal, PATHSNAP_API_URL=http://127.0.0.1:8000
curl -s localhost:8000/health
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"trip_id":"00000056","mode":"hmm"}'
python3 scripts/run_eval.py --partition student_test --modes hmm classifier \
  --out results/integration/run_log.csv
```

**Artifacts:** preliminary `artifacts/model.joblib` v`6c229c610200`; default
`artifacts/hmm_config.json`; evidence in `results/integration/` and
`results/screenshots/`.

**Known failures:** HMM Top-1 regression under the default config (C); evaluator
trips not available locally; physical two-laptop path untested.

**Next three actions:** 1) C tunes the decoder and publishes
`results/hmm/classifier_vs_hmm.csv`; 2) D runs the two-laptop rehearsal;
3) B exports the tuned artifact and D refreshes §7.4–7.5.

**Do not touch:** `pathsnap/contracts.py`, `config.FEATURE_NAMES`, metric
definitions, or the prediction/ground-truth boundary without an ADR and a note
in `STATUS.md`.

---

## C Phase 2 handoff

T25, T26 and T27 runners are implemented on `feat/c-phase2-hmm`:
`scripts/05_tune_hmm.py`, `scripts/05_eval_hmm.py`, and
`scripts/06_hmm_failure_analysis.py`. The raw `data/` directory is not present
in this checkout, so the tuning table, final comparison CSV, and failure maps
must be generated after mounting the permitted train/student-test trips.

## 8. Open questions / blockers

| Question | Impact | Owner | Status |
|---|---|---|---|
| Are the 10 evaluator trips available before the final demo? | UI/integration can only use `student_test` until then | D | open |
| Which scoring definition will the organisers use for route accuracy? | We report multiple metrics to be safe | A/B | open |
| Report format: LaTeX vs Word/Google Docs? | Affects compilation | A | open |
| One-day or two-day scheduling? | Stage timeboxes | all | open |

---

### Handoff — 2026-11-18 11:10 — B (features/models)

**State:** Phase-1 B deliverables done on `feat/b-models`. Dataset built (80 train trips); 8 model families + nearest-road baseline compared with GroupKFold(5); HistGradientBoosting selected (CV Top-1 0.887, macro F1 0.928, log loss 0.084); feature ablation + permutation importance written; preliminary untuned artifact exported and verified through `backend.service.Predictor` (ready=true, one live student-test prediction, no route in the response).
**Last verified commit:** base `b41ffea` + uncommitted B changes on `feat/b-models`.
**Commands that reproduce it:**
```bash
python3 scripts/02_build_dataset.py --data-root ./data          # also writes results/models/dataset_summary.json, cv_splits.csv
python3 scripts/03_train_models.py --folds 5 --profile full --out-dir results/models
python3 scripts/03b_feature_ablation.py --model hist_gradient_boosting --folds 5 --repeats 3
python3 scripts/06_export_artifacts.py --dataset cache/dataset.npz --selected results/models/selected_model.json --artifact-dir artifacts
```
**Artifacts:** `artifacts/model.joblib` version `6c229c610200`; dataset `cache/dataset.npz` (train, 1,732,459 rows, FEATURE_VERSION v1).
**Known failures:** none. `tests/test_api.py` needs `httpx` (not in `requirements.txt`; D to add). `artifacts/hmm_config.json` holds defaults until C tunes.
**Next three actions:** 1) D integrates the artifact (T19/T20); 2) B runs Optuna on HistGradientBoosting (T23) and tuned-vs-untuned incl. student-test (T24); 3) C uses `cache/oof_hist_gradient_boosting.npy` / the artifact for classifier-vs-HMM (T26).
**Do not touch:** `config.FEATURE_NAMES` (v1 frozen), `results/models/` outputs (regenerate via the scripts above).

---

### Handoff — 2026-11-18 11:30 — A (data/EDA/labels/maps)

**State:** Phase-1 A deliverables done on `feat/a-data`. Full EDA on 90 trips (no invalid GPS rows or timestamp disorder; 30 trips with gaps > 30 s, max 153,064 s; 11 routes with breaks; 8 loop/out-and-back routes). Candidate sweep keeps the production config (60 m + 100/150 m fallbacks, K=15; miss 0.092% train). Label coverage table + 5 visual checks; only train `00000033` and `00000054` are unreliable. First Folium maps with screenshots.
**Last verified commit:** base `b41ffea` + A changes on `feat/a-data`.
**Commands that reproduce it:**
```bash
python3 scripts/01_eda.py --data-root ./data
python3 scripts/01b_candidate_sweep.py --data-root ./data
python3 scripts/01c_label_coverage.py --data-root ./data
python3 visualisation/make_maps.py --data-root ./data
```
**Known issues:** `scripts/02_build_dataset.py` (B) also writes `results/labels/coverage.csv`/`summary.json` (train only, older schema) and `results/eda/candidate_miss_summary.csv`; running `make dataset` after T12 overwrites A's files. B should stop writing into `results/labels/` (or write to `results/models/`). Candidate config unchanged → no impact on B's dataset.
**Next three actions:** 1) B: stop `02_build_dataset.py` writing A's result files; 2) A (Phase 2): T28 maps with predicted routes once C's HMM config exists, report §2 (T29); 3) optional A/B check: how often nearest-route-arc labels pick the opposite-direction arc on out-and-back routes (e.g. train `00000011`).
**Do not touch:** `results/eda/`, `results/labels/`, `results/maps/` outputs by hand — regenerate with the commands above.
