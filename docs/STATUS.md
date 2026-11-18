# Project status — live state board

> **Read this second, after `AGENTS.md`.** This is the authoritative “what is
> true now”. Keep edits small and localised. Remove your claim when you finish.
> Newest handoff at the bottom of the file.

**Last updated:** 2026-11-18 11:35
**Updated by:** D (platform/integration)
**Phase:** Phase 0 scaffold · Phase 1: B preliminary model/artifact done; A/C pending; D integrating
**Current focus:** D runs the live API/UI and 10-trip integration on B’s preliminary artifact; A full EDA/labels; C decoder tuning.

---

## 1. Health

| Area | State | Evidence / note |
|---|---|---|
| Unit/contract tests | ✅ 32 passed | `make test` (B added `tests/test_models.py`; needs `httpx` for `tests/test_api.py`, see §8) |
| Package imports | ✅ | `make smoke` |
| Repo scaffold | ✅ | library, API, UI, scripts, docs, tests |
| EDA | ⬜ not run | A owns; `python scripts/01_eda.py --data-root ./data` |
| Candidate generator | ✅ baseline code | A to validate + radius sweep |
| Route labels/alignment | ✅ baseline code | A to validate + produce coverage table |
| Dataset cache | ✅ built (local) | 80 train trips, 1,732,459 rows; evidence `results/models/dataset_summary.json`, `cv_splits.csv` |
| Model comparison | ✅ Phase 1 | 8 families + baseline, GroupKFold(5): `results/models/model_comparison.csv` |
| Selected model | ✅ preliminary | HistGradientBoosting, untuned, CV Top-1 0.887: `results/models/selected_model.json` |
| Optuna study | ⬜ | B owns |
| Inference artifact | ✅ preliminary | `artifacts/model.joblib` v`6c229c610200`; `Predictor` ready=true, live predict verified on 1 student-test trip — **D can integrate** |
| FastAPI | 🟨 ready to integrate | B’s preliminary artifact is present; D to run the live API/UI smoke |
| Streamlit UI | 🟨 scaffold | code complete; needs the live API process |
| HMM decoder | ✅ baseline + ⬜ tuning | toy/synthetic tests pass; C owns tuning/comparison |
| Integration (10 trips) | 🟨 ready to run | artifact available; D runs `scripts/run_eval.py` |
| Two-laptop demo | ⬜ | `docs/DEPLOYMENT.md` after integration |

**B’s preliminary dataset, model comparison and artifact are committed. A and C
outputs are still pending.** Quote only numbers that exist in `results/`.

---

## 2. Active claims

Add a row **before** editing an area. Areas follow `docs/ASSIGNMENTS.md` §1.

| Area / files | Agent | Branch | Since | Status |
|---|---|---|---|---|
| `backend/`, `ui/`, `docs/DEPLOYMENT.md`, integration | D | `feat/d-platform` | 2026-11-18 10:20 | active |
| `pathsnap/io.py`, `geo.py`, `network.py`, `candidates.py`, `labels.py`, EDA | A | — | — | unclaimed |
| `pathsnap/features.py`, `dataset.py`, `models.py`, `metrics.py`, Optuna | B | `feat/b-models` | 2026-11-18 10:45 | released 11:10 (Phase 1 done; Optuna T23 next) |
| `pathsnap/hmm.py`, `scoring.py`, decoder tuning | C | — | — | unclaimed |

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
| No Optuna study | requirement unmet | B | `make tune` |
| No full EDA/label evidence | report §2 and first viva evidence missing | A | `python scripts/01_eda.py` |
| HMM tuning not run | decoder comparison missing | C | tune on train/student-test |
| No evaluator trips locally | UI falls back to `student_test` | D | place data on Laptop 1; label fallback |
| No screenshots | report §7 incomplete | D | capture after artifact |

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
| P0 | Full EDA + inventory | A | `python scripts/01_eda.py --data-root ./data --out-dir results/eda` |
| P0 | Label coverage table | A | `make dataset` (or alignment script) → `results/labels/` |
| P0 | Integrate B's preliminary artifact (T19/T20) | D | `make api` → `GET /health` ready; `POST /predict` |
| P1 | Optuna + tuned-vs-untuned | B | `make tune` → `results/optuna/` |
| P1 | HMM tune + comparison | C | `python scripts/05_eval_hmm.py` |
| P1 | Live API/UI smoke | D | `make api`, `make ui`, `curl /predict` |
| P2 | 10-trip integration | D | `python scripts/run_eval.py --partition student_test` |
| P2 | Two-laptop + screenshots | D | `docs/DEPLOYMENT.md` |
| P2 | Report/viva numbers | all | `viva.md` §18 ledger |

Detailed packages: [`docs/ASSIGNMENTS.md`](ASSIGNMENTS.md).

---

## 7. Current handoff

**State:** the scaffold is committed and green. No dataset cache, model
comparison or artifact exists. A/B/C can start in parallel from
`docs/ASSIGNMENTS.md`. D keeps the API/UI contract green and integrates once B
exports `artifacts/model.joblib`.

**Last verified commit:** `<fill after the scaffold commit>`

**Reproduce the scaffold:**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ln -sfn "../Student folder/train" data/train
ln -sfn "../Student folder/student_test" data/student_test
make test
make smoke
make status
```

**Artifacts:** `artifacts/model.joblib` (preliminary HistGradientBoosting,
untuned), `feature_config.json`, `hmm_config.json`, `model_card.json`; dataset
evidence in `results/models/`. The dataset cache itself is ignored.

**Known failures:** the live API/UI path has not yet been smoke-tested
end-to-end by D; the UI shows the `student_test` fallback until evaluator trips
are placed.

**Next three actions:** 1) D starts the API and runs one live `/predict`;
2) A runs the full EDA and label coverage; 3) C tunes the decoder and produces
the classifier-vs-HMM comparison.

**Do not touch:** `pathsnap/contracts.py`, `config.FEATURE_NAMES`, metric
definitions, or the prediction/ground-truth boundary without an ADR and a note
in `STATUS.md`.

---

## 8. Open questions / blockers

| Question | Impact | Owner | Status |
|---|---|---|---|
| Are the 10 evaluator trips available before the final demo? | UI/integration can only use `student_test` until then | D | open |
| Which scoring definition will the organisers use for route accuracy? | We report multiple metrics to be safe | A/B | open |
| Report format: LaTeX vs Word/Google Docs? | Affects compilation | A | open |
| One-day or two-day scheduling? | Stage timeboxes | all | open |
| `httpx` missing from `requirements.txt` (Starlette `TestClient` needs it; clean install fails `tests/test_api.py` collection) | `make test` on a fresh venv | D | open — B worked around with `pip install httpx` locally |

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
