# Project status — live state board

> **Read this second, after `AGENTS.md`.** This is the authoritative “what is
> true now”. Keep edits small and localised. Remove your claim when you finish.
> Newest handoff at the bottom of the file.

**Last updated:** 2026-11-18 11:20
**Updated by:** D (platform/integration)
**Phase:** Phase 0 scaffold committed · Phase 1 not started by A/B/C
**Current focus:** owners start their work packages from `docs/ASSIGNMENTS.md`

---

## 1. Health

| Area | State | Evidence / note |
|---|---|---|
| Unit/contract tests | ✅ 27 passed | `make test` |
| Package imports | ✅ | `make smoke` |
| Repo scaffold | ✅ | library, API, UI, scripts, docs, tests |
| EDA | ✅ T10 | 90 trips; `results/eda/inventory.csv`, `summary.json`, `quality_checks.csv`, `flags.csv`, figures; narrative `docs/DATA_NOTES.md` |
| Candidate generator | ✅ validated (T11) | sweep: production 60 m + fallbacks, K=15 → miss 0.092% train / 0.032% student-test; `results/eda/candidate_sweep.csv` |
| Route labels/alignment | ✅ validated (T12) | train 183,052/208,417 labelled; unreliable train `00000033`, `00000054`; `results/labels/coverage.csv`, 5 visual checks |
| Dataset cache | ⬜ not built | B owns; `make dataset` |
| Model comparison | ⬜ not run | B owns; `make train` |
| Selected model | ⬜ | B owns |
| Optuna study | ⬜ | B owns |
| Inference artifact | ⬜ missing | B must export; D then integrates |
| FastAPI | 🟨 scaffold | contract tests pass with an injected predictor; live prediction waits for the artifact |
| Streamlit UI | 🟨 scaffold | code complete; needs a live API |
| HMM decoder | ✅ baseline + ⬜ tuning | toy/synthetic tests pass; C owns tuning/comparison |
| Integration (10 trips) | ⬜ blocked | waits for B's artifact |
| Two-laptop demo | ⬜ | `docs/DEPLOYMENT.md` after integration |

**No result, dataset cache or model artifact is committed yet.** The exploratory
runs from the scaffold phase were removed so that A/B/C produce and own the
committed evidence. Do not quote a metric that is not in `results/`.

---

## 2. Active claims

Add a row **before** editing an area. Areas follow `docs/ASSIGNMENTS.md` §1.

| Area / files | Agent | Branch | Since | Status |
|---|---|---|---|---|
| `backend/`, `ui/`, `docs/DEPLOYMENT.md`, integration | D | `feat/d-platform` | 2026-11-18 10:20 | active |
| `pathsnap/io.py`, `geo.py`, `network.py`, `candidates.py`, `labels.py`, `visualize.py`, EDA, maps | A | `feat/a-data` | 2026-11-18 11:00 | released 11:30 (Phase 1 T10–T12, T21 done) |
| `pathsnap/features.py`, `dataset.py`, `models.py`, `metrics.py`, Optuna | B | — | — | unclaimed |
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
| No dataset cache | training and integration blocked | B (with A) | `make dataset` |
| No model comparison | no model-selection evidence | B | `make train` |
| No Optuna study | requirement unmet | B | `make tune` |
| No exported artifact | `/predict` returns 503 live | B | `make export` |
| HMM tuning not run | decoder comparison missing | C | tune on train/student-test |
| No evaluator trips locally | UI falls back to `student_test` | D | place data on Laptop 1; label fallback |
| No screenshots | report §7 incomplete | D | capture after artifact |

---

## 5. Dataset / artifact versions

| Item | Value | Source |
|---|---|---|
| Dataset cache | **not built** | `cache/dataset.*` (ignored) |
| Model artifact | **missing** | `artifacts/model.joblib` |
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
| P0 | Build dataset | B | `make dataset` → `cache/dataset.npz` |
| P0 | 5-model comparison (5-fold, full) | B | `python scripts/03_train_models.py --folds 5 --profile full` |
| P0 | Export artifact | B | `make export` → `artifacts/model.joblib` |
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

**Artifacts:** none. `cache/` and `results/` contain only README files.

**Known failures:** `/predict` live returns 503 until B exports the artifact; the
UI shows the `student_test` fallback until evaluator trips are placed.

**Next three actions:** 1) A runs the full EDA; 2) B builds the dataset and
starts the model comparison; 3) C reviews/tunes the decoder while waiting for
B’s artifact.

**Do not touch:** `pathsnap/contracts.py`, `config.FEATURE_NAMES`, metric
definitions, or the prediction/ground-truth boundary without an ADR and owner
review.

---

## 8. Open questions / blockers

| Question | Impact | Owner | Status |
|---|---|---|---|
| Are the 10 evaluator trips available before the final demo? | UI/integration can only use `student_test` until then | D | open |
| Which scoring definition will the organisers use for route accuracy? | We report multiple metrics to be safe | A/B | open |
| Report format: LaTeX vs Word/Google Docs? | Affects compilation | A | open |
| One-day or two-day scheduling? | Stage timeboxes | all | open |

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
