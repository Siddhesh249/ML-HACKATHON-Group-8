# Project status — live state board

> **Read this second, after `AGENTS.md`.** This is the authoritative “what is
> true now”. Keep edits small and localised. Remove your claim when you finish.
> Newest handoff at the bottom of the file.

**Last updated:** 2026-11-18 12:10
**Updated by:** C (HMM/decoder)
**Phase:** Phase 0 scaffold committed · C Phase 1 implementation in progress
**Current focus:** C transition tests/design complete; B review and real-trip benchmark remain

---

## 1. Health

| Area | State | Evidence / note |
|---|---|---|
| Unit/contract tests | ✅ 27 passed | `make test` |
| Package imports | ✅ | `make smoke` |
| Repo scaffold | ✅ | library, API, UI, scripts, docs, tests |
| EDA | ⬜ not run | A owns; `python scripts/01_eda.py --data-root ./data` |
| Candidate generator | ✅ baseline code | A to validate + radius sweep |
| Route labels/alignment | ✅ baseline code | A to validate + produce coverage table |
| Dataset cache | ⬜ not built | B owns; `make dataset` |
| Model comparison | ⬜ not run | B owns; `make train` |
| Selected model | ⬜ | B owns |
| Optuna study | ⬜ | B owns |
| Inference artifact | ⬜ missing | B must export; D then integrates |
| FastAPI | 🟨 scaffold | contract tests pass with an injected predictor; live prediction waits for the artifact |
| Streamlit UI | 🟨 scaffold | code complete; needs a live API |
| HMM decoder | ✅ baseline + ✅ transition tests / ⬜ tuning | 10 focused HMM tests pass; real-trip benchmark and tuning remain |
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
| `pathsnap/io.py`, `geo.py`, `network.py`, `candidates.py`, `labels.py`, EDA | A | — | — | unclaimed |
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
| No full EDA/label evidence | report §2 and first viva evidence missing | A | `python scripts/01_eda.py` |
| HMM tuning not run | decoder comparison missing | C | tune on train/student-test after B artifact |
| Real-trip HMM latency not recorded | T18 evidence incomplete | C | run `python -m scripts.benchmark_hmm` when raw data is mounted |
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
| P1 | B review of HMM design + real-trip transition benchmark | B/C | `docs/HMM_DESIGN.md`; `python scripts/benchmark_hmm.py ...` |
| P1 | Live API/UI smoke | D | `make api`, `make ui`, `curl /predict` |
| P2 | 10-trip integration | D | `python scripts/run_eval.py --partition student_test` |
| P2 | Two-laptop + screenshots | D | `docs/DEPLOYMENT.md` |
| P2 | Report/viva numbers | all | `viva.md` §18 ledger |

Detailed packages: [`docs/ASSIGNMENTS.md`](ASSIGNMENTS.md).

---

## 7. Current handoff

**State:** C’s Phase 1 HMM design note, corridor transition documentation,
transition edge-case tests, and a real-trip benchmark command are committed on
`feat/c-hmm`. The focused HMM suite passes 10 tests. The raw dataset is not
mounted, so no real-trip latency CSV exists yet; B review of the design note,
decoder tuning, and classifier-vs-HMM comparison remain pending.

**Last verified commit:** `7a9dab0` on `feat/c-hmm`

**C handoff:** B should review `docs/HMM_DESIGN.md`; once `data/train/` is
mounted, run `python -m scripts.benchmark_hmm --data-root ./data --partition
train --trip-id <largest-trip-id>`. Decoder tuning and classifier-vs-HMM
comparison wait for B’s exported model artifact.

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
