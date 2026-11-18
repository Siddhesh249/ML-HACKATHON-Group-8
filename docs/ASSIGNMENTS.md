# Work assignments and handoffs

> **Purpose:** give each member a self-contained starting point and make the
> dependencies between them explicit. The scaffold is committed; the experiment
> work below belongs to the named owner. D does not train or select models.
>
> Read `AGENTS.md`, `docs/STATUS.md` and `viva.md` §0 before starting.

## 0. Current state (after the scaffold commit)

| Item | State |
|---|---|
| Shared library, API, UI, scripts, tests | ✅ scaffold committed; 27 tests pass |
| Dataset cache | ⬜ not built (`cache/` is git-ignored) |
| Full EDA / label evidence | ⬜ A owns |
| Model comparison / Optuna / artifact | ⬜ B owns |
| HMM tuning / classifier-vs-HMM | ⬜ C owns |
| Live API/UI integration | 🟨 D scaffolded; blocked on B's artifact |
| Report / viva numbers | ⬜ owners fill; D coordinates the demo |

**Nothing in `results/` or `artifacts/` is a final result yet.** The exploratory
runs from the scaffold phase were removed so the owners can produce the
committed evidence themselves.

---

## 1. Ownership contract

| Area | Owner | Must not be edited by |
|---|---|---|
| `pathsnap/io.py`, `geo.py`, `network.py`, `candidates.py`, `labels.py`, `visualize.py`, `scripts/01_eda.py`, `results/eda/`, `results/labels/` | **A** | B/C/D (do not edit) |
| `pathsnap/features.py`, `dataset.py`, `models.py`, `metrics.py`, `training/`, `optimization/`, `results/models/`, `results/optuna/`, `artifacts/model.joblib` | **B** | A/C/D (do not edit) |
| `pathsnap/hmm.py`, `scoring.py`, `results/hmm/`, `artifacts/hmm_config.json` | **C** | A/B/D (do not edit) |
| `backend/`, `ui/`, `scripts/run_eval.py`, `docs/DEPLOYMENT.md`, `results/integration/`, `results/screenshots/`, `report/07_deployment.md` | **D** | A/B/C (do not edit) |
| `README.md`, `viva.md`, `report/` | shared; section owner writes, A compiles | — |
| `AGENTS.md`, `docs/STATUS.md`, `docs/AGENT_LOG.md`, `docs/DECISIONS.md` | shared append-only / small edits | — |

If you need a file owned by someone else, commit the change and mention the
owner in the commit message. Do not silently “fix” another area.

---

## 2. A — Data, EDA, candidates, labels, maps

**First action:** build the full inventory and label evidence.

```bash
python3 scripts/01_eda.py --data-root ./data --out-dir results/eda
```

This writes `results/eda/inventory.csv`, `summary.json` and figures. It does not
train anything.

### A1 · Full EDA and data-quality record (T10)

- Run the command above on all 90 provided trips.
- Add: missing/invalid counts, sampling-interval histogram, GPS plots, network
  connectivity/size tables, flagged trips.
- Write the narrative in `docs/DATA_NOTES.md` and `report/02_data.md`.
- **DoD:** `results/eda/` committed; every number in the report has a file.

### A2 · Candidate generation validation and radius sweep (T11)

- Validate the baseline `CandidateGenerator` against the reference arcs.
- Run a radius/top-K sweep (30/50/60/80/100 m) and write
  `results/eda/candidate_sweep.csv` with miss rate and candidate counts.
- Justify the chosen radius/top-K with the measured trade-off.
- **DoD:** sweep CSV, one figure, a paragraph in the report.

### A3 · Label policy review and coverage table (T12)

- Review the D-009 decision (nearest route arc; monotonic progress only as a
  diagnostic). Run `make dataset` or the underlying alignment on all trips.
- Produce `results/labels/coverage.csv` and `summary.json`: coverage, median/p90
  projection distance, backward fraction, candidate miss rate per trip.
- Visual-check at least 3 trips, including a loop/reversed route and a flagged
  trip. Document the two genuinely unreliable routes.
- **DoD:** evidence committed; A can defend the label policy in the viva.

### A4 · Folium maps (T21, T28)

- Implement `visualisation/make_maps.py` using `pathsnap/visualize.py`.
- Generate HTML for: GPS + network, GPS + reference + prediction, and one
  classifier-vs-HMM example.
- Save screenshots for the PDF.
- **DoD:** `results/maps/*.html` + screenshots + the generation script.

### A5 · Report §2 and figure set

- Own `report/02_data.md`, the EDA figures and the data section of the viva.
- **DoD:** matches `report/REPORT_OUTLINE.md`; every number is in `results/`.

**Merge:** directly to `main` when the DoD and `make test` pass. **Do not touch:** `backend/`, `ui/`, model training.

---

## 3. B — Features, models, metrics, Bayesian optimisation, artifact

**First action:** build the dataset and run the minimum five-model comparison.

```bash
make dataset
python3 scripts/03_train_models.py --folds 5 --profile full \
  --out-dir results/models
```

`--profile fast` exists for a quick integration pass, but the reported comparison
must state the profile and preferably use `full` with 5 folds.

### B1 · Feature matrix and ablation (T13)

- Review `pathsnap/features.py`; run an ablation: local only → + relative →
  + temporal/graph.
- Record `results/models/feature_ablation.csv`.
- Freeze any feature changes with a `FEATURE_VERSION` bump and ADR.
- **DoD:** ablation table; feature importance plot; viva §7 updated.

### B2 · Dataset builder and grouped splits (T14)

- Run `make dataset`; verify 80 trips, no trip crosses folds, no evaluator data.
- Record the dataset metadata and per-trip counts.
- **DoD:** `cache/dataset.json` (local) + a committed summary in `results/`.

### B3 · Metrics and ≥5-model comparison (T15, T16)

- Run the full 5-fold comparison on at least five distinct families.
- Write `results/models/model_comparison.csv` with CV Top-1, macro F1, positive
  F1, log loss, train/inference time.
- Save per-fold reports and OOF probabilities (local cache).
- Select the final model on CV Top-1 with F1/log-loss guardrails; write
  `results/models/selected_model.json`.
- **DoD:** every model has the required report subsection answers.

### B4 · Bayesian optimisation (T23, T24)

```bash
python3 scripts/04_tune.py --model <selected> --trials 30 --folds 3
```

- Record the search space, objective, trials, best params, cost and plots in
  `results/optuna/`.
- Compare tuned vs untuned on the same folds and on the 10 student-test trips.
- **DoD:** `study.db`, `trials.csv`, `best_params.json`, tuned-vs-untuned CSV.

### B5 · Export the inference artifact (T17)

```bash
python3 scripts/06_export_artifacts.py \
  --dataset cache/dataset.npz \
  --selected results/models/selected_model.json \
  --best-params results/optuna/best_params.json \
  --artifact-dir artifacts
```

- **DoD:** `artifacts/model.joblib`, `feature_config.json`, `model_card.json`;
  `GET /health` reports `ready=true` on D’s API.
- Notify D in `docs/STATUS.md` when the artifact is available.

### B6 · Report §3–4 and model figures

- Own `report/03_models.md`, `report/04_optuna.md`, the comparison bars and the
  real learning curves (only for models with epoch histories).
- **DoD:** actual results only; no fabricated curves.

**Merge:** directly to `main` when the DoD and `make test` pass. **Do not touch:** `backend/`, `ui/`, HMM tuning.

---

## 4. C — HMM/Viterbi, decoder tuning, sequence analysis

**First action:** review the seeded HMM design and run the toy/synthetic tests.

```bash
make test
python3 -m pytest tests/test_hmm.py -q
```

### C1 · HMM design review (T17)

- Review `docs/HMM_DESIGN.md` and `pathsnap/hmm.py`: states, emissions,
  transitions, same-arc handling, gap resets, corridor graph, complexity.
- Update the design note with any changes and an ADR if needed.
- **DoD:** design note matches the code; B consulted if the interface changes.

### C2 · Transition implementation and unit tests (T18)

- Add tests for: same-arc progress, reversed arc, unreachable candidate, gap
  reset, U-turn penalty, empty candidate layer.
- Measure shortest-path/corridor performance on a large trip.
- **DoD:** tests pass; latency recorded in `results/hmm/`.

### C3 · Decoder tuning (T25)

- Tune `lam`, `sigma_m`, `beta_m`, `max_speed_mps`, `gap_reset_s`,
  `uturn_penalty` on training/student-test only.
- Write `results/hmm/decoder_tuning.csv` and choose `artifacts/hmm_config.json`.
- **DoD:** no evaluator data; search record committed.

### C4 · Classifier-only vs HMM (T26, T27)

```bash
python3 scripts/05_eval_hmm.py
```

- Requires B’s artifact. Produce `results/hmm/classifier_vs_hmm.csv` with Top-1,
  continuity, LCS/arc-set F1 and timings.
- Analyse at least three failure cases: missing candidate, gap, junction,
  over-smoothing.
- **DoD:** comparison table + failure maps + `report/05_hmm.md`.

**Merge:** directly to `main` when the DoD and `make test` pass. **Do not touch:** model selection, candidate generation.

---

## 5. D — Platform, integration, deployment, demo

**First action:** keep the scaffold green and wait for B’s artifact; do not train.

```bash
make test
make status
```

### D1 · API/UI scaffold (done)

- FastAPI endpoints, ground-truth isolation, logging, optional API key, CORS.
- Streamlit UI: health, trips, predict, metrics, map, timings, raw JSON.
- **Done:** contract tests pass with an injected predictor.

### D2 · Integrate B’s artifact (T19, T20)

- When `artifacts/model.joblib` exists, run the API and one live prediction.
- Capture the log line and the UI screenshot.
- **DoD:** `GET /health` ready; `/predict` returns real metrics; no `.route` in
  the response.

### D3 · Integration across the 10 student-test trips (T32)

```bash
python3 scripts/run_eval.py --partition student_test --modes hmm classifier
```

- Write `results/integration/run_log.csv` and latency percentiles.
- Investigate and log any failing trip; do not hardcode.
- **DoD:** 10/10 pass; timings and failures documented.

### D4 · Two-laptop deployment and demo (T33)

- Rehearse `docs/DEPLOYMENT.md` on LAN/hotspot.
- Capture API terminal logs and UI screenshots in `results/screenshots/`.
- **DoD:** evaluator can select any of the ten trips and see a live result.

### D5 · Report §7 and final hardening (T31, T36)

- Fill `report/07_deployment.md` with actual screenshots/logs.
- Secrets scan, public-repo check, tag `v1.0-final`.

**Merge:** directly to `main` when the DoD and `make test` pass. **Do not train, tune or select models.**

---

## 6. Handoff contract (what “done” means for the next person)

| Producer | Deliverable | Path | Consumer | Acceptance check |
|---|---|---|---|---|
| A | EDA inventory + figures | `results/eda/` | all | files exist; numbers reproducible |
| A | Label coverage + flags | `results/labels/coverage.csv`, `summary.json` | B, C, D | coverage/miss table matches the script |
| A | Candidate sweep | `results/eda/candidate_sweep.csv` | B | radius/top-K justified |
| B | Dataset cache | `cache/dataset.npz` + `cache/dataset.json` | B, C | 80 trips; no evaluator data |
| B | Model comparison | `results/models/model_comparison.csv` | all | ≥5 models; actual metrics |
| B | Selected model | `results/models/selected_model.json` | C, D | params + rationale |
| B | Artifact | `artifacts/model.joblib` + configs | C, D | `GET /health` ready |
| C | HMM config | `artifacts/hmm_config.json` | D | tuned on permitted data |
| C | Classifier vs HMM | `results/hmm/classifier_vs_hmm.csv` | all | same candidates/labels |
| D | Integration log | `results/integration/run_log.csv` | all | 10/10 trips, real latencies |
| D | Demo evidence | `results/screenshots/` | A | API log + UI screenshot |

**Rule:** the consumer must be able to reproduce the deliverable from the
command in this file and the metadata in `results/`. If not, the handoff is not
done.

---

## 7. Suggested order and checkpoints

| Order | A | B | C | D |
|---|---|---|---|---|
| 1 | Full EDA (A1) | Dataset + features (B2, B1) | HMM review/tests (C1, C2) | Scaffold green (D1) |
| 2 | Labels + sweep (A2, A3) | Model comparison (B3) | Decoder tuning (C3) | Wait for artifact |
| 3 | Maps (A4) | Optuna (B4) | Comparison (C4, needs artifact) | Live API/UI (D2) |
| 4 | Report §2 | Export + report §3–4 (B5, B6) | Report §5 | 10-trip integration (D3) |
| 5 | — | tuned-vs-untuned (B4) | failure maps | Deployment/screenshots (D4, D5) |

**Phase-1 gate:** A has EDA + labels, B has ≥5 models and a preliminary artifact,
C has HMM tests + design, D has one live API/UI response.
**Final gate:** all handoff rows accepted; report PDF; public repo; demo rehearsed.

---

## 8. Branch and merge policy

- Branch per task: `feat/a-data`, `feat/b-models`, `feat/c-hmm`, `feat/d-platform`, `docs/report`.
- Commit, rebase on `main`, and **merge directly to `main`** when the DoD is met,
  `make test` passes, and the evidence path exists. **No PR required.**
- **No reviewer.** For a shared-contract change, mention the affected owners in
  the commit message; do not block on approval.
- `AGENTS.md` §9 is the Definition of Done.
