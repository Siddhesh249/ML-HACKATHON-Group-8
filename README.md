# PathSnap — Distributed Map-Matching & Route Reconstruction

**Course:** UE24CS352A Machine Learning · **Team:** Group 8
**Task:** Reconstruct realistic vehicle routes from noisy 1 Hz GPS traces by matching observations to a road network, using supervised candidate classification followed by HMM/Viterbi sequence decoding.

> **Status:** Phase 0 scaffold complete. Phase 1 work is assigned below.
> Every number reported in this repository or in the technical report must come from a file in `results/` (see [Numbers policy](#numbers-and-evidence-policy)).
>
> **First viva:** the first viva focuses on the **task, dataset, preprocessing and feature selection**. Read [`viva.md` §0](viva.md#0-first-viva--task-dataset-preprocessing-and-features-read-this-first) before it. Model/decoder numbers are Phase 2; never quote a `TBD`.
>
> **New here — human or AI agent? Read [`AGENTS.md`](AGENTS.md) first**, then [`docs/STATUS.md`](docs/STATUS.md) and the last three entries of [`docs/AGENT_LOG.md`](docs/AGENT_LOG.md). Claim your task before editing; log it when you finish.
>
> **For AI agents and members:** `viva.md` is a living document. Read [§16 Viva maintenance protocol](#16-viva-maintenance-protocol-mandatory-for-members-and-ai-agents) before changing code and update it in the **same** commit/PR. Cross-agent state lives in [`AGENTS.md`](AGENTS.md), [`docs/STATUS.md`](docs/STATUS.md), [`docs/AGENT_LOG.md`](docs/AGENT_LOG.md) and [`docs/DECISIONS.md`](docs/DECISIONS.md) — see [§17](#17-cross-agent-communication-and-state-sharing).

---

## 1. Team and ownership

| Code | Role | Primary scope | Primary files | Viva sections |
|------|------|---------------|---------------|---------------|
| **A** | Data & EDA Lead | Ingestion, validation, EDA, geo math, network utilities, candidate generation, reference-route alignment/labels, Folium maps, report compilation | `pathsnap/io.py`, `geo.py`, `network.py`, `candidates.py`, `labels.py`, `visualize.py`, `scripts/01_eda.py`, `report/` | 2–6, 17–18 |
| **B** | Modelling & Optimisation Lead | Feature matrix, dataset cache, model zoo, CV harness, metrics, model selection, Bayesian optimisation, artifacts | `pathsnap/features.py`, `dataset.py`, `models.py`, `metrics.py`, `training/`, `optimization/` | 7–11 |
| **C** | Sequence Modelling Lead | HMM/Viterbi implementation, transition graph, gap handling, decoder tuning, classifier-vs-HMM comparison, error analysis | `pathsnap/hmm.py`, `pathsnap/scoring.py`, `scripts/05_eval_hmm.py` | 12–13 |
| **D** | Platform & Integration Lead | FastAPI backend, Streamlit UI, two-laptop deployment, logging/proof of inference, integration over all trips, demo runbook | `backend/`, `ui/`, `docs/DEPLOYMENT.md`, `scripts/07_run_api.sh`, `scripts/08_run_ui.sh` | 14–16, 20 |

**Replace the codes above with member names/handles before Phase-1 evaluation.**
Reviewer pairs: **A ↔ D** and **B ↔ C**. Every PR needs one reviewer from the paired owner.

Detailed, self-contained work packages with commands, files, DoD and handoff contracts: [`docs/ASSIGNMENTS.md`](docs/ASSIGNMENTS.md).

Shared responsibilities: report writing (each owner writes their sections), `viva.md` upkeep (everyone), final demo rehearsal (everyone).

---

## 2. What we are building

```
GPS + per-trip road network
        │
        ▼
loader / validator ──► local metric projection ──► candidate generation (KD-tree)
                                                          │
                                                          ▼
                                       features per (observation, candidate)
                                                          │
                                                          ▼
                                        classifier → candidate probabilities
                                                          │
                                                          ▼
                                  HMM + Viterbi (network-aware transitions)
                                                          │
                     ┌────────────────────────────────────┴──────────────────┐
                     ▼                                                       ▼
      predicted arc sequence                                   server-side scoring
                     │                                        (reference `.route` never
                     ▼                                         enters the predictor)
             FastAPI (Laptop 1) ──────────► Streamlit UI (Laptop 2)
```

The system is **distributed in deployment**: Laptop 1 holds the model, HMM, road networks and ground truth; Laptop 2 is a thin client. Ground truth is never sent to the UI and never read by the prediction path.

---

## 3. Deliverables map (rubric → repository)

| Required component | Where it lives | Owner |
|---|---|---|
| Data processing | `pathsnap/io.py`, `geo.py`, `network.py`, `scripts/01_eda.py` | A |
| Feature engineering | `pathsnap/candidates.py`, `features.py` | A + B |
| Model training | `pathsnap/models.py`, `training/`, `scripts/03_train_models.py` | B |
| Model evaluation | `pathsnap/metrics.py`, `results/models/`, `results/figures/` | B |
| Bayesian optimisation | `optimization/`, `scripts/04_tune.py`, `results/optuna/` | B |
| HMM and Viterbi | `pathsnap/hmm.py`, `scripts/05_eval_hmm.py`, `results/hmm/` | C |
| API | `backend/` | D |
| UI | `ui/` | D |
| Deployment | `Makefile`, `requirements.txt`, `.env.example`, `docs/DEPLOYMENT.md` | D |
| Visualisations | `pathsnap/visualize.py`, `visualisation/`, `results/maps/` | A |
| Supporting results | `results/`, `artifacts/` | A/B/C |
| Technical report | `report/` | A (all write) |

---

## 3.1 Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ (verified on 3.14) | One language for data, ML, API and UI |
| Numerics | NumPy, SciPy | Vectorised geometry, `cKDTree` candidate index, sparse Dijkstra |
| Data / EDA | pandas, matplotlib, seaborn | Inventory, quality tables, report figures |
| ML | scikit-learn | Logistic regression, GaussianNB, decision tree, RandomForest, ExtraTrees, HistGradientBoosting, MLP, calibrated LinearSVC |
| Bayesian optimisation | Optuna (TPE) | Grouped-CV objective, trial records, parameter importance |
| Sequence model | custom HMM + Viterbi (NumPy) | Explicit emissions, network transitions, gap handling |
| Maps | Folium | GPS/network/reference/prediction layers, committed HTML + PDF screenshots |
| Backend | FastAPI, Uvicorn, Pydantic | Typed API, validation, CORS, live request logs |
| UI | Streamlit, requests | Thin client on Laptop 2; no model or data imports |
| Artifacts | joblib + JSON | Model, feature contract, HMM config, model card |
| Testing | pytest | Geometry, candidates, labels, metrics, HMM, API contract, leakage |
| Tooling | Makefile, Git/GitHub, `.env` | One-command reproduce/run; no committed secrets |

**Deliberately excluded:** PyTorch/TensorFlow (candidate classification + HMM is the right model class), XGBoost/LightGBM (Python 3.14 wheel risk; HistGradientBoosting covers boosting), geopandas/pyproj (local metric projection is accurate at trip scale), a database (NPZ/CSV artifacts suffice), Docker/cloud (the brief requires two-laptop deployment).

---

## 4. Repository layout

```
README.md                  ← task board, setup, workflow
AGENTS.md                  ← read-first contract for humans and AI agents
viva.md                    ← viva preparation, explanations, Q&A bank
docs/
  STATUS.md                ← live state: health, claims, handoff, next actions
  AGENT_LOG.md             ← append-only session history (what/when/why)
  DECISIONS.md             ← append-only ADRs (why this, not that)
  GLOSSARY.md              ← shared vocabulary
  DATA_NOTES.md  EXPERIMENTS.md  DEPLOYMENT.md  HMM_DESIGN.md
requirements.txt
Makefile
pytest.ini
.env.example
pathsnap/                  ← shared library (no UI imports, no plot side effects)
  config.py  contracts.py  io.py  geo.py  network.py
  candidates.py  labels.py  features.py  dataset.py
  metrics.py  models.py  hmm.py  scoring.py  visualize.py
backend/                   ← FastAPI service (Laptop 1)
ui/                        ← Streamlit client (Laptop 2)
scripts/                   ← reproducible stage scripts
training/                  ← experiment scripts and logs
optimization/              ← Optuna configs, search records
visualisation/             ← map-generation entry points
tests/                     ← unit + contract + leakage tests
docs/                      ← data notes, experiment protocol, deployment runbook
results/                   ← CSV metrics, figures, logs (committed)
artifacts/                 ← model.joblib, configs, model card (committed)
report/                    ← report sections, figures, final PDF
data/                      ← provided dataset (NOT committed)
```

Rules:

- `pathsnap/` must not import `streamlit`, `fastapi` or `folium` at module import time.
- `ui/` must import only `streamlit`, `requests`, `folium` and the browser component — never the model or data loaders.
- No secrets, API keys or private credentials anywhere in the repo.

---

## 5. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env          # adjust paths if needed
make test                     # unit tests, no dataset required
```

Python 3.11+ is recommended; the current machine runs 3.14. `scikit-learn`, `scipy`, `optuna`, `folium`, `fastapi` and `streamlit` install on 3.14. **Do not depend on XGBoost/LightGBM** — use `HistGradientBoostingClassifier` for the boosting family.

### Dataset placement

The provided dataset is **not committed** (91 MB of course material). Place it as:

```
data/
├── train/<trip_id>/<trip_id>.{track,nodes,arcs,route}           # 80 trips
├── student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}    # 10 trips
└── evaluator_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}  # 10 reserved trips
```

The student package contains only `train/` and `student_test/`. The ten evaluator trips are placed on Laptop 1 at evaluation time; the API discovers them automatically. Override the root with `PATHSNAP_DATA_ROOT` and the evaluation folder with `PATHSNAP_EVALUATOR_DIR`.

The original folders in this workspace are `Student folder/` and `PathSnap_student_materials/`; both are git-ignored.

---

## 6. Run

```bash
make eda                 # results/eda: inventory, quality stats, figures
make dataset             # build cached candidate/feature matrix (B)
make train               # train + compare >= 5 models (B)
make tune                # Optuna study + tuned model (B)
make eval-hmm            # decoder experiments + comparison (C)
make export              # copy best model + configs into artifacts/ (B/C)
make api                 # FastAPI on 0.0.0.0:8000 (Laptop 1)
make ui                  # Streamlit on 0.0.0.0:8501 (Laptop 2)
make test                # pytest
```

API environment:

| Variable | Default | Purpose |
|---|---|---|
| `PATHSNAP_DATA_ROOT` | `./data` | Root containing partition folders |
| `PATHSNAP_EVALUATION_PARTITION` | `evaluator_test` | Folder the UI may select trips from |
| `PATHSNAP_FALLBACK_PARTITION` | `student_test` | Fallback for local demos when evaluator trips are absent |
| `PATHSNAP_ARTIFACT_DIR` | `./artifacts` | Model, feature config, HMM config |
| `PATHSNAP_API_KEY` | empty | Optional `X-API-Key`; never commit a value |
| `PATHSNAP_CORS_ORIGINS` | `*` (dev only) | Comma-separated allowed origins |
| `PATHSNAP_API_URL` | `http://localhost:8000` | Used by the UI |

---

## 7. Two-laptop deployment

Full runbook: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). Short version:

```bash
# Laptop 1 — model + API + ground truth
PATHSNAP_DATA_ROOT=./data uvicorn backend.app:app --host 0.0.0.0 --port 8000

# Laptop 2 — UI only
PATHSNAP_API_URL=http://<LAPTOP1_IP>:8000 streamlit run ui/app.py --server.address 0.0.0.0
```

Before the demo: same Wi-Fi/hotspot, client isolation disabled, firewall port 8000 open, `curl http://<LAPTOP1_IP>:8000/health` succeeds, terminal logs visible for the evaluator.

---

## 8. Task board

Status: ✅ done · 🟨 in progress · ⬜ todo · 🚫 blocked

### Phase 0 — Foundation (done)

| ID | Owner | Task | Files | Definition of done | Status |
|---|---|---|---|---|---|
| T00 | D | Repo scaffold, environment, Makefile, smoke tests | repo root | `make test` passes on a clean clone | ✅ |
| T01 | D | README task board, ownership, setup/deployment instructions | `README.md` | This document | ✅ |
| T02 | A | `viva.md` skeleton + numbers ledger + Q&A bank | `viva.md` | Sections, owners, TBD policy | ✅ |
| T03 | A | Typed contracts and shared config | `pathsnap/contracts.py`, `config.py` | Imported by all modules | ✅ |
| T04 | A | Loader/validator + inventory | `pathsnap/io.py` | All 90 provided trips parse; anomalies logged | ✅ |
| T05 | A | Geo primitives (projection, distance, bearing, smoothing) | `pathsnap/geo.py` | Unit tests pass | ✅ |
| T06 | A | Baseline candidate generator + index | `pathsnap/candidates.py` | Truth arc recalled on fixture; radius sweep runnable | ✅ |
| T07 | D | API skeleton (`/health`, `/version`, `/trips`) | `backend/` | `curl` returns trip list | ✅ |
| T08 | D | UI skeleton + API client | `ui/app.py` | Lists trips, shows health | ✅ |
| T09 | C | Generic Viterbi + toy test | `pathsnap/hmm.py`, `tests/test_hmm.py` | Toy path decoded correctly | ✅ |

### Phase 1 — Before lunch (vertical slice)

| ID | Owner | Task | Files | Definition of done | Status |
|---|---|---|---|---|---|
| T10 | A | Dataset inventory + EDA report with figures and data-quality flags | `scripts/01_eda.py`, `results/eda/`, `docs/DATA_NOTES.md` | 6 required exploration outputs; flagged trips documented | ⬜ |
| T11 | A | Validate/refine candidate generation; radius/top-K sweep + miss-rate table | `pathsnap/candidates.py`, `results/eda/candidate_sweep.csv` | Miss rate reported per trip and vs radius | ⬜ |
| T12 | A | Route labels: nearest-route-arc + confidence flags; monotonic progress as diagnostic | `pathsnap/labels.py`, `results/labels/` | Visual check on ≥3 trips; low-confidence points excluded; flagged trips documented | ⬜ |
| T13 | A+B | Feature matrix v1 + feature-group ablation harness | `pathsnap/features.py` | FEATURE_NAMES frozen; ablation script produces CSV | ⬜ |
| T14 | A+B | Dataset builder + cache + GroupKFold splits | `pathsnap/dataset.py` | Reproducible split file; no trip crosses folds | ⬜ |
| T15 | B | Metrics + evaluation harness (candidate, top-1, miss, continuity, LCS, log loss) | `pathsnap/metrics.py` | Unit tests pass; metric definitions match `docs/EXPERIMENTS.md` | ⬜ |
| T16 | B | First 5 models + CV comparison table + logs | `training/`, `results/models/` | Bank of ≥5 models with CV top-1/macro-F1/log-loss/timing | ⬜ |
| T17 | C | HMM design note: states, emissions, transitions, gaps, complexity | `docs/HMM_DESIGN.md` | Reviewed by B; formulas match code | 🟨 implementation complete; B review pending |
| T18 | C | Network transition builder (corridor subgraph + cached Dijkstra) | `pathsnap/hmm.py` | Distance matrix unit-tested on toy graph | 🟨 tests complete; real-trip latency pending |
| T19 | D | Wire the real pipeline into `POST /predict` | `backend/service.py` | One train/student trip returns predictions through the API | ⬜ |
| T20 | D | UI: select trip → run → metrics + map + timings | `ui/app.py` | End-to-end demo on one trip | ⬜ |
| T21 | A | First Folium map: GPS + network + reference route | `pathsnap/visualize.py`, `results/maps/` | HTML committed; screenshot in `results/figures/` | ⬜ |

**Phase 1 gate (before lunch):** public repo, README, EDA, working candidate/label pipeline, ≥5 models training, one real end-to-end API/UI response, `viva.md` populated with real numbers so far. Tag `v0.1-phase1`.

### Phase 2 — After lunch (finalise)

| ID | Owner | Task | Files | Definition of done | Status |
|---|---|---|---|---|---|
| T22 | B | Complete model comparison (≥5 families) + selection rationale | `results/models/model_comparison.csv`, `report/03_models.md` | Actual results only; every model subsection answered | ⬜ |
| T23 | B | Optuna TPE study: search space, 5-fold GroupKFold, ≥30 trials, best config refit | `optimization/`, `results/optuna/` | `study.db`, `trials.csv`, history/importance plots | ⬜ |
| T24 | B | Tuned vs untuned comparison on CV + student-test (same metrics) | `results/models/tuned_vs_untuned.csv` | Decision documented; no tuning on evaluator data | ⬜ |
| T25 | C | HMM emission/transition tuning on permitted data only | `results/hmm/decoder_tuning.csv` | λ/σ/β/gap grid recorded; no evaluator leakage | ⬜ |
| T26 | C | Classifier-only vs HMM comparison + continuity/sequence analysis | `results/hmm/classifier_vs_hmm.csv`, `report/05_hmm.md` | Top-1, macro-F1, continuity, LCS-F1, failure cases | ⬜ |
| T27 | C | HMM failure analysis: missing candidates, gaps, junctions, over-smoothing | `results/hmm/failures.md`, figures | At least 3 concrete failure cases with maps | ⬜ |
| T28 | A | Folium maps for report: GPS, network, reference, prediction, layers/legend | `visualisation/`, `results/maps/` | Interactive HTML + screenshot in PDF | ⬜ |
| T29 | A | Report sections 2 (data/EDA) + figure set; compile full PDF | `report/` | PDF follows required structure | ⬜ |
| T30 | B | Report sections 3–4 (models, Optuna) + all graphs from results | `report/main.tex`/`report.md` | No fabricated curves; only real histories | ⬜ |
| T31 | D | Report section 7 (API/UI/deployment) + screenshots + API logs | `report/07_deployment.md`, `results/screenshots/` | Proof of endpoint use on Laptop 1 console | ⬜ |
| T32 | D | Integration: all 10 student-test trips through API/UI; timings; errors handled | `scripts/run_eval.py`, `results/integration/` | 10/10 pass; worst-case latency recorded | ⬜ |
| T33 | D | Two-laptop live test + logs + demo script + contingency cards | `docs/DEPLOYMENT.md`, `viva.md` | Live test on hotspot/LAN; screenshots captured | ⬜ |
| T34 | A | Claim/evidence audit: every number maps to a CSV/figure | `viva.md`, `report/` | Audit table clean; no TBD left | ⬜ |
| T35 | A+C | Mock viva round 1 (after S3) and round 2 (after S5); gaps assigned | `viva.md` | Everyone can explain one section they did not own | ⬜ |
| T36 | D | Final hardening: secrets scan, public-repo check, tag `v1.0-final` | repo root | Checklist complete; repo visible logged-out | ⬜ |
| T37 | D | Cross-agent protocol: read-first contract, live status, append-only log, ADRs, glossary, snapshot tool | `AGENTS.md`, `docs/STATUS.md`, `docs/AGENT_LOG.md`, `docs/DECISIONS.md`, `docs/GLOSSARY.md`, `scripts/agent_status.py` | `make status` works; PR template requires the updates | ✅ |

---

## 9. Milestones and gates

| Milestone | When | Gate |
|---|---|---|
| M0 Scaffold | Phase 0 | `make test` passes; contracts frozen; task board published |
| M1 Vertical slice | S1 (~10:15) | One trip: GPS → candidates → model → Viterbi → API → UI |
| M2 Phase-1 submission | Before lunch | Tag `v0.1-phase1`; public repo; ≥5 models; demo works |
| M3 Model freeze | After Optuna (~14:45) | Final model + decoder configs; results tables written |
| M4 Integration | ~16:00 | 10/10 student-test trips pass; two-laptop demo verified |
| M5 Submission | End of day | PDF report, `viva.md` complete, tag `v1.0-final` |

If the event runs over two days, insert a sleep break between M2 and M3; do **not** change the gates.

---

## 10. Working agreement

### Branches and PRs

| Owner | Branch | Reviews |
|---|---|---|
| A | `feat/a-data` | D |
| B | `feat/b-models` | C |
| C | `feat/c-hmm` | B |
| D | `feat/d-platform` | A |
| all | `docs/report` | A |

- One task per branch; keep PRs under ~2 hours of work.
- Never push directly to `main` after Phase 0; open a PR.
- A PR is mergeable only when the task's Definition of Done is met, `make test` passes, and the owner updated the README status.
- Commit messages: `type(scope): summary` — e.g. `feat(features): add smoothed heading features`.
- Tag `v0.1-phase1` before lunch and `v1.0-final` at submission.

### Contract-first rule

`pathsnap/contracts.py`, `config.py` and `pathsnap/metrics.py` are shared interfaces. Changes require review from A/B/C/D. Do not rename fields or change metric definitions silently.

### Experiment and results conventions

- Every run writes a CSV/JSON with: git commit, seed, data partition, model name, config, metrics, training time, inference time.
- Figures are generated by scripts from those files — never hand-edited.
- Store best params, Optuna trial history and decoder tuning grids in `results/`.
- Do not overwrite a previous experiment result; add a new row/file.

### Numbers and evidence policy

Every number in the README, report or `viva.md` must have a `source` path in the ledger in `viva.md`, for example:

| Claim | Value | Source | Owner |
|---|---|---|---|
| Final model CV top-1 | TBD | `results/models/model_comparison.csv` | B |
| Candidate miss rate (50 m) | TBD | `results/eda/candidate_sweep.csv` | A |
| HMM continuity improvement | TBD | `results/hmm/classifier_vs_hmm.csv` | C |
| API worst-case latency | TBD | `results/integration/latencies.csv` | D |

No estimates, no invented metrics, no synthetic training curves. This is an academic-integrity requirement, not a style preference.

### Ground-truth safety (non-negotiable)

- `backend/service.py` (prediction path) must never open a `.route` file.
- Scoring lives in `pathsnap/scoring.py` / `backend/scoring.py` and runs **after** the prediction is produced.
- The API response must not contain reference-route geometry or per-observation ground-truth labels.
- `ui/` must never receive or read `.route`.
- `tests/test_leakage.py` must assert that the prediction path does not open `.route`.
- No hardcoded trip outputs. The evaluator must see live requests in the API terminal logs.

### AI-use policy

AI assistance is allowed for scaffolding, debugging, ideation and editing, but every member must be able to explain the code and results. Disclose AI assistance in the report appendix. Never submit generated analysis as evidence of experiments that were not run.

---

## 11. Metrics used in this project

| Metric | Definition | Computed on |
|---|---|---|
| Candidate precision / recall / F1 | Binary classification over all (observation, candidate) pairs; positive = reference arc | CV folds, student-test |
| Macro F1 | Unweighted mean of positive/negative F1 | CV folds, student-test |
| Log loss | Calibrated candidate probabilities | CV folds, student-test |
| Observation Top-1 | Argmax candidate per observation equals aligned reference arc | CV folds, student-test |
| Candidate miss rate | Fraction of labeled observations whose reference arc never appears in the candidate set | Dataset diagnostics |
| Route continuity | Fraction of consecutive predicted arcs that are graph-connected (strict and relaxed) | Decoder comparison |
| Sequence LCS F1 | Precision/recall/F1 of arc sequence against reference using longest common subsequence | Decoder comparison |
| Training / inference time | Wall clock, same hardware, recorded per experiment | Model comparison |

Full protocol: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

---

## 12. Dataset facts to verify in T10

The scaffold investigation produced preliminary observations, but **the evidence
files were removed** so A can produce and own them. Do not quote counts until
`results/eda/inventory.csv` and `results/labels/summary.json` exist.

Structural facts (from the brief and the provided folders):

- 80 training trips, 10 student-test trips, 10 evaluator trips (not in the
  student package).
- Each trip ships its own local road network; different trips can cover
  different cities.
- `.track` = `(lon, lat, t_seconds)`; `.nodes` = `(lon, lat)` with row-index IDs;
  `.arcs` = `(from_node, to_node)` with row-index arc IDs; `.route` = reference
  arc sequence (server-side, not a per-point label).
- Some arcs are contracted chords; some routes are looped, reversed or partial.
  Label policy: nearest route arc with a confidence threshold; monotonic progress
  is a diagnostic (D-009).

A must fill: exact observation/node/arc counts, missing/invalid analysis,
sampling-gap distribution, label coverage, candidate miss rate, and the list of
genuinely unreliable reference routes. Every value goes in the `viva.md` ledger.

---

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `data/` missing / no trips | Check `PATHSNAP_DATA_ROOT`, folder layout, and file names |
| API starts but `/predict` returns 503 | `artifacts/model.joblib` is missing — run `make export` |
| UI cannot reach API | Use Laptop 1's LAN IP, open port 8000, disable client isolation, test `curl /health` |
| CORS error in UI | Set `PATHSNAP_CORS_ORIGINS` to the UI origin (or `*` for the demo) |
| Optuna not installed | `pip install optuna`; fallback must still be a Bayesian method, documented |
| XGBoost/LightGBM install fails | Expected on Python 3.14; use `HistGradientBoostingClassifier` |
| Evaluator trips not listed | Place them under `data/evaluator_test/` or set `PATHSNAP_EVALUATION_PARTITION` |
| Slow decoding on a large trip | Restrict shortest paths to the corridor subgraph and cache per candidate node |

---

## 14. Viva

Read [`viva.md`](viva.md) before every demo. It contains the system explanation, formulas, per-owner must-know questions, a 50-question bank, the demo script, the evidence ledger and the final checklist.

---

## 15. AI assistance disclosure

Template for the report appendix:

> AI tools were used for [scaffolding / debugging / grammar / plot boilerplate]. All experiments, measurements and reported results were executed by the team. Every member can explain the submitted code, tables and figures.

Update this section with the actual tools used and the actual scope before submission.

---

## 16. Viva maintenance protocol (mandatory for members and AI agents)

`viva.md` is not a one-time document. The viva is examined live, and a stale document is worse than no document because it creates contradictions between the code, the report and what a member says. Keeping it current is part of **every** task's Definition of Done.

### 16.1 When `viva.md` must be updated

Update it in the **same commit/PR** whenever you change any of the following:

- data loading, validation, partitions or leakage rules;
- preprocessing, projection, smoothing or gap handling;
- candidate generation, radius, top-K or miss-rate numbers;
- label alignment, confidence thresholds or coverage;
- features, the feature contract or an ablation result;
- a model, hyperparameter, metric definition or model-selection decision;
- Bayesian optimisation space, objective, trials or best configuration;
- HMM states, emissions, transitions, gap handling or decoder results;
- API endpoints, response schema, ground-truth isolation or logging;
- UI flow, demo script or deployment steps;
- any number that also appears in the report, README or presentation.

If a change makes a viva statement false, fix the statement in the same PR. Do not leave a “TODO later” that can reach the demo.

### 16.2 Section ownership

| Viva section | Owner | Must review when |
|---|---|---|
| §0 First viva pack | A | Any change to data, candidates, labels, preprocessing or features |
| §2–§6 Dataset, EDA, geo, candidates, labels | A | Data pipeline change |
| §7–§9 Features, models, evaluation | B | Feature/model/metric change |
| §10–§13 Selection, Optuna, HMM, decoder | B + C | Model or decoder change |
| §14–§16 API, UI, deployment | D | Backend, UI, runbook or log change |
| §17–§18 Figures, numbers ledger | A | Any new figure or number |
| §19–§24 Q&A, failures, demo, checklist, disclosure | all | Before every mock viva and submission |

### 16.3 Non-negotiable rules for AI agents

If you are an AI agent working in this repository:

1. **Read first.** Read `viva.md` §0 and the sections relevant to your task before editing code. Do not contradict the documented design without discussing it.
2. **Update in the same change.** When your code or experiment changes a documented behaviour, edit the relevant `viva.md` section and the README task status **in the same commit**.
3. **Never invent numbers.** If a metric has not been measured, write `TBD` and the path of the file that will contain it. Do not estimate, extrapolate or copy a plausible-looking value.
4. **Add evidence.** Every new number gets a row in the ledger (§18) with `value`, `source` and `owner`.
5. **Update the first-viva pack.** If the pipeline, feature list, label strategy or verified dataset facts change, update §0 — it is what the first viva will examine.
6. **Keep the task board honest.** Mark a task ✅ only when its Definition of Done is met and the evidence exists. Otherwise leave it 🟨 or ⬜.
7. **No data or secrets.** Never commit `data/`, `Student folder/`, `PathSnap_student_materials/`, `.env`, API keys or credentials.
8. **Run the checks.** `make test` and `make smoke` must pass before you finish.
9. **Record the change.** Add a row to the changelog (§16.5) with date, author/agent, sections and a one-line reason.
10. **Do not fabricate experiments.** AI may help edit and analyse, but the team must actually run the training, tuning and decoding that produce reported results.

### 16.4 Definition of Done for documentation

A PR that touches model/data/API behaviour is not done until:

- [ ] The affected `viva.md` sections are accurate and use the new code paths.
- [ ] Any new number is in §18 with a source path.
- [ ] `TBD` remains only where Phase 2 work is genuinely pending.
- [ ] The README task board status is updated.
- [ ] The first-viva pack (§0) is still correct.
- [ ] The demo script (§23) still works with the current API/UI.
- [ ] The changelog has a new row.
- [ ] `make test`, `make smoke` pass.
- [ ] No `data/`, secrets or credentials were added.

### 16.5 Changelog

Append a row every time the document changes. Keep the newest row at the top.

| Date | Author | Sections updated | Reason |
|---|---|---|---|
| 2026-11-18 | C (AI-assisted) | status · HMM tasks | Document Phase 1 HMM transition implementation and verification boundary |
| 2026-11-18 | scaffold (D) | all · §0 added | Initial viva pack + first-viva focus |
| | | | |

### 16.6 Audit commands

Run these before marking a milestone complete:

```bash
# 1. Find unresolved placeholders (must be intentional and explained)
grep -n "TBD" viva.md

# 2. Check that every ledger row has a source path
grep -n "^|" viva.md | grep -i "TBD" || true

# 3. Check the repo for secrets and data
git status --short
git ls-files | grep -E "^(data/|Student folder/|PathSnap_student_materials/)" && echo "DATA COMMITTED" || true
grep -RIn --exclude-dir=.git --exclude-dir=.venv "api_key\|password\|secret" . || true

# 4. Run the code checks
make test
make smoke
```

### 16.7 Mock-viva routine

- **After M2 (Phase 1):** 20 minutes. Each member explains §0 and one section they do not own. Record unanswered questions in §19.1 and assign an owner.
- **After M4 (integration):** 30 minutes. Full demo rehearsal (§23) plus five questions from each owner's bank. No notes for §0.
- **T-60 minutes:** run the final checklist (§22) and the audit commands above.

If a question cannot be answered, the answer is **not** “it is in the document”. The member must be able to explain it. Assign a follow-up task and update §19.1.

---

## 17. Cross-agent communication and state sharing

Four people and several AI agents work on this repo, sometimes concurrently. Git tells us *what* changed; it does not tell the next agent *what is true now*, *why a choice was made*, or *what to do next*. We use a four-layer protocol so no one has to read a long chat history.

### 17.1 Read order (start of every session)

1. [`AGENTS.md`](AGENTS.md) — the working contract.
2. [`docs/STATUS.md`](docs/STATUS.md) — authoritative current state, active claims, handoff.
3. Last three entries of [`docs/AGENT_LOG.md`](docs/AGENT_LOG.md) — what just happened.
4. [`viva.md` §0](viva.md) and the sections you will affect.
5. [`docs/DECISIONS.md`](docs/DECISIONS.md) — decisions you must not reverse silently.
6. This README's task board — your task ID and Definition of Done.

Then run `make status` and `make test`.

### 17.2 The files and their rules

| File | Answers | Write rule | Update when |
|---|---|---|---|
| `AGENTS.md` | How must I work? | Stable; team agreement only | Process changes |
| `docs/STATUS.md` | What is true now? | Small local edits; claim shared areas | Every session |
| `docs/AGENT_LOG.md` | What happened, and why? | **Append-only**, newest at the bottom | Every session |
| `docs/DECISIONS.md` | Why this design? | **Append-only** ADRs; supersede, never delete | Any constraining decision |
| `docs/GLOSSARY.md` | What do the words mean? | Add definitions; no silent redefinition | New term/metric |
| `viva.md` | How do we defend it? | Update with every behaviour/number change | Every task |
| `README.md` | Plan and ownership | Update task status | Every task |
| `scripts/agent_status.py` | Mechanical state | Generated | On demand (`make status`) |

### 17.3 Claiming work (avoid concurrent edits)

Before editing a shared area, add a row to **Active claims** in `docs/STATUS.md`:

| Area / files | Agent | Branch | Since | Status |
|---|---|---|---|---|
| `backend/`, `ui/`, integration | D | `feat/d-platform` | 2026-11-18 10:20 | active |

- One active claim per shared file/area.
- Remove the claim when the task is merged or handed off.
- If you need a claimed file, coordinate in the PR instead of editing it.
- GitHub Issues/Projects can complement this, but `STATUS.md` is the offline source of truth.

### 17.4 Session log entry (append at the bottom)

```markdown
## AL-YYYYMMDD-HHMM — <agent/member> — <title>

- **Task:** T16
- **Branch/commit:** `feat/b-models` @ `abc1234`
- **Files:** ...
- **Commands:** ...
- **Result:** value + source path
- **Interpretation:** what it means
- **Decision/ADR:** D-0xx
- **Next:** ...
- **Blockers:** ...
```

Rules: one entry per logical session; results include a path; failed experiments are logged; never rewrite old entries; append and rebase to avoid conflicts.

### 17.5 Handoff (end of session)

Update the **Current handoff** block in `docs/STATUS.md` with:

- one paragraph of state (works / broken);
- last verified commit;
- exact reproduction commands;
- dataset/model artifact versions;
- known failures;
- next three actions;
- “do not touch” list.

A task is not done if the next agent cannot reproduce it from `STATUS.md` alone.

### 17.6 Why not one giant file?

A single shared file becomes stale, and every agent edits the same lines, causing conflicts and silent contradictions. Separating documents by write frequency gives:

- **stability** (`AGENTS.md`, `DECISIONS.md`);
- **freshness** (`STATUS.md`);
- **history without conflicts** (`AGENT_LOG.md`, append-only);
- **shared language** (`GLOSSARY.md`);
- **defensibility** (`viva.md`).

The entry point is still one file: `AGENTS.md`.

### 17.7 For subagents and parallel agents

- A subagent reports a proposed log entry and file list to its parent; the parent commits.
- Do not push from a subagent unless explicitly asked.
- Never let two subagents write the same file.
- If you inherit a broken state, reproduce it and log it **before** changing code.
- Do not “clean up” another owner's work without a claim and review.

### 17.8 Audit

```bash
make status
make test
grep -n "TBD" viva.md
git status --short
```

`make status` prints git state, dataset/artifact versions, task-board progress, unresolved `TBD`s, code TODOs and the last log entries. Use `python scripts/agent_status.py --json` for machine-readable output.
