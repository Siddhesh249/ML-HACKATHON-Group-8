# Team, task board and working agreement

> Moved here from `README.md` so the README can describe the project itself.
> This file is the task board read by `make status` and referenced by
> `AGENTS.md`. Section numbers are kept from the original README.


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

## 8. Task board

Status: ✅ done · 🟨 in progress · ⬜ todo · 🚫 blocked

### Phase 0 — Foundation (done)

| ID | Owner | Task | Files | Definition of done | Status |
|---|---|---|---|---|---|
| T00 | D | Repo scaffold, environment, Makefile, smoke tests | repo root | `make test` passes on a clean clone | ✅ |
| T01 | D | README task board, ownership, setup/deployment instructions | `README.md` | This document | ✅ |
| T02 | A | `viva.md` skeleton + numbers ledger + Q&A bank | `viva.md` | Sections, owners, TBD policy | ✅ A skeleton; D simplified to a 350-line pack (explanations, ledger, rapid-fire Q&A, demo, maintenance) |
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
| T10 | A | Dataset inventory + EDA report with figures and data-quality flags | `scripts/01_eda.py`, `results/eda/`, `docs/DATA_NOTES.md` | 6 required exploration outputs; flagged trips documented | ✅ A: `results/eda/` (inventory, quality checks, flags, 9 figures), maps via T21, `docs/DATA_NOTES.md` |
| T11 | A | Validate/refine candidate generation; radius/top-K sweep + miss-rate table | `pathsnap/candidates.py`, `results/eda/candidate_sweep.csv` | Miss rate reported per trip and vs radius | ✅ A: `scripts/01b_candidate_sweep.py` → `results/eda/candidate_sweep*.csv`; 60 m/K=15 kept (0.092% train) |
| T12 | A | Route labels: nearest-route-arc + confidence flags; monotonic progress as diagnostic | `pathsnap/labels.py`, `results/labels/` | Visual check on ≥3 trips; low-confidence points excluded; flagged trips documented | ✅ A: `results/labels/coverage.csv`, 5 visual checks; unreliable: train 33, 54 |
| T13 | A+B | Feature matrix v1 + feature-group ablation harness | `pathsnap/features.py` | FEATURE_NAMES frozen; ablation script produces CSV | ✅ B: `scripts/03b_feature_ablation.py` → `results/models/feature_ablation.csv`, `feature_importance.csv`; v1 unchanged |
| T14 | A+B | Dataset builder + cache + GroupKFold splits | `pathsnap/dataset.py` | Reproducible split file; no trip crosses folds | ✅ B: `results/models/cv_splits.csv`, `dataset_summary.json` (all checks true) |
| T15 | B | Metrics + evaluation harness (candidate, top-1, miss, continuity, LCS, log loss) | `pathsnap/metrics.py` | Unit tests pass; metric definitions match `docs/EXPERIMENTS.md` | ✅ B: per-fold val/train metrics; `tests/test_models.py`; 32 tests pass |
| T16 | B | First 5 models + CV comparison table + logs | `training/`, `results/models/` | Bank of ≥5 models with CV top-1/macro-F1/log-loss/timing | ✅ B: 8 families + baseline in `results/models/model_comparison.csv`; preliminary artifact exported |
| T17 | C | HMM design note: states, emissions, transitions, gaps, complexity | `docs/HMM_DESIGN.md` | Reviewed by B; formulas match code | ✅ reviewed against code 2026-11-18 (D cross-cover for B); two implementation details added to §4; record in §11. B review: formulas match + 4 follow-ups for T25 (same-arc backward jitter forbids 18.8% of true steps) — `docs/reviews/T17_hmm_review_B.md` |
| T18 | C | Network transition builder (corridor subgraph + cached Dijkstra) | `pathsnap/hmm.py` | Distance matrix unit-tested on toy graph | ✅ tests + real-trip benchmark: `results/hmm/transition_benchmark.csv` (5.17 s / 19,313 obs) |
| T19 | D | Wire the real pipeline into `POST /predict` | `backend/service.py` | One train/student trip returns predictions through the API | ✅ D: live `/predict` on 00000052 + 00000056; log `results/screenshots/api_log_excerpt.txt` |
| T20 | D | UI: select trip → run → metrics + map + timings | `ui/app.py` | End-to-end demo on one trip | ✅ D: dark-theme UI; screenshots `results/screenshots/ui_dark_hmm_trip00000000.png` + classifier mode |
| T21 | A | First Folium map: GPS + network + reference route | `pathsnap/visualize.py`, `results/maps/` | HTML committed; screenshot in `results/figures/` | ✅ A: `visualisation/make_maps.py` → `results/maps/*.html`, `results/figures/map_*.png` |

**Phase 1 gate (before lunch):** public repo, README, EDA, working candidate/label pipeline, ≥5 models training, one real end-to-end API/UI response, `viva.md` populated with real numbers so far. Tag `v0.1-phase1`.

### Phase 2 — After lunch (finalise)

| ID | Owner | Task | Files | Definition of done | Status |
|---|---|---|---|---|---|
| T22 | B | Complete model comparison (≥5 families) + selection rationale | `results/models/model_comparison.csv`, `report/03_models.md` | Actual results only; every model subsection answered | ✅ B: 8 models × 7 questions in `report/03_models.md`; student-test in `results/models/student_test_comparison.csv` |
| T23 | B | Optuna TPE study: search space, 5-fold GroupKFold, ≥30 trials, best config refit | `optimization/`, `results/optuna/` | `study.db`, `trials.csv`, history/importance plots | ✅ B: 40 TPE trials × GroupKFold(5), `results/optuna/`; best CV Top-1 0.8872 vs 0.8868 untuned |
| T24 | B | Tuned vs untuned comparison on CV + student-test (same metrics) | `results/models/tuned_vs_untuned.csv` | Decision documented; no tuning on evaluator data | ✅ B: keep untuned (gain < fold noise, guardrails worse) — `results/models/tuned_vs_untuned_decision.json` |
| T25 | C | HMM emission/transition tuning on permitted data only | `results/hmm/decoder_tuning.csv` | λ/σ/β/gap grid recorded; no evaluator leakage | ✅ v1: 48 student-test configs (C); **v2: same-arc tolerance + wider grid selected on 24 train trips (OOF), held-out student-test Top-1 0.930 vs 0.900 classifier** — `results/hmm/decoder_tuning_v2*`, `artifacts/hmm_config.json` |
| T26 | C | Classifier-only vs HMM comparison + continuity/sequence analysis | `results/hmm/classifier_vs_hmm.csv`, `report/05_hmm.md` | Top-1, macro-F1, continuity, LCS-F1, failure cases | ✅ 20 rows across 10 student-test trips |
| T27 | C | HMM failure analysis: missing candidates, gaps, junctions, over-smoothing | `results/hmm/failures.md`, figures | At least 3 concrete failure cases with maps | ✅ 3 distinct cases + Folium maps |
| T28 | A | Folium maps for report: GPS, network, reference, prediction, layers/legend | `visualisation/`, `results/maps/` | Interactive HTML + screenshot in PDF | ✅ A: `visualisation/make_maps.py --predict` → classifier-vs-HMM maps for student-test 52/56/77 (`results/maps/*_classifier_vs_hmm.html`, full + zoom PNGs in `results/figures/`, metrics `results/maps/prediction_maps.csv`); HMM on default config — re-run after T25 |
| T29 | A | Report sections 2 (data/EDA) + figure set; compile full PDF | `report/` | PDF follows required structure | ⬜ |
| T30 | B | Report sections 3–4 (models, Optuna) + all graphs from results | `report/main.tex`/`report.md` | No fabricated curves; only real histories | ✅ B: `report/03_models.md`, `report/04_optuna.md`; figures in `results/figures/`, `results/optuna/` |
| T31 | D | Report section 7 (API/UI/deployment) + screenshots + API logs | `report/07_deployment.md`, `results/screenshots/` | Proof of endpoint use on Laptop 1 console | ✅ D preliminary: report + UI/API screenshots + logs; refresh after tuned artifacts |
| T32 | D | Integration: all 10 student-test trips through API/UI; timings; errors handled | `scripts/run_eval.py`, `results/integration/` | 10/10 pass; worst-case latency recorded | ✅ D: 10/10 both modes; error-state screenshot `results/screenshots/ui_dark_api_offline.png`; `run_log.csv`, `summary.json`, `stage_timings.csv` |
| T33 | D | Two-laptop live test + logs + demo script + contingency cards | `docs/DEPLOYMENT.md`, `viva.md` | Live test on hotspot/LAN; screenshots captured | 🟨 local two-process loopback done; physical Laptop 2 pending |
| T34 | A | Claim/evidence audit: every number maps to a CSV/figure | `viva.md`, `report/` | Audit table clean; no TBD left | ✅ A: `scripts/audit_claims.py` → `results/audit/summary.json` `clean: true` — 26 ledger claims, 25 auto-verified + 1 by hand; 0 TBD; 0 broken references |
| T35 | A+C | Mock viva round 1 (after S3) and round 2 (after S5); gaps assigned | `viva.md` | Everyone can explain one section they did not own | ⬜ needs the people: A and C run the mock viva from `viva.md` §10 |
| T36 | D | Final hardening: secrets scan, public-repo check, tag `v1.0-final` | repo root | Checklist complete; repo visible logged-out | ⬜ |
| T37 | D | Cross-agent protocol: read-first contract, live status, append-only log, ADRs, glossary, snapshot tool | `AGENTS.md`, `docs/STATUS.md`, `docs/AGENT_LOG.md`, `docs/DECISIONS.md`, `docs/GLOSSARY.md`, `scripts/agent_status.py` | `make status` works; merge checklist in README | ✅ |

---

## 9. Milestones and gates

| Milestone | When | Gate | Status |
|---|---|---|---|
| M0 Scaffold | Phase 0 | `make test` passes; contracts frozen; task board published | ✅ |
| M1 Vertical slice | S1 (~10:15) | One trip: GPS → candidates → model → Viterbi → API → UI | ✅ T10–T21 |
| M2 Phase-1 submission | Before lunch | Tag `v0.1-phase1`; public repo; ≥5 models; demo works | ✅ `v0.1-phase1` |
| M3 Model freeze | After Optuna (~14:45) | Final model + decoder configs; results tables written | ⬜ |
| M4 Integration | ~16:00 | 10/10 student-test trips pass; two-laptop demo verified | 🟨 10/10 pass; physical laptop pending |
| M5 Submission | End of day | PDF report, `viva.md` complete, tag `v1.0-final` | ⬜ |

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
| Final model CV top-1 | 0.887 HistGradientBoosting (student-test 0.882) | `results/models/model_comparison.csv` | B |
| Candidate miss rate (60 m + fallbacks, top-K 15) | 0.092% train / 0.032% student-test | `results/eda/candidate_sweep.csv` | A |
| HMM vs classifier-only (tuned) | Top-1 0.900 → 0.930, strict continuity 0.755 → 0.906 (mean of 10 held-out student-test trips) | `results/hmm/classifier_vs_hmm.csv` | C |
| API worst-case latency | 1.85 s (HMM mode, trip `00000056`, 3,958 points) | `results/integration/run_log.csv` | D |

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

## 15. AI assistance disclosure

Template for the report appendix:

> AI tools were used for [scaffolding / debugging / grammar / plot boilerplate]. All experiments, measurements and reported results were executed by the team. Every member can explain the submitted code, tables and figures.

Update this section with the actual tools used and the actual scope before submission.

---

## 16. Viva maintenance

`viva.md` is short on purpose. A stale or bloated document creates contradictions
between the code, the report and what a member says, so keep edits small and
factual.

### 16.1 What to update, and when

Update in the same commit, and only when your work actually changed something:

| Change | Update |
|---|---|
| A reported number | Its ledger row in `viva.md` §17, plus any sentence quoting it |
| Pipeline, preprocessing, features, model choice, decoder, API contract | The one paragraph that describes it |
| A question you could not answer | A one-line row in the Q&A bank (§18) |
| Demo steps | The demo script (§19) |

Do not restructure the file, add sections, or keep a second copy of a number.
Quote from the ledger.

### 16.2 Section ownership

| Area | Owner |
|---|---|
| Task, data, EDA, preprocessing, candidates, labels, features, maps | A |
| Models, metrics, selection, Optuna | B |
| HMM, decoder tuning, classifier-vs-HMM | C |
| API, UI, deployment, demo evidence | D |

A compiles before a mock viva; D checks the demo evidence. Everyone must be able
to explain their own section and one they do not own.

### 16.3 Rules for AI agents

1. **Read first.** Read `AGENTS.md`, `docs/STATUS.md` and the `viva.md` paragraph
   you will affect before editing code.
2. **Edit narrowly.** Only the affected paragraph, ledger row or Q&A row.
3. **Never invent numbers.** Unmeasured means `TBD` plus the path that will
   contain it.
4. **Add evidence.** Every new number gets a ledger row with value, source and
   owner.
5. **No data or secrets.** Never commit `data/`, course materials, `.env` or keys.
6. **Run the checks.** `make test` and `make smoke` before you finish.

### 16.4 Definition of Done for documentation

- [ ] Ledger row updated for every changed number.
- [ ] Any sentence quoting that number is still true.
- [ ] `TBD` remains only for planned Phase 2 work.
- [ ] README task status updated.
- [ ] `make test` and `make smoke` pass.

### 16.5 Audit

```bash
grep -n "TBD" viva.md          # every hit must be planned work
make test && make smoke
```

### 16.6 Mock viva

- **After Phase 1:** 20 minutes. Each member explains their section and one
  foreign section; add unanswered questions to §18.
- **After integration:** 30 minutes. Full demo rehearsal (`viva.md` §19) plus
  five questions per owner.
- **T-60 minutes:** run the checklist in `viva.md` §20.

---

## 17. Cross-agent communication and state sharing

Four people and several AI agents work on this repo, sometimes concurrently. Git tells us *what* changed; it does not tell the next agent *what is true now*, *why a choice was made*, or *what to do next*. We use a four-layer protocol so no one has to read a long chat history.

### 17.1 Read order (start of every session)

1. [`AGENTS.md`](AGENTS.md) — the working contract.
2. [`docs/STATUS.md`](docs/STATUS.md) — authoritative current state, active claims, handoff.
3. Last three entries of [`docs/AGENT_LOG.md`](docs/AGENT_LOG.md) — what just happened.
4. [`viva.md`](viva.md) — the explanations and numbers for your task.
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
