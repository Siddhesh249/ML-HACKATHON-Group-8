# AGENTS.md — read this first

This repository is worked on by **four humans and multiple AI agents**, sometimes
concurrently. Git records *what* changed. This file and the four companion
documents record **what is true now, why it is true, and what the next agent
should do**.

If you are an AI agent: follow this protocol before touching code.

---

## 0. 60-second briefing

**Project:** PathSnap — reconstruct vehicle routes from noisy GPS by matching
observations to a per-trip road network. Candidate classifier → HMM/Viterbi.
FastAPI on Laptop 1, Streamlit on Laptop 2. Public repo + PDF report.

**Current phase:** see [`docs/STATUS.md`](docs/STATUS.md). Do not trust a phase
description in an old chat; the status file is authoritative.

**Read in this order (5–10 minutes):**

1. `AGENTS.md` (this file) — the contract.
2. `docs/STATUS.md` — what works, what is broken, active claims, next actions.
3. The last three entries of `docs/AGENT_LOG.md` — what just happened.
4. `viva.md` — the short explanations, numbers ledger, Q&A bank and demo script.
5. `docs/DECISIONS.md` — the decisions you must not silently reverse.
6. `README.md` task board — your task ID, owner, definition of done.

Then run:

```bash
make status
make test
```

---

## 1. The communication system

One file is not enough: a single “everything” file becomes stale and every agent
edits the same lines, which causes conflicts and silent contradictions. We use
**one entry point plus documents separated by write frequency**.

| File | Question it answers | Write rule | How often |
|---|---|---|---|
| `AGENTS.md` | How must I work here? | Stable; change only by team agreement | Rare |
| `docs/GLOSSARY.md` | What do these words mean? | Add definitions, do not redefine silently | Occasional |
| `docs/STATUS.md` | What is true **right now**? | Small edits; one editor at a time; claim shared areas | Every session |
| `docs/AGENT_LOG.md` | What happened, and why? | **Append-only**, newest at the bottom; never rewrite history | Every session |
| `docs/DECISIONS.md` | Why is the system built this way? | **Append-only** ADRs; supersede, never delete | When a decision is made |
| `README.md` | What is the plan and who owns it? | Update task status | Per task |
| `viva.md` | How do we explain and defend it? | Only when a reported number or behaviour changes | When it changes |
| `scripts/agent_status.py` | What is the mechanical state? | Generated output | On demand |

**Rule of thumb:** code diff = *what*; `AGENT_LOG` = *when and why*; `STATUS` =
*now*; `DECISIONS` = *why not something else*; `viva.md` = *how we defend it*.

### 1.1 `docs/STATUS.md` — the live state board

- Header: last updated, updated by, phase, current focus.
- Health table: tests, EDA, dataset, models, artifact, API, UI, decoder.
- **Active claims**: who is touching which area, on which branch.
- What works / what is broken / known issues.
- Next actions with owner and task ID.
- Current handoff: exact commands to reproduce the state.
- Open questions/blockers.

Edit it in **small, localised changes**. Do not rewrite the whole file. If two
agents must edit it, one claims it first (see §5).

### 1.2 `docs/AGENT_LOG.md` — append-only lab notebook

Every session appends one entry at the **bottom**:

```markdown
## AL-YYYYMMDD-HHMM — <agent/member> — <one-line title>

- **Task:** T16 (README task board)
- **Branch/commit:** `feat/b-models` @ `abc1234`
- **Files:** `pathsnap/models.py`, `results/models/model_comparison.csv`
- **Commands:** `make dataset`, `python scripts/03_train_models.py ...`
- **Result:** CV Top-1 0.xxx, macro F1 0.xxx (source CSV)
- **Interpretation:** what this means for the design
- **Decision/ADR:** D-0xx if any
- **Next:** the next concrete step
- **Blockers:** none / exact blocker and who can unblock
```

Rules:

- Append; never edit a previous entry (typo fixes are the only exception).
- One entry per logical session, not per commit.
- Results must include a **path**, not just a number.
- Failed experiments are logged; they are part of the record.
- If you used an AI agent, say so and name the scope.

### 1.3 `docs/DECISIONS.md` — Architecture Decision Records

Use an ADR when a choice constrains other people: data format, metric
definition, model family, API contract, dependency, splitting strategy, leakage
policy, ownership boundary.

Each ADR is immutable once accepted. To change it, add a new ADR that
**supersedes** the old one and update the index. Never rewrite history.

### 1.4 `docs/GLOSSARY.md`

Shared vocabulary prevents subtle disagreements (“coverage”, “miss rate”,
“continuity”, “Top-1”). If you introduce a new metric, define it here and in
`pathsnap/metrics.py`.

### 1.5 `scripts/agent_status.py`

Mechanical snapshot: git branch/commit/dirty files, dataset metadata, artifact
version, partition counts, unresolved `TBD`s, task-board progress, last log
entries. Run it at the start and end of every session:

```bash
make status
python scripts/agent_status.py --json
```

---

## 2. Start-of-session checklist

1. Read the files in §0.
2. `make status` and `make test` — know the baseline before changing it.
3. Pick a task from `README.md`; read its Definition of Done.
4. **Claim it** in `docs/STATUS.md` (or a GitHub issue if the team uses one):
   add a row to *Active claims* with your name, task ID, branch, timestamp.
5. Create a branch from the latest `main`:
   `git switch -c agent/<name>/<task-id>-<slug>`.
6. Reproduce the current baseline command for your area (see STATUS handoff).
7. Read the `viva.md` paragraph and Q&A row you will affect.

Do not start by editing a shared file. Claim first.

---

## 3. End-of-session checklist

1. `make test` and `make smoke` pass.
2. Append one entry to `docs/AGENT_LOG.md` with the real commands and results.
3. Update `docs/STATUS.md`: health row(s), remove your claim, set next actions,
   and update the handoff block with exact commands.
4. Add an ADR if you made a constraining decision.
5. Update `README.md` task status (✅ only when the DoD is met).
6. Update `viva.md` only where your work changed a fact: the ledger row (§17) and
   the one paragraph or Q&A row that quotes it. Do not restructure the file.
7. Commit the code, the log, the status, the README and the viva **together**.
8. Merge to `main` when tests pass (no PR required).
9. Hand off with a one-paragraph summary in the commit message.

A change is not done if the next agent cannot reproduce it from STATUS alone.

---

## 4. Commit and merge rules

- Commit format: `type(scope): summary` (`feat`, `fix`, `docs`, `exp`, `test`,
  `chore`). Example: `exp(hmm): tune beta on grouped CV`.
- One logical change per commit; no mixed code + unrelated docs.
- Keep changes small (≤ ~2 hours of work) and reference a task ID.
- A commit/merge includes: what changed, commands run, results with paths,
  checklist (tests, log, status, viva, no data/secrets, ground-truth safety).
- **No PR and no reviewer.** Merge directly to `main` when the checklist is
  green. For a shared contract change (`pathsnap/config.py`, `contracts.py`,
  `metrics.py`, API schema), mention the affected owners in the commit message
  so they can object, but do not block on approval.
- Pull/rebase before pushing: `git pull --rebase origin main`.
- Never force-push `main`; never commit data, caches or `.env`.

---

## 5. Concurrent agents

### 5.1 Claiming

Before editing a shared area, add a claim to `docs/STATUS.md`:

| Area / files | Agent | Branch | Since | Status |
|---|---|---|---|---|
| `backend/`, `ui/`, `docs/DEPLOYMENT.md` | D | `feat/d-platform` | 2026-11-18 10:20 | active |

Areas are the ownership blocks in `README.md` §1. If an area is claimed, do not
edit it; coordinate or work on another task. Removing a claim is part of the
end-of-session checklist.

### 5.2 Avoiding conflicts

- Append-only files (`AGENT_LOG`, `DECISIONS`) conflict only if two agents add
  at the same location. **Always append at the bottom**, then rebase.
- `STATUS.md` edits must be small and localised; never reformat the file.
- If a conflict happens, keep **both** facts and merge them; never delete
  another agent's result.
- Rebase before pushing: `git pull --rebase origin main`.
- If two agents need the same shared file, coordinate in the commit message or
  `STATUS.md`; rebase and merge when tests pass.

### 5.3 Subagents and background agents

- A subagent reports back to its parent with a proposed `AGENT_LOG` entry and a
  list of files changed. The parent decides what to commit.
- Do not push from a subagent unless the parent explicitly asks.
- Never let two subagents write the same file.

### 5.4 If you inherit a broken state

1. Read the last three log entries and the current handoff.
2. Reproduce the failure with the exact command.
3. Write a log entry with the failure and the fix attempt **before** changing
   code.
4. Do not “clean up” another owner's work without a claim and a note in the log.

---

## 6. Repository map and owners

See `README.md` §1 for the full table. Summary:

| Area | Owner | Key paths |
|---|---|---|
| Data, EDA, geo, candidates, labels, maps | A | `pathsnap/io.py`, `geo.py`, `network.py`, `candidates.py`, `labels.py`, `visualize.py`, `results/eda/` |
| Features, dataset, models, metrics, Optuna | B | `pathsnap/features.py`, `dataset.py`, `models.py`, `metrics.py`, `training/`, `optimization/` |
| HMM/Viterbi, decoder tuning, sequence metrics | C | `pathsnap/hmm.py`, `scoring.py`, `results/hmm/` |
| API, UI, deployment, integration, demo | D | `backend/`, `ui/`, `docs/DEPLOYMENT.md`, `scripts/run_eval.py` |

Cross-cutting contracts (`config.py`, `contracts.py`, `metrics.py`, API schema)
are announced in the commit message and covered by tests; there is no approval gate.

---

## 7. Non-negotiable rules

1. **Ground truth stays server-side.** The prediction path must not read
   `.route`; the UI must not receive the reference route or labels.
2. **No fabricated results.** Every number comes from `results/` and appears in
   the `viva.md` ledger. Use `TBD` until measured.
3. **No data or secrets in git.** `data/`, `Student folder/`,
   `PathSnap_student_materials/`, `.env`, keys and caches stay out.
4. **No evaluator leakage.** Never use evaluator trips for training, tuning,
   thresholds, feature choice or model selection.
5. **Tests before handoff.** `make test` and `make smoke` pass.
6. **Update the docs in the same commit.** Code + log + status + README; add a
   viva ledger row only when a reported number changed.
7. **Respect ownership.** Claim before editing another area.
8. **Explain it.** A human must be able to defend every claim in the viva.

---

## 8. Commands

```bash
make status        # mechanical snapshot
make test          # unit/contract/leakage tests
make smoke         # import check
make eda           # EDA outputs
make dataset       # build cache/dataset.npz
make train         # model comparison
make tune          # Optuna
make eval-hmm      # classifier vs HMM
make export        # artifacts/model.joblib
make api           # Laptop 1
make ui            # Laptop 2
python scripts/run_eval.py --partition student_test
```

See `README.md` §5–§7 and `docs/DEPLOYMENT.md`.

---

## 9. Definition of Done (every task)

- [ ] Task DoD from `README.md` met with evidence.
- [ ] `make test` and `make smoke` pass.
- [ ] `docs/AGENT_LOG.md` entry appended with commands and result paths.
- [ ] `docs/STATUS.md` updated (health, claim removed, next actions, handoff).
- [ ] `README.md` task status updated.
- [ ] `viva.md` ledger row (and any sentence quoting it) updated if a number changed.
- [ ] ADR added if a constraining decision was made.
- [ ] No data, secrets or caches committed.
- [ ] Changes merged to `main` when tests pass (no PR, no reviewer).

---

## 10. Handoff block (copy into STATUS)

```markdown
### Handoff — <date/time> — <agent>

**State:** what works and what is broken in one paragraph.
**Last verified commit:** `<sha>`
**Commands that reproduce it:**
```bash
...
```
**Artifacts:** `artifacts/model.joblib` version `<...>`, dataset `<...>`
**Known failures:** ...
**Next three actions:** 1) ... 2) ... 3) ...
**Do not touch:** ...
```

---

## 11. Phase snapshot (update only at milestones)

| Milestone | Status | Evidence |
|---|---|---|
| M0 scaffold | ✅ | `make test` 27 passed; scaffold committed |
| M1 vertical slice | ✅ | T10–T21 complete; live API/UI; 10/10 student-test trips |
| M2 phase-1 | ✅ | tag `v0.1-phase1`; public repo; 8 model families; demo works |
| M3 model freeze | ⬜ | — |
| M4 integration | ⬜ | — |
| M5 submission | ⬜ | — |

The authoritative version is `docs/STATUS.md`.
