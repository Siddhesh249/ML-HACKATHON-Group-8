# HMM and Viterbi design note

> **Owner: C.** D seeded this note with the interface and the maths so the API
> and report can reference a stable design. The Phase 1 implementation is
> reviewed against this note (§11); decoder tuning and the comparison numbers
> are Phase 2 (T25–T27).

## 1. Why an HMM

The candidate classifier scores each observation independently. Nothing forces
consecutive predictions to be connected in the road graph, so a noisy point can
produce a teleport. The HMM adds sequence context and network feasibility.

## 2. Hidden states

The hidden state at observation `t` is a **specific candidate arc** from
`CandidateSet`. The state space is the candidate set `C_t` (≈15 arcs), not the
whole network. Keeping the candidate occurrence, rather than only the arc ID,
preserves the observation-to-arc correspondence when a route revisits an arc.

## 3. Emission

For candidate `j` at observation `t`:

```
p_t(j) = model probability that candidate j is the reference arc
e_t(j) = p_t(j) / sum_k p_t(k)            (candidate probabilities are not exclusive)
log e_t(j) -= lambda * d_t(j)^2 / (2 * sigma^2)
```

- `d_t(j)` is the point-to-arc distance in metres.
- `lambda` and `sigma_m` are decoder hyperparameters, tuned on train/student-test
  only.
- If all model probabilities are zero, fall back to a uniform distribution.

## 4. Transition

From candidate `i` at `t-1` to candidate `j` at `t`:

```
exit_offset_i  = (1 - ratio_i) * length_i
entry_offset_j = ratio_j * length_j
d_net(i, j)    = exit_offset_i + shortest_path(to_node_i, from_node_j) + entry_offset_j
```

- `ratio` is the projection ratio along the arc.
- The shortest path uses a directed corridor subgraph around the trace.
- If `d_net` is infinite (no path) or exceeds `max_speed_mps * dt + slack`, the
  transition is infeasible (`-inf`).
- Otherwise:
  ```
  log a_ij = -|d_net - gps_displacement| / beta_m - uturn_penalty * [j is reverse of i]
  ```
- Each row is normalised with a log-sum-exp so `a_ij` is a proper distribution
  over the next candidates.
- If every target is infeasible for a previous state, fall back to a weak
  uniform row instead of killing the state.

The implementation builds this transition geometry once per `DecoderContext`:

- `pathsnap.network.corridor_subgraph` keeps nodes within the configured margin
  of the smoothed GPS trace and adds endpoints of incident arcs. The resulting
  directed CSR graph is weighted by projected arc length.
- `DecoderContext.distances_from` caches one bounded directed Dijkstra result
  per source node. Multiple candidate transitions leaving the same arc therefore
  reuse the same shortest-path row.
- Candidate occurrences remain separate even when their `arc_id` is repeated.
  Same-arc transitions use forward projection progress directly; they do not
  route through the graph and backwards progress is infeasible.
- A reverse directed arc is still a legal transition when it is reachable, but
  receives `uturn_penalty`. This keeps the decoder from preferring a U-turn
  merely because its geometry is close.
- A large time gap returns an equal-score transition matrix, which removes the
  speed and displacement constraint for that observation boundary.
- The displacement cap is
  `max(max_speed_mps * dt + transition_slack_m, 300 m)`: the 300 m floor keeps a
  single 1 s urban step from being rejected while the speed term still governs
  longer intervals.
- Cached Dijkstra rows are bounded at 5 km. That exceeds
  `max_speed_mps * gap_reset_s` (≈1.65 km), so the bound cannot reject a
  transition that survives the gap rule, and it keeps the corridor cache small.

## 5. Viterbi

```
delta_1(j) = log e_1(j) + log start_prior(j)
delta_t(j) = log e_t(j) + max_i [ delta_{t-1}(i) + log a_ij ]
psi_t(j)   = argmax_i [ delta_{t-1}(i) + log a_ij ]
```

Backtrack from `argmax_j delta_T(j)`. Everything is in log space.

Implementation: `pathsnap/hmm.py::viterbi_layers` accepts a list of emission
vectors and transition matrices, so layers may have different candidate counts.

## 6. Gaps, missing candidates and invalid transitions

| Case | Handling |
|---|---|
| `dt >= gap_reset_s` | Reset: log-uniform transition (no distance constraint) |
| Candidate set empty at `t` | Skip the state layer; carry the previous state; output `-1` for that observation |
| No path between candidates | `-inf`, decoder must choose another candidate |
| All transitions from a state infeasible | Weak uniform fallback row |
| Route revisits an arc | State keeps the candidate occurrence; emissions are per observation |
| Very large trip | Corridor subgraph + cached Dijkstra; record latency with the benchmark command in `results/hmm/` |

## 7. Tuning protocol

Parameters: `lam`, `sigma_m`, `beta_m`, `max_speed_mps`, `gap_reset_s`,
`uturn_penalty`.

- Grid or Bayesian search on training/student-test only.
- Objective: mean observation Top-1 plus route continuity (record the trade-off;
  do not tune to a single hand-picked trip).
- Store `results/hmm/decoder_tuning.csv` and the chosen values in
  `artifacts/hmm_config.json`.
- Never use evaluator trips.

## 8. Comparison protocol

Run the same candidate sets, model probabilities and labels in both modes:

- classifier-only: per-observation argmax;
- HMM: Viterbi path.

Report per trip and aggregate:

- observation Top-1;
- strict/relaxed route continuity;
- LCS sequence precision/recall/F1 and exact match;
- arc-set F1;
- inference time;
- number of impossible jumps fixed;
- concrete failure cases (miss, gap, junction, over-smoothing).

## 9. Interface

```python
from pathsnap import hmm

cfg = hmm.HMMConfig(lam=..., sigma_m=..., beta_m=..., gap_reset_s=...)
prediction = hmm.decode(trip, candidates, probs, mode="hmm", cfg=cfg)
prediction.arc_ids          # (N,) one arc per observation, -1 if unmatched
prediction.route_arcs()     # deduped arc sequence
prediction.timings_ms       # filled by the caller (API)
```

The API calls `hmm.decode`; `backend/service.py` must not duplicate the decoder.

## 10. Phase 1 verification

`tests/test_hmm.py` covers the transition contract with synthetic directed
networks: forward same-arc progress, reversed progress, unreachable targets,
gap resets, U-turn penalties, and empty candidate layers. The same tests also
exercise the public decoder path so an empty candidate observation remains
unmatched (`-1`) rather than being fabricated.

For a real-trip latency record, run:

```bash
python -m scripts.benchmark_hmm --data-root ./data --partition train --trip-id <largest-trip-id>
```

The command writes `results/hmm/transition_benchmark.csv` with the trip,
corridor size, number of cached Dijkstra sources, warm transition timings, and
the current commit. It refuses to invent a result when the raw dataset is not
available.

## 11. Review record

- **2026-11-18 — code-vs-doc cross-review (T17, D cross-covering B).** Every
  formula, config field, fallback, gap rule and interface in this note was
  checked against `pathsnap/hmm.py`, `pathsnap/network.py` and
  `pathsnap/config.py`. Two implementation details were missing from §4 (the
  300 m displacement floor and the 5 km Dijkstra bound); both are now
  documented. No code change was required. The synthetic tests in
  `tests/test_hmm.py` cover the transition contract described here, and the
  real-trip latency record is in `results/hmm/transition_benchmark.csv`.
