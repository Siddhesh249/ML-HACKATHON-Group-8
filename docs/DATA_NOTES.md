# Dataset notes — verified facts and known limitations

> Re-run these checks in T10 and update this file if the numbers change. Every
> figure quoted here must also appear in `viva.md` §0 / §4 with a source.

## 1. Layout

```
data/
├── train/<trip_id>/<trip_id>.{track,nodes,arcs,route}           # 80 trips
├── student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}    # 10 trips
└── evaluator_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}  # 10 reserved trips (not in the student package)
```

The raw files are **not committed** (91 MB of course material). The API reads
them from `PATHSNAP_DATA_ROOT`. `data/dataset.npz`, `cache/` and `data/` are
git-ignored.

## 2. File semantics

| File | Columns | Semantics | Important detail |
|---|---|---|---|
| `.track` | 3: `lon`, `lat`, `t_seconds` | Noisy GPS trace, nominally 1 Hz | Not every step is 1 s; gaps exist |
| `.nodes` | 2: `lon`, `lat` | Road-network nodes | **Node ID = 0-based row index** |
| `.arcs` | 2: `from_node`, `to_node` | Directed road segments | **Arc ID = 0-based row index** |
| `.route` | 1: `arc_id` | Chronological reference arc sequence | Not a per-observation label |

## 3. Inventory to measure (A, T10)

> The scaffold investigation produced preliminary counts, but the evidence files
> were removed so A can produce and own them. Fill this table from
> `results/eda/inventory.csv`; do not quote a count before it exists.

| Fact | Value |
|---|---|
| Trips | 80 train + 10 student-test |
| Total size | TBD |
| Train GPS observations | TBD |
| Student-test GPS observations | TBD |
| Nodes per trip | TBD |
| Arcs per trip | TBD |
| Reference arcs per trip | TBD |
| File completeness | TBD |
| NaNs / non-finite values | TBD |
| Trips with large gaps | TBD |
| Largest observed gap | TBD |

## 4. Known quirks

1. **Per-trip networks, different cities.** Coordinate ranges show Darmstadt,
   Ulm, the Netherlands and others. Never use raw coordinates or node IDs as
   model features; they do not transfer.
2. **Contracted arcs.** Arcs are straight chords between endpoints. Some are
   kilometres long, so the straight line cuts away from the curved road. The
   candidate index samples along the arc, not only at endpoints.
3. **GPS jitter.** Consecutive 1 Hz points are noisy by several metres. Raw
   speed and bearing are unreliable; the pipeline smooths positions over a
   short centred window before computing heading and speed.
4. **Reference routes are not perfect truth.** The route is connected in every
   trip (`to_node` of arc *i* equals `from_node` of arc *i+1`), but some routes
   are partial, offset, looped or even ordered opposite to the track. Labels
   therefore use the **nearest route arc** per observation; monotonic progress
   is retained only as a diagnostic. Gating labels on monotonicity makes clean
   loop/reversed routes look broken (false low coverage).
   - A must measure coverage, candidate miss rate and the unreliable trips in
     T10/T12 and write them to `results/labels/` and `results/eda/`.
   - Do not quote a coverage or miss-rate number until those files exist.
5. **Zero-length arcs and disconnected components.** Guard every division by a
   segment length; do not assume the network is strongly connected. The HMM
   treats unreachable transitions as infeasible.
6. **Large gaps.** Gap-aware features and decoder resets are required; a 1 s
   transition model cannot be applied across an hour-long gap.

## 5. Reproduction commands

```bash
# full inventory + EDA figures
python scripts/01_eda.py --data-root "Student folder" --out-dir results/eda

# one-trip smoke test (fast)
python scripts/01_eda.py --data-root "Student folder" --max-trips 5

# topology only, no label alignment
python scripts/01_eda.py --data-root "Student folder" --no-labels
```

The EDA script writes `results/eda/inventory.csv`, `summary.json` and figures.
These are the source for §2 of the report and §0 of `viva.md`.

## 6. Assumptions to state in the report

- The supplied route is reference information, not independently verified
  physical truth.
- Long arcs are treated as straight segments between their endpoints.
- The local equirectangular projection is accurate at trip scale; distances are
  reported in metres with sub-0.5% error at these extents.
- Observations whose reference arc is missing from the candidate set are
  excluded from supervised fitting, and the miss rate is reported separately.
- Evaluator trips are never used for training, tuning, thresholds or feature
  selection.
