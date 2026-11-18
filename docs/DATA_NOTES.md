# Dataset notes — verified facts and known limitations

> Owner: **A**. Every number below comes from a file in `results/eda/` or
> `results/labels/` (generated 2026-11-18, T10–T12). Re-run the scripts in §5 and
> update this file if they change. The same numbers are in the `viva.md` ledger.

## 1. Layout

```
data/
├── train/<trip_id>/<trip_id>.{track,nodes,arcs,route}           # 80 trips
├── student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}    # 10 trips
└── evaluator_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}  # 10 reserved trips (not in the student package)
```

The raw files are **not committed** (94.1 MB of course material). The API reads
them from `PATHSNAP_DATA_ROOT`. `cache/` and `data/` are git-ignored.

## 2. File semantics

| File | Columns | Semantics | Important detail |
|---|---|---|---|
| `.track` | 3: `lon`, `lat`, `t_seconds` | Noisy GPS trace, nominally 1 Hz | 96.7% of steps are exactly 1 s; gaps exist |
| `.nodes` | 2: `lon`, `lat` | Road-network nodes | **Node ID = 0-based row index** |
| `.arcs` | 2: `from_node`, `to_node` | Directed road segments | **Arc ID = 0-based row index** |
| `.route` | 1: `arc_id` | Chronological reference arc sequence | Not a per-observation label |

A GPS row does not correspond to a node, arc or route row; the link between
them is geometric (projection), which is what the pipeline learns.

## 3. Inventory (measured)

Sources: `results/eda/inventory.csv`, `summary.json`, `quality_checks.csv`, `network_stats.csv`.

| Fact | Value |
|---|---|
| Trips | 80 train + 10 student-test; 90/90 have all four files |
| Total size | 94,106,043 bytes (94.1 MB) |
| Train GPS observations | 208,417 |
| Student-test GPS observations | 19,160 |
| GPS points per trip | 262 – 19,313 (median 1,496.5) |
| Trip length (GPS path) | 5.5 – 168.9 km (median 24.0 km); median duration 28.5 min |
| Nodes per trip | 299 – 155,003 (median 9,305) |
| Arcs per trip | 649 – 307,392 (median 18,761.5) |
| Reference arcs per trip | 59 – 2,266 (median 272.5) |
| File completeness / column counts | all files present; every file has the expected column count; arc and route IDs are integers |
| NaNs / non-finite / out-of-range GPS rows | **0** (also 0 rows at (0, 0)) |
| Timestamp ordering | sorted in every trip; **0** repeated or decreasing timestamps |
| Sampling | 96.7% of steps = 1 s; median 1 s; p99 3 s |
| Trips with a gap > 5 s / > 30 s | 47 / 30 |
| Largest observed gap | **153,064 s (≈ 42.5 h)** in train `00000090`; next 52,810 s in train `00000078` |
| Implausible raw speed (> 55 m/s) | 966 steps in 28 trips (GPS jumps; median step speed 11.3 m/s, p99 45.3 m/s) |
| Arc length | median 25.6 m, p95 141.5 m, max 9,308 m; 0.25% of arcs > 500 m |
| Network anomalies | 24 self-loop arcs, 1,903 duplicate arcs, 106 zero-length arcs (all trips pooled) |
| Connectivity (weak components) | median 19 per trip, max 421; largest component < 90% of nodes in 3 trips (`00000000`, `00000072`, `00000095`, min 69%) |
| Two-way roads | median 95% of arcs have a reciprocal arc |
| GPS → nearest reference-route arc | median 5.6 m; 88.8% of all points within 30 m |

## 4. Known quirks

1. **Per-trip networks, worldwide.** Trip centroids span the Americas, Europe,
   Asia and Australia (`results/eda/gps_locations.png`), not one city. Never use
   raw coordinates or node IDs as model features; they do not transfer.
2. **Contracted arcs.** Arcs are straight chords between endpoints. Some are
   kilometres long (max 9.3 km), so the straight line cuts away from the curved
   road. The candidate index samples along the arc, not only at endpoints.
3. **GPS jitter.** Consecutive 1 Hz points are noisy by several metres. Raw
   speed and bearing are unreliable; the pipeline smooths positions over a
   short centred window before computing heading and speed.
4. **Reference routes are reference information, not perfect truth.**
   - **Routes are not always connected** (correction of an earlier note):
     11/90 routes have 1–3 places where consecutive arcs do not share a node,
     with endpoint gaps of 15–597 m (`route_breaks` in `results/eda/inventory.csv`).
     The worst is train `00000019` (3 breaks, max 597 m).
   - **Loop / out-and-back routes:** 8 trips have a time-vs-route-progress rank
     correlation < 0.8 (train `00000011`, `00000033`, `00000046`, `00000048`,
     `00000061`, `00000069`, `00000081`; student-test `00000056`). In `00000011`
     the vehicle drives a road out and back (correlation 0.08). No route is
     stored fully reversed (no negative correlation).
   - Labels therefore use the **nearest route arc** per observation (D-009);
     monotonic progress is only a diagnostic. The earlier `backward_fraction`
     diagnostic is too noisy to flag loops (median 0.10, because stop-and-go
     jitter counts as "backward"); use `route_order_corr` instead.
5. **Two genuinely unreliable references (train only).** Coverage < 70%:
   - train `00000054` (11.5%): after ~50 min the GPS runs a steady 100–200 m from
     the reference for 4.5 h — a systematic offset (parallel road or shifted
     GPS). The trip is also the longest (168.9 km).
   - train `00000033` (29.1%): the route itself matches where the vehicle moves;
     the vehicle is parked ~120 m off the network for ~55 min and ~50 m off for
     ~15 min. Those points are correctly excluded.
   - Both are kept in the dataset; their low-confidence points carry no label.
     Student-test has **no** unreliable trip (min coverage 98.0%) — an earlier
     note naming student-test `00000077` came from the superseded monotonic
     alignment and is withdrawn.
   - Evidence: `results/labels/coverage.csv`, `visual_check_00000033.png`, `visual_check_00000054.png`.
6. **Zero-length arcs, self-loops, duplicate arcs and disconnected components.**
   Guard every division by a segment length; do not assume the network is
   strongly connected. The HMM treats unreachable transitions as infeasible.
7. **Large gaps.** 30 trips have a gap > 30 s and two exceed 14 h. Gap-aware
   features and decoder resets are required; a 1 s transition model cannot be
   applied across such gaps.
8. **Candidate generation is not the bottleneck.** With the production setting
   (60 m + fallbacks, top-K 15) the truth arc is missing for 0.092% of labelled
   train observations (0.032% student-test); the worst trip is train `00000072`
   (8.5%), followed by `00000061` (3.5%). See §5 of `viva.md` and
   `results/eda/candidate_sweep.csv`.

## 5. Reproduction commands

```bash
# T10: inventory, quality checks, flags and figures  -> results/eda/
python3 scripts/01_eda.py --data-root ./data

# T11: candidate radius / top-K sweep                 -> results/eda/candidate_sweep*.{csv,json,png}
python3 scripts/01b_candidate_sweep.py --data-root ./data

# T12: label coverage table + visual checks           -> results/labels/
python3 scripts/01c_label_coverage.py --data-root ./data

# T21: first Folium maps (+ headless-Chrome screenshots) -> results/maps/, results/figures/
python3 visualisation/make_maps.py --data-root ./data
```

Anomaly flags and their rules are in `results/eda/flags.csv` and at the top of
`scripts/01_eda.py`. Flags are diagnostics; no trip is dropped.

## 6. Assumptions to state in the report

- The supplied route is reference information, not independently verified
  physical truth (two train trips are demonstrably unreliable; 11 routes have breaks).
- Long arcs are treated as straight segments between their endpoints.
- The local equirectangular projection is accurate at trip scale; distances are
  reported in metres with sub-0.5% error at these extents.
- Observations further than 30 m from the reference route carry no label;
  observations whose reference arc is missing from the candidate set are
  excluded from supervised fitting, and the miss rate is reported separately.
- Evaluator trips are never used for training, tuning, thresholds or feature
  selection, and `make_maps.py` refuses to draw reference maps for them.
