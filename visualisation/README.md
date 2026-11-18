# Visualisation

Owner: **A**. D only seeds the entry point.

## Purpose

Generate the report and demo maps from actual pipeline outputs:

1. GPS observations;
2. road-network geometry;
3. reference route (report/training maps only);
4. predicted route (classifier-only and HMM).

## Code

- `pathsnap/visualize.py` — Folium map builder (`trip_map`, `save_map`).
- `visualisation/make_maps.py` — CLI that loads a trip + artifact, predicts, and
  writes HTML plus a PNG screenshot for the report.

## Rules

- Do not fabricate a map. Every layer must come from the dataset or the
  predictor.
- The **live evaluation UI** must not show the reference route.
- Report maps may show it because they are generated offline from permitted
  data.
- Use distinguishable colours, a legend/layer control, and an appropriate zoom.
- Commit the interactive HTML under `results/maps/` and a screenshot under
  `results/figures/`.
