# HMM decoder: tuning, comparison and failures

## Scope

The classifier predicts each observation independently. The HMM adds sequence
context: emissions are classifier probabilities and transitions compare directed
network travel distance with GPS displacement. Viterbi then selects one coherent
candidate sequence in log space.

## T25 — decoder tuning

`scripts/05_tune_hmm.py` searches `lam`, `sigma_m`, `beta_m`,
`max_speed_mps`, `gap_reset_s`, and `uturn_penalty`. It accepts only `train` or
`student_test`, writes every configuration and aggregate objective to
`results/hmm/decoder_tuning.csv`, and writes the selected configuration to
`artifacts/hmm_config.json`. Evaluator partitions are rejected by the argument
parser.

The final tuning table is pending in this checkout because raw `data/train/`
and `data/student_test/` are not committed to the public repository. No tuned
number is claimed here until the command is run on the permitted data.

## T26 — classifier-only versus HMM

Run:

```bash
python scripts/05_eval_hmm.py --partition student_test
```

The runner uses the same candidate sets, classifier probabilities, labels and
scoring implementation for both modes. It records observation Top-1,
candidate macro-F1/log loss, strict and relaxed continuity, sequence LCS-F1,
arc-set F1, and decode time in `results/hmm/classifier_vs_hmm.csv`.

The existing preliminary integration evidence is in
`results/integration/per_trip_comparison.csv`: across the student-test set,
classifier-only measured Top-1 0.897, strict continuity 0.756 and LCS-F1
0.763; default HMM measured Top-1 0.851, strict continuity 0.839 and LCS-F1
0.729. This is a baseline, not the final tuned result.

## T27 — failure analysis

Run:

```bash
python scripts/06_hmm_failure_analysis.py
```

The script selects three cases mechanically from the comparison table: the
largest Top-1 regression, the largest continuity recovery, and the largest
sequence regression. With raw trips mounted it writes three Folium maps under
`results/hmm/failure_maps/` and the narrative table to
`results/hmm/failures.md`. This makes the failure selection reproducible and
avoids hand-picking a flattering example.

The cases should be explained as candidate misses, GPS gaps, junction or
parallel-road ambiguity, and over-smoothing. The HMM cannot repair a missing
truth candidate; it can also choose a connected but wrong road when the
emission is ambiguous.
