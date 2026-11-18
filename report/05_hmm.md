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

The student-test run evaluated 48 configurations. The selected configuration
was `lam=0`, `sigma_m=8`, `beta_m=10`, `max_speed_mps=55`, `gap_reset_s=60`,
and `uturn_penalty=0`; it is recorded in `artifacts/hmm_config.json`. The
source grid and objective values are in `results/hmm/decoder_tuning.csv`.

## T26 — classifier-only versus HMM

Run:

```bash
python scripts/05_eval_hmm.py --partition student_test
```

The runner uses the same candidate sets, classifier probabilities, labels and
scoring implementation for both modes. It records observation Top-1,
candidate macro-F1/log loss, strict and relaxed continuity, sequence LCS-F1,
arc-set F1, and decode time in `results/hmm/classifier_vs_hmm.csv`.

The tuned comparison across ten student-test trips is in
`results/hmm/classifier_vs_hmm.csv`:

| Mode | Top-1 | Candidate macro-F1 | Strict continuity | LCS-F1 | Decode ms |
|---|---:|---:|---:|---:|---:|
| Classifier-only | 0.8995 | 0.9366 | 0.7547 | 0.7723 | 1.6 |
| Tuned HMM (T25 v2) | 0.9297 | 0.9366 | 0.9057 | 0.9335 | 475.7 |

With the T25 v2 config (τ (same-arc backward tolerance) 40 m, β 5, U-turn penalty 40, λ 0, σ 8, gap reset 60 s, max speed 55 m/s) the HMM improves Top-1 by +0.0302, strict
continuity by +0.1510 and LCS-F1 by +0.1611. The first grid
(`results/hmm/decoder_tuning.csv`) traded Top-1 for continuity because same-arc backward GPS
jitter was infeasible and the decoder flipped to opposite-direction arcs; the v2 grid adds a
backward tolerance, selects on 24 train trips with out-of-fold classifier probabilities, and
reports on the held-out student-test trips (`scripts/05d_tune_hmm_v2.py`,
`results/hmm/decoder_tuning_v2*.csv`).

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

The generated cases are `00000077` (largest Top-1 regression), `00000091`
(largest continuity recovery), and `00000040` (largest sequence regression).
Their maps are under `results/hmm/failure_maps/`, and the measured table is in
`results/hmm/failures.md`. These cases illustrate connected-but-wrong
over-smoothing and the opposite benefit: removing an independent-classifier
teleport. The HMM cannot repair a missing truth candidate.
