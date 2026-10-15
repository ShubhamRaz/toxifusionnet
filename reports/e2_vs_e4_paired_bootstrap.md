# E2 vs E4: Paired-Bootstrap Evaluation Report

**Project:** ToxiFusionNet  
**Date:** 2026-10-15  
**Purpose:** Compare combined-dataset text-only training against combined-dataset training with CLIP.

## 1. Experiments

| Experiment | Configuration | Test mean macro-F1 |
|---|---|---:|
| E2 | Combined MultiBully + HOT, text-only | 0.6132 |
| E4 | Combined MultiBully + HOT, text + frozen CLIP | 0.6044 |
| Difference (E4 - E2) | | -0.0088 |

Prediction files:

- `runs/exp2_multidataset_text/preds_test.csv`
- `runs/exp4_multidataset_clip/preds_test.csv`

The comparison script checked that prediction rows aligned by
`source_dataset` and `id`, that there were no duplicate keys, and that
the gold labels matched for each task.

## 2. Method

A paired bootstrap comparison was performed on the saved predictions.

- Bootstrap resamples: 2,000
- Random seed: 42
- Metric: macro-F1
- Comparison: E4 minus E2
- Confidence intervals: percentile 95% intervals
- Predictions: fixed, already-trained model checkpoints

Missing or invalid gold labels were excluded separately for each task.
Therefore, the sample count differs by task and dataset source.

These intervals quantify test-example resampling uncertainty for the
two fixed checkpoints. They do not measure variability from retraining
with different random seeds.

## 3. Overall test results

| Task | E2 F1 | E4 F1 | Difference |
|---|---:|---:|---:|
| Sarcasm | 0.6275 | 0.6200 | -0.0075 |
| Hate | 0.7252 | 0.7239 | -0.0013 |
| Target | 0.4664 | 0.4872 | +0.0208 |
| Severity | 0.6336 | 0.5866 | -0.0470 |

E4 has a higher target macro-F1 point estimate, but lower sarcasm,
hate, and severity point estimates.

## 4. Paired-bootstrap results

All differences are E4 minus E2.

| Dataset | Task | n | E2 F1 | E4 F1 | Difference | 95% CI |
|---|---|---:|---:|---:|---:|---|
| MultiBully | Sarcasm | 858 | 0.5703 | 0.5581 | -0.0123 | [-0.0562, +0.0302] |
| MultiBully | Hate | 858 | 0.6773 | 0.6571 | -0.0202 | [-0.0558, +0.0149] |
| MultiBully | Target | 216 | 0.3931 | 0.4194 | +0.0263 | [-0.0968, +0.1375] |
| MultiBully | Severity | 237 | 0.3851 | 0.3382 | -0.0469 | [-0.1087, +0.0216] |
| HOT | Sarcasm | 337 | 0.7614 | 0.7841 | +0.0227 | [-0.0236, +0.0705] |
| HOT | Hate | 455 | 0.7940 | 0.8235 | +0.0295 | [-0.0066, +0.0694] |
| HOT | Target | 145 | 0.4286 | 0.4387 | +0.0101 | [-0.0173, +0.0375] |
| HOT | Severity | 145 | 0.4321 | 0.4714 | +0.0393 | [-0.0214, +0.0994] |
| All | Sarcasm | 1195 | 0.6275 | 0.6200 | -0.0075 | [-0.0420, +0.0258] |
| All | Hate | 1313 | 0.7252 | 0.7239 | -0.0013 | [-0.0287, +0.0255] |
| All | Target | 361 | 0.4664 | 0.4872 | +0.0208 | [-0.0813, +0.1198] |
| All | Severity | 382 | 0.6336 | 0.5866 | -0.0470 | [-0.1173, +0.0292] |

## 5. Interpretation

Every reported 95% confidence interval includes zero. The current
comparison therefore does not provide clear evidence that E4 is
better or worse than E2 on any individual task and source subset.

Observed patterns:

- MultiBully: E4 has a lower point estimate for sarcasm, hate, and
  severity, but a higher point estimate for target classification.
- HOT: E4 has higher point estimates for all four tasks.
- Overall: E4 does not improve the combined test mean macro-F1.

These are descriptive comparisons of single trained checkpoints.
The confidence intervals do not capture random-seed or training-run
variation. The results should not be interpreted as proof that CLIP
causes the observed changes.

## 6. Recommended next research step

Design a controlled comparison using identical five-fold assignments
for E2 and E4. Preserve the current experiment outputs and use separate
output directories. Compare mean and standard deviation across folds,
alongside per-task and per-source results.

Before launching that comparison, verify dataset splits, configuration
equivalence, and that fold assignments prevent leakage.

No new training job was launched to generate this report.
