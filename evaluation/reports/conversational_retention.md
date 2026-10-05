# Conversational retention proxy: XGBoost experiment

The optional model predicts whether an extracted conversational fact is invalid,
short-term, or long-term. These are human-annotated labels from
[Personal Facts (MSC)](https://huggingface.co/datasets/adugeen/personal-facts-msc).
The source has **no numeric importance labels**. The 0–1 score used by CogniMem
is an explicit policy mapping of predicted class probabilities:
`P(long_term) + 0.5 × P(short_term)`.

## Method

- Source: published Personal Facts (MSC) train/test Parquet files, downloaded
  with fixed SHA-256 checksums by the training script.
- Samples: 2,223 published train rows and 556 published test rows. One train
  row was excluded because it was marked valid but had no duration label.
- Inputs: the fact `text` only. Neither `duration`, `broken`, `main_category`,
  nor the preceding `context` is used as a model input. Features match the
  existing optional `MLFeatureExtractor` used at inference.
- Selection: four XGBoost configurations compared on a stratified 20% portion
  of the published training split, maximizing macro F1. The selected setting
  was refitted to all 2,222 usable training rows before a single test evaluation.
- Class weights compensate for the relatively rare invalid and short-term rows.
- Majority baseline: always predict long-term, the largest training class.

## Published test-split results

| Metric | XGBoost | Majority baseline |
| --- | ---: | ---: |
| Accuracy | 69.24% | 68.17% |
| Balanced accuracy | 65.60% | 33.33% |
| Macro F1 | 0.6114 | 0.2702 |
| Weighted F1 | 0.7071 | 0.5526 |

Class-level test results:

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Invalid | 85 | 0.4175 | 0.5059 | 0.4574 |
| Short-term | 92 | 0.4823 | 0.7391 | 0.5837 |
| Long-term | 379 | 0.8782 | 0.7230 | 0.7931 |

Confusion matrix (rows are gold; columns are predicted), ordered invalid,
short-term, long-term:

| Gold / predicted | Invalid | Short-term | Long-term |
| --- | ---: | ---: | ---: |
| Invalid | 43 | 17 | 25 |
| Short-term | 11 | 68 | 13 |
| Long-term | 49 | 56 | 274 |

The mean absolute error of the *constructed proxy* on the test split was
0.2768. This is **not** an importance-rating MAE because the source has no such
ratings. Accuracy improves only 1.08 percentage points over majority prediction;
the main gain is recognizing minority classes, reflected in macro F1. Invalid
precision of 0.4175 is too low to treat this as a deletion decision maker.

The source facts were annotated by one person; its split is not grouped by
speaker or conversation. It contains extracted facts rather than representative
raw greetings, task requests, and mixed messages. Lifecycle thresholds have not
been calibrated to these scores. The heuristic remains the default.

## Reproduce

```bash
.venv/bin/python -m training.train_conversational_retention
.venv/bin/python -m unittest tests.test_ml_importance -v
```

Machine-readable metrics are in [conversational_retention.json](conversational_retention.json).
The model and metadata are in `artifacts/importance/`.
