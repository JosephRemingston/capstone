# Default importance model: csv5k Ensemble + thresholds

Generated 2026-10-15T22:30:02+00:00 by `python -m training.ensemble.run`. The serving artifact is `artifacts/importance_ensemble/` (metadata sha256 `17286f0439fb`).

csv5k Ensemble + thresholds: fit-only XGBoost + frozen-BGE deep head (3-seed average), class-weighted. The system's default importance score is 25 % this model's class value plus 75 % the heuristic score (`main/ensemble.py`).

## Dataset

| Item | Value |
| --- | --- |
| Source | `data/experiments/csv5k.csv` (sha256 `29b47e54b159`) |
| Rows / distinct sentences | 5,000 / 443 |
| Words per sentence | mean 6.2, max 13 |
| Importance score | 0.20–1.00 (mean 0.708) |
| Classes | low < 0.4 ≤ mid < 0.7 ≤ high |
| Split | grouped by sentence (StratifiedGroupKFold, 5 folds), seed 42 |
| Train / validation / test rows | 3,200 / 792 / 1,008 |
| Train / validation / test sentences | 286 / 71 / 86 |
| Test class counts | low 120 / mid 272 / high 616 |
| Repeated sentences whose labels disagree | 86 of 410 |

## Test results

Macro F1 intervals are 95% bootstrap intervals over test rows and over whole test sentences (1,000 resamples, seed 42).

| Model | Accuracy | Balanced accuracy | Macro F1 | Row CI | Sentence CI | Weighted F1 | F1 low / mid / high |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- |
| Always the most common class (high) | 61.11% | 33.33% | 0.2529 | [0.245, 0.260] | [0.220, 0.283] | 0.4636 | 0.00 / 0.00 / 0.76 |
| Best possible (oracle) | 91.17% | 92.86% | 0.9230 | [0.906, 0.938] | [0.885, 0.957] | 0.9126 | 1.00 / 0.84 / 0.93 |
| **Ensemble + thresholds** | 89.09% | 87.14% | 0.8912 | [0.872, 0.909] | [0.809, 0.949] | 0.8836 | 1.00 / 0.76 / 0.92 |

The oracle predicts each test sentence's most common test label: a ceiling that reads the test labels, not a model.

Confusion matrix (rows gold, columns predicted):

| Gold \ predicted | low | mid | high |
| --- | ---: | ---: | ---: |
| low | 120 | 0 | 0 |
| mid | 0 | 171 | 101 |
| high | 0 | 9 | 607 |

Validation macro F1 (optimistic, used for tuning): 0.9141. Blended probabilities before the class weights: test NLL 0.2193, ECE 0.0346.

## Model

| Part | Value |
| --- | --- |
| XGBoost member | 525 features; max depth 3, min child weight 3, 304 rounds (fit split only); temperature 0.8962 |
| Deep head | frozen-bge-deep-head-v1 on BAAI/bge-small-en-v1.5:fastembed-0.8.1; seeds 42 (epoch 12, T 0.7544), 43 (epoch 7, T 0.9594), 44 (epoch 6, T 0.7122); seed-average temperature 0.9060 |
| Blend | 0.5 × XGBoost + 0.5 × deep head (validation: w 0.25 → 0.9104, w 0.5 → 0.9117, w 0.75 → 0.9100) |
| Class weights | 1 / 0.7 / 1 (validation macro F1 0.9117 → 0.9141) |
| Class values (serving score) | low 0.3102, mid 0.5792, high 0.8372 (mean csv5k score of the fit + validation rows in each class; the test split is excluded) |

## Serving checks

| Check | Rows | Class mismatches | Max probability difference |
| --- | ---: | ---: | ---: |
| ONNX + saved XGBoost, validation (training embeddings) | 792 | 0 | 8.41e-08 |
| ONNX + saved XGBoost, test (training embeddings) | 1,008 | 0 | 3.96e-08 |
| Live single-text BGE embeddings, test | 1,008 | 0 | 3.16e-05 |

## Caveats

- csv5k is AI-generated: template sentences with synthetic importance scores, not human ratings.
- The test split has 1,008 rows but only 86 distinct sentences; the sentence-bootstrap interval is the honest one.
- The 0.4 / 0.7 bin edges are a modelling choice.
- csv5k sentences are short (at most 13 words); conversation turns are much longer, so serving runs far outside the training distribution. The system's 25 % model weight is a fixed choice, not tuned; this is a reason to keep it small.
- Validation chose the temperatures, blend weight and class weights; only the test numbers are unbiased.
