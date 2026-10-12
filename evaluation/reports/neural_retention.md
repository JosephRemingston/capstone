# Neural conversational retention model (frozen-bge-deep-head-v1)

Dataset: https://huggingface.co/datasets/adugeen/personal-facts-msc. Fit/validation/test: 1777/445/556. Artifact: `artifacts/importance_neural`.

## Seeds (validation)

| Seed | Best epoch | Macro F1 | NLL |
| ---: | ---: | ---: | ---: |
| 42 | 12 | 0.7208 | 0.8370 |
| 43 | 3 | 0.7083 | 0.5503 |
| 44 | 8 | 0.7291 | 0.7913 |

Selected seed 44; temperature 2.4057 reduced validation NLL from 0.7913 to 0.5385.

## Published test split

| Metric | Neural | XGBoost |
| --- | ---: | ---: |
| accuracy | 0.7806 | 0.6924 |
| balanced_accuracy | 0.6778 | 0.6560 |
| macro_f1 | 0.6732 | 0.6114 |
| weighted_f1 | 0.7768 | 0.7071 |
| proxy_mae | 0.2160 | 0.2768 |
| nll | 0.5547 | — |
| ece_10_bins | 0.0163 | — |
| warm latency p50 / p95 ms | 4.04 / 5.86 | — |

| Gold class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| invalid | 85 | 0.5902 | 0.4235 | 0.4932 |
| short_term | 92 | 0.5913 | 0.7391 | 0.6570 |
| long_term | 379 | 0.8684 | 0.8707 | 0.8696 |

| Gold \ predicted | invalid | short_term | long_term |
| --- | ---: | ---: | ---: |
| invalid | 36 | 18 | 31 |
| short_term | 5 | 68 | 19 |
| long_term | 20 | 29 | 330 |

## Depth comparison (train split only)

| Architecture | Parameters | Mean validation macro F1 |
| --- | ---: | ---: |
| frozen-bge-two-branch-v1 | 60547 | 0.7134 |
| frozen-bge-deep-head-v1 | 131491 | 0.7194 |

Caveat: Exploratory comparison: published test split was previously inspected for XGBoost; no human numeric importance labels.
