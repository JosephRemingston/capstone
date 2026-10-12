# Neural conversational retention model (frozen-bge-two-branch-v1)

Dataset: https://huggingface.co/datasets/adugeen/personal-facts-msc. Fit/validation/test: 1777/445/556. Artifact: `artifacts/importance_neural_shallow`.

## Seeds (validation)

| Seed | Best epoch | Macro F1 | NLL |
| ---: | ---: | ---: | ---: |
| 42 | 4 | 0.7175 | 0.5644 |
| 43 | 2 | 0.7057 | 0.5279 |
| 44 | 11 | 0.7169 | 0.7912 |

Selected seed 42; temperature 1.3946 reduced validation NLL from 0.5644 to 0.5332.

## Published test split

| Metric | Neural | XGBoost |
| --- | ---: | ---: |
| accuracy | 0.7806 | 0.6924 |
| balanced_accuracy | 0.6586 | 0.6560 |
| macro_f1 | 0.6504 | 0.6114 |
| weighted_f1 | 0.7703 | 0.7071 |
| proxy_mae | 0.1957 | 0.2768 |
| nll | 0.5345 | — |
| ece_10_bins | 0.0428 | — |
| warm latency p50 / p95 ms | 3.96 / 5.79 | — |

| Gold class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| invalid | 85 | 0.6585 | 0.3176 | 0.4286 |
| short_term | 92 | 0.5547 | 0.7717 | 0.6455 |
| long_term | 379 | 0.8682 | 0.8865 | 0.8773 |

| Gold \ predicted | invalid | short_term | long_term |
| --- | ---: | ---: | ---: |
| invalid | 27 | 26 | 32 |
| short_term | 2 | 71 | 19 |
| long_term | 12 | 31 | 336 |

## Depth comparison (train split only)

| Architecture | Parameters | Mean validation macro F1 |
| --- | ---: | ---: |
| frozen-bge-two-branch-v1 | 60547 | 0.7134 |
| frozen-bge-deep-head-v1 | 131491 | 0.7194 |

Caveat: Exploratory comparison: published test split was previously inspected for XGBoost; no human numeric importance labels.
