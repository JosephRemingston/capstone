# XGBoost conversational retention model

Dataset: https://huggingface.co/datasets/adugeen/personal-facts-msc. Target: human-annotated fact validity and expected duration; numeric score is a policy proxy. Model selection: 20% stratified validation from published train split; maximize macro F1.

Rows used: 2222 train, 556 test (1 train row omitted).

## Validation trials

| max_depth | min_child_weight | Best iteration | Macro F1 | Balanced accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 3 | 395 | 0.5934 | 0.6347 |
| 3 | 8 | 398 | 0.6022 | 0.6494 |
| 5 | 3 | 360 | 0.6002 | 0.6359 |
| 5 | 8 | 395 | 0.6211 | 0.6632 |

Selected: max_depth 5, min_child_weight 8, 396 trees on refit.

## Published test split

| Metric | XGBoost | Always long-term |
| --- | ---: | ---: |
| accuracy | 0.6924 | 0.6817 |
| balanced_accuracy | 0.6560 | 0.3333 |
| macro_f1 | 0.6114 | 0.2702 |
| weighted_f1 | 0.7071 | 0.5526 |
| proxy_mae | 0.2768 | — |

| Gold class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| invalid | 85 | 0.4175 | 0.5059 | 0.4574 |
| short_term | 92 | 0.4823 | 0.7391 | 0.5837 |
| long_term | 379 | 0.8782 | 0.7230 | 0.7931 |

| Gold \ predicted | invalid | short_term | long_term |
| --- | ---: | ---: | ---: |
| invalid | 43 | 17 | 25 |
| short_term | 11 | 68 | 13 |
| long_term | 49 | 56 | 274 |

Reproduce: `python -m training.train_conversational_retention`.
