# Retention models on three datasets (isolated experiment)

Generated 2026-10-15T21:21:40+00:00. Shipped artifacts and reports are untouched; every number below is copied from `artifacts/experiments/retention_datasets/<dataset>/combined.json`.

Run: git `6f0d074`, argv `/Users/josephremingstonl/Downloads/code/capstone/training/experiments/run_all.py --skip-build --skip-xgboost --skip-neural --skip-replay`, failed steps: none.

## msc

### Diagnostics

| Diagnostic | Value |
| --- | --- |
| rows / unique sentences | 2778 / 2778 |
| singleton / multi-row groups | 2778 / 0 |
| largest group | 1 rows |
| multi-row groups disagreeing on class | 0 (—) |
| mean within-group score std | — |
| bin edges / tertile edges | — / — |
| text normalisation | none (parquet text as shipped) |
| split protocol / seed | published / 42 |
| pin rule | None → nothing pinned |
| fit / val / test rows | 1777 / 445 / 556 |
| fit / val / test sentences | 1777 / 445 / 556 |
| class counts (all) | invalid 424, short_term 470, long_term 1884 |
| class counts (fit) | invalid 271, short_term 302, long_term 1204 |
| class counts (val) | invalid 68, short_term 76, long_term 301 |
| class counts (test) | invalid 85, short_term 92, long_term 379 |
| majority class (fit+val) | long_term |

The oracle uses the test labels, is the best any deterministic text function can score, is not a model, and a model exceeding it indicates a bug.

### Variants (tuned on validation only; val macro-F1 is optimistic)

| Variant | Val macro-F1 | Test acc | Bal acc | Macro-F1 [row CI] [sentence CI] | NLL | ECE | Tuned |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| majority | 0.2690 | 0.6817 | 0.3333 | 0.2702 [0.261, 0.279] — | — | — | long_term |
| oracle | — | — | — | — | — | — | — |
| xgb_asis | 0.6211 | 0.6924 | 0.6560 | 0.6114 [0.562, 0.656] — | 0.7255 | 0.0246 | max_depth=5, min_child_weight=8, best_iteration=395 |
| xgb_fitonly | 0.6211 | 0.6871 | 0.6418 | 0.6012 [0.553, 0.647] — | 0.7313 | 0.0247 | max_depth=5, min_child_weight=8, best_iteration=395 |
| deep_asis | 0.7291 | 0.7806 | 0.6778 | 0.6732 [0.621, 0.720] — | 0.5547 | 0.0163 | seed=44, best_epoch=8, T=2.406 |
| shallow_asis | 0.7175 | 0.7806 | 0.6586 | 0.6504 [0.600, 0.700] — | 0.5345 | 0.0428 | seed=42, best_epoch=4, T=1.395 |
| deep_seedavg | 0.7142 | 0.8076 | 0.6833 | 0.6992 [0.648, 0.745] — | 0.4979 | 0.0317 | ensemble_temperature=0.887, seeds=42/43/44 |
| shallow_seedavg (exploratory) | 0.7136 | 0.7878 | 0.6710 | 0.6631 [0.611, 0.712] — | 0.5152 | 0.0538 | ensemble_temperature=0.926, seeds=42/43/44 |
| ensemble | 0.7278 | 0.8040 | 0.7139 | 0.7132 [0.662, 0.758] — | 0.5399 | 0.1171 | w=0.5, xgb_temperature=1.046 |
| thr_xgb_fitonly (exploratory) | 0.6577 | 0.7338 | 0.6113 | 0.6155 [0.563, 0.664] — | — | — | cw=0.5/0.5/1 |
| thr_deep_asis (exploratory) | 0.7291 | 0.7806 | 0.6778 | 0.6732 [0.621, 0.720] — | — | — | cw=1/1/1 |
| thr_shallow_asis (exploratory) | 0.7175 | 0.7806 | 0.6586 | 0.6504 [0.600, 0.700] — | — | — | cw=1/1/1 |
| thr_ensemble | 0.7278 | 0.8040 | 0.7139 | 0.7132 [0.662, 0.758] — | — | — | cw=1/1/1 |
| xgb_tertile | — | — | — | — | — | — | — |

Ensemble weight search on validation:

| w (XGBoost share) | Val macro-F1 | Val NLL |
| ---: | ---: | ---: |
| 0.25 | 0.7204 | 0.5053 |
| 0.5 | 0.7278 | 0.5460 |
| 0.75 | 0.6778 | 0.6140 |

### Paired comparisons (candidate − baseline on the shared bootstrap matrix)

| Baseline → candidate | Δ macro-F1 | row CI | sentence CI | one-sided p | Δ accuracy | acc CI | acc p | Primary |
| --- | ---: | --- | --- | ---: | ---: | --- | ---: | --- |
| deep_asis → deep_seedavg | 0.0260 | [-0.011, 0.058] | — | 0.0699 | 0.0270 | [0.005, 0.052] | 0.0280 | yes (α=0.0125) |
| deep_seedavg → ensemble | 0.0140 | [-0.024, 0.053] | — | 0.2408 | -0.0036 | [-0.027, 0.022] | 0.8981 | yes (α=0.0125) |
| xgb_fitonly → ensemble | 0.1120 | [0.063, 0.159] | — | 0.0010 | 0.1169 | [0.079, 0.153] | 0.0010 | yes (α=0.0125) |
| ensemble → thr_ensemble | 0.0000 | [0.000, 0.000] | — | 1.0000 | 0.0000 | [0.000, 0.000] | 1.0000 | yes (α=0.0125) |
| shallow_asis → shallow_seedavg | 0.0127 | [-0.013, 0.038] | — | 0.1548 | 0.0072 | [-0.005, 0.022] | 0.4416 | exploratory |
| xgb_fitonly → xgb_asis | 0.0102 | [-0.015, 0.038] | — | 0.2198 | 0.0054 | [-0.016, 0.029] | 0.7512 | exploratory |
| xgb_fitonly → thr_xgb_fitonly | 0.0143 | [-0.015, 0.044] | — | 0.1698 | 0.0468 | [0.020, 0.076] | 0.0040 | exploratory |
| deep_asis → thr_deep_asis | 0.0000 | [0.000, 0.000] | — | 1.0000 | 0.0000 | [0.000, 0.000] | 1.0000 | exploratory |
| shallow_asis → thr_shallow_asis | 0.0000 | [0.000, 0.000] | — | 1.0000 | 0.0000 | [0.000, 0.000] | 1.0000 | exploratory |
| deep_asis → shallow_asis | -0.0228 | [-0.065, 0.016] | — | 0.8531 | 0.0000 | [-0.025, 0.023] | 1.0000 | exploratory |

### Neural seeds (validation)

| Head | Seed | Best epoch | Val macro-F1 | Val NLL | T |
| --- | ---: | ---: | ---: | ---: | ---: |
| deep | 42 | 12 | 0.7208 | 0.8370 | 2.7073 |
| deep | 43 | 3 | 0.7083 | 0.5503 | 1.1199 |
| deep | 44 | 8 | 0.7291 | 0.7913 | 2.4057 |
| shallow | 42 | 4 | 0.7175 | 0.5644 | 1.3946 |
| shallow | 43 | 2 | 0.7057 | 0.5279 | 1.0437 |
| shallow | 44 | 11 | 0.7169 | 0.7912 | 2.5863 |

### XGBoost validation trials

| Labels | max_depth | min_child_weight | Best iteration | Val macro-F1 | Val balanced acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| main | 3 | 3 | 395 | 0.5934 | 0.6347 |
| main | 3 | 8 | 398 | 0.6022 | 0.6494 |
| main | 5 | 3 | 360 | 0.6002 | 0.6359 |
| main | 5 | 8 | 395 | 0.6211 | 0.6632 |
| tertile | — | — | — | — | — |

### Replay of the shipped artifacts through this harness (must match exactly)

| Artifact | Macro-F1 | Confusion matrix | Exact match |
| --- | ---: | --- | --- |
| importance | 0.6114 | `[[43, 17, 25], [11, 68, 13], [49, 56, 274]]` | yes |
| importance_neural | 0.6732 | `[[36, 18, 31], [5, 68, 19], [20, 29, 330]]` | yes |
| importance_neural_shallow | 0.6504 | `[[27, 26, 32], [2, 71, 19], [12, 31, 336]]` | yes |

### Reproduction of the shipped MSC numbers (retrained here)

| Variant | Shipped macro-F1 | This run | Δ |
| --- | ---: | ---: | ---: |
| xgb_asis | 0.6114 | 0.6114 | 0.0000 |
| deep_asis | 0.6732 | 0.6732 | 0.0000 |
| shallow_asis | 0.6504 | 0.6504 | -0.0000 |

Self-checks: w1_reproduces_xgb_fitonly=True, w0_reproduces_deep_seedavg=True, all_ones_thresholds_reproduce_argmax=True, y_test_identical_across_sources=True, y_test_sources=['data/splits.npz', 'xgboost/y_test.npy', 'neural/y_test.npy'], models_varying_within_sentence=[], oracle_accuracy_dominates=—, oracle_macro_f1_dominates=—, neural_consistency={'deep': {'selected_seed_matches_result': True, 'temperature_max_abs_diff_vs_info': 0.0}, 'shallow': {'selected_seed_matches_result': True, 'temperature_max_abs_diff_vs_info': 0.0}}.
Missing pieces: none.

## csv5k

### Diagnostics

| Diagnostic | Value |
| --- | --- |
| rows / unique sentences | 5000 / 443 |
| singleton / multi-row groups | 33 / 410 |
| largest group | 43 rows |
| multi-row groups disagreeing on class | 86 (0.210) |
| mean within-group score std | 0.0356 |
| bin edges / tertile edges | [0.4, 0.7] / [0.67, 0.83] |
| text normalisation | strip ' (example N)' suffix and collapse whitespace; used for grouping and model input |
| split protocol / seed | grouped / 42 |
| pin rule | 0.05 → nothing pinned |
| fit / val / test rows | 3200 / 792 / 1008 |
| fit / val / test sentences | 286 / 71 / 86 |
| class counts (all) | low 559, mid 1359, high 3082 |
| class counts (fit) | low 358, mid 869, high 1973 |
| class counts (val) | low 81, mid 218, high 493 |
| class counts (test) | low 120, mid 272, high 616 |
| majority class (fit+val) | high |

The oracle uses the test labels, is the best any deterministic text function can score, is not a model, and a model exceeding it indicates a bug.

### Variants (tuned on validation only; val macro-F1 is optimistic)

| Variant | Val macro-F1 | Test acc | Bal acc | Macro-F1 [row CI] [sentence CI] | NLL | ECE | Tuned |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| majority | 0.2558 | 0.6111 | 0.3333 | 0.2529 [0.245, 0.260] [0.220, 0.283] | — | — | high |
| oracle | — | 0.9117 | 0.9286 | 0.9230 [0.906, 0.938] [0.885, 0.957] | — | — | — |
| xgb_asis | 0.9090 | 0.8671 | 0.8905 | 0.8846 [0.867, 0.902] [0.824, 0.930] | 0.2239 | 0.0238 | max_depth=3, min_child_weight=3, best_iteration=303 |
| xgb_fitonly | 0.9090 | 0.8522 | 0.8612 | 0.8654 [0.846, 0.884] [0.792, 0.917] | 0.2358 | 0.0491 | max_depth=3, min_child_weight=3, best_iteration=303 |
| deep_asis | 0.9104 | 0.8800 | 0.8716 | 0.8850 [0.867, 0.903] [0.810, 0.941] | 0.2608 | 0.0421 | seed=44, best_epoch=6, T=0.712 |
| shallow_asis | 0.9001 | 0.8482 | 0.8419 | 0.8538 [0.834, 0.873] [0.770, 0.926] | 0.2700 | 0.0584 | seed=44, best_epoch=40, T=1.018 |
| deep_seedavg | 0.8987 | 0.8760 | 0.8468 | 0.8699 [0.849, 0.889] [0.778, 0.944] | 0.2131 | 0.0240 | ensemble_temperature=0.906, seeds=42/43/44 |
| shallow_seedavg (exploratory) | 0.9001 | 0.8720 | 0.8467 | 0.8678 [0.848, 0.886] [0.778, 0.941] | 0.2682 | 0.0431 | ensemble_temperature=0.916, seeds=42/43/44 |
| ensemble | 0.9117 | 0.8839 | 0.8737 | 0.8882 [0.871, 0.906] [0.811, 0.945] | 0.2193 | 0.0346 | w=0.5, xgb_temperature=0.896 |
| thr_xgb_fitonly (exploratory) | 0.9141 | 0.8720 | 0.8481 | 0.8687 [0.848, 0.888] [0.771, 0.934] | — | — | cw=1/0.5/1 |
| thr_deep_asis (exploratory) | 0.9104 | 0.8800 | 0.8716 | 0.8850 [0.867, 0.903] [0.810, 0.941] | — | — | cw=1/1/1 |
| thr_shallow_asis (exploratory) | 0.9001 | 0.8482 | 0.8419 | 0.8538 [0.834, 0.873] [0.770, 0.926] | — | — | cw=1/1/1 |
| thr_ensemble | 0.9141 | 0.8909 | 0.8714 | 0.8912 [0.872, 0.909] [0.809, 0.949] | — | — | cw=1/0.7/1 |
| xgb_tertile (exploratory) (tertile labels) | 0.8769 | 0.8155 | 0.8158 | 0.8156 [0.792, 0.839] [0.747, 0.867] | 0.3889 | 0.0435 | max_depth=5, min_child_weight=3, best_iteration=323, edges=0.67/0.83 |

Ensemble weight search on validation:

| w (XGBoost share) | Val macro-F1 | Val NLL |
| ---: | ---: | ---: |
| 0.25 | 0.9104 | 0.1847 |
| 0.5 | 0.9117 | 0.1876 |
| 0.75 | 0.9100 | 0.1938 |

### Paired comparisons (candidate − baseline on the shared bootstrap matrix)

| Baseline → candidate | Δ macro-F1 | row CI | sentence CI | one-sided p | Δ accuracy | acc CI | acc p | Primary |
| --- | ---: | --- | --- | ---: | ---: | --- | ---: | --- |
| deep_asis → deep_seedavg | -0.0151 | [-0.031, -0.001] | [-0.070, 0.015] | 0.9830 | -0.0040 | [-0.018, 0.012] | 0.7253 | yes (α=0.0125) |
| deep_seedavg → ensemble | 0.0184 | [0.005, 0.034] | [-0.008, 0.068] | 0.0080 | 0.0079 | [-0.006, 0.021] | 0.3377 | yes (α=0.0125) |
| xgb_fitonly → ensemble | 0.0228 | [0.005, 0.040] | [-0.023, 0.076] | 0.0070 | 0.0317 | [0.014, 0.050] | 0.0020 | yes (α=0.0125) |
| ensemble → thr_ensemble | 0.0030 | [-0.007, 0.012] | [-0.004, 0.009] | 0.2927 | 0.0069 | [-0.002, 0.017] | 0.2388 | yes (α=0.0125) |
| shallow_asis → shallow_seedavg | 0.0140 | [0.002, 0.026] | [-0.001, 0.038] | 0.0180 | 0.0238 | [0.011, 0.037] | 0.0010 | exploratory |
| xgb_fitonly → xgb_asis | 0.0192 | [0.006, 0.032] | [-0.008, 0.065] | 0.0030 | 0.0149 | [0.002, 0.029] | 0.0390 | exploratory |
| xgb_fitonly → thr_xgb_fitonly | 0.0033 | [-0.014, 0.020] | [-0.038, 0.031] | 0.3956 | 0.0198 | [0.001, 0.038] | 0.0470 | exploratory |
| deep_asis → thr_deep_asis | 0.0000 | [0.000, 0.000] | [0.000, 0.000] | 1.0000 | 0.0000 | [0.000, 0.000] | 1.0000 | exploratory |
| shallow_asis → thr_shallow_asis | 0.0000 | [0.000, 0.000] | [0.000, 0.000] | 1.0000 | 0.0000 | [0.000, 0.000] | 1.0000 | exploratory |
| deep_asis → shallow_asis | -0.0312 | [-0.047, -0.015] | [-0.080, 0.006] | 1.0000 | -0.0317 | [-0.049, -0.015] | 0.0020 | exploratory |

### Neural seeds (validation)

| Head | Seed | Best epoch | Val macro-F1 | Val NLL | T |
| --- | ---: | ---: | ---: | ---: | ---: |
| deep | 42 | 12 | 0.9001 | 0.1878 | 0.7544 |
| deep | 43 | 7 | 0.8987 | 0.1836 | 0.9594 |
| deep | 44 | 6 | 0.9104 | 0.1947 | 0.7122 |
| shallow | 42 | 12 | 0.9001 | 0.2100 | 1.0150 |
| shallow | 43 | 4 | 0.9001 | 0.2567 | 1.5502 |
| shallow | 44 | 40 | 0.9001 | 0.2030 | 1.0184 |

### XGBoost validation trials

| Labels | max_depth | min_child_weight | Best iteration | Val macro-F1 | Val balanced acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| main | 3 | 3 | 303 | 0.9090 | 0.9114 |
| main | 3 | 8 | 395 | 0.9042 | 0.9089 |
| main | 5 | 3 | 273 | 0.9070 | 0.9101 |
| main | 5 | 8 | 271 | 0.9047 | 0.9079 |
| tertile | 3 | 3 | 399 | 0.8590 | 0.8568 |
| tertile | 3 | 8 | 399 | 0.8590 | 0.8568 |
| tertile | 5 | 3 | 323 | 0.8769 | 0.8703 |
| tertile | 5 | 8 | 339 | 0.8721 | 0.8649 |

### Sentence-level secondary table (test deduplicated by sentence, label = test majority)

| Variant | Sentences | Accuracy | Balanced acc | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| majority | 86 | 0.6279 | 0.3333 | 0.2571 |
| oracle | 86 | 1.0000 | 1.0000 | 1.0000 |
| xgb_asis | 86 | 0.8605 | 0.9074 | 0.8941 |
| xgb_fitonly | 86 | 0.8488 | 0.8889 | 0.8827 |
| deep_asis | 86 | 0.8837 | 0.9012 | 0.9056 |
| shallow_asis | 86 | 0.8372 | 0.8642 | 0.8678 |
| deep_seedavg | 86 | 0.9070 | 0.9012 | 0.9190 |
| shallow_seedavg | 86 | 0.8953 | 0.8951 | 0.9101 |
| ensemble | 86 | 0.8953 | 0.9074 | 0.9141 |
| thr_xgb_fitonly | 86 | 0.9070 | 0.9012 | 0.9190 |
| thr_deep_asis | 86 | 0.8837 | 0.9012 | 0.9056 |
| thr_shallow_asis | 86 | 0.8372 | 0.8642 | 0.8678 |
| thr_ensemble | 86 | 0.9186 | 0.9136 | 0.9301 |

Self-checks: w1_reproduces_xgb_fitonly=True, w0_reproduces_deep_seedavg=True, all_ones_thresholds_reproduce_argmax=True, y_test_identical_across_sources=True, y_test_sources=['data/splits.npz', 'xgboost/y_test.npy', 'xgboost_tertile/y_test.npy', 'neural/y_test.npy'], models_varying_within_sentence=[], oracle_accuracy_dominates=True, oracle_macro_f1_dominates=True, neural_consistency={'deep': {'selected_seed_matches_result': True, 'temperature_max_abs_diff_vs_info': 0.0}, 'shallow': {'selected_seed_matches_result': True, 'temperature_max_abs_diff_vs_info': 0.0}}.
Missing pieces: none.

## csv10k

### Diagnostics

| Diagnostic | Value |
| --- | --- |
| rows / unique sentences | 10000 / 2410 |
| singleton / multi-row groups | 1393 / 1017 |
| largest group | 500 rows |
| multi-row groups disagreeing on class | 603 (0.593) |
| mean within-group score std | 0.0877 |
| bin edges / tertile edges | [0.4, 0.7] / [0.56, 0.73] |
| text normalisation | strip ' (example N)' suffix and collapse whitespace; used for grouping and model input |
| split protocol / seed | grouped / 42 |
| pin rule | 0.05 → 500 rows |
| fit / val / test rows | 6578 / 1521 / 1901 |
| fit / val / test sentences | 1533 / 401 / 476 |
| class counts (all) | low 540, mid 5381, high 4079 |
| class counts (fit) | low 385, mid 3550, high 2643 |
| class counts (val) | low 69, mid 814, high 638 |
| class counts (test) | low 86, mid 1017, high 798 |
| majority class (fit+val) | mid |

The oracle uses the test labels, is the best any deterministic text function can score, is not a model, and a model exceeding it indicates a bug.

### Variants (tuned on validation only; val macro-F1 is optimistic)

| Variant | Val macro-F1 | Test acc | Bal acc | Macro-F1 [row CI] [sentence CI] | NLL | ECE | Tuned |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| majority | 0.2324 | 0.5350 | 0.3333 | 0.2324 [0.226, 0.239] [0.217, 0.243] | — | — | mid |
| oracle | — | 0.7491 | 0.5199 | 0.5264 [0.499, 0.558] [0.480, 0.609] | — | — | — |
| xgb_asis | 0.5083 | 0.5976 | 0.6989 | 0.5322 [0.505, 0.557] [0.482, 0.577] | 0.7850 | 0.1478 | max_depth=5, min_child_weight=3, best_iteration=150 |
| xgb_fitonly | 0.5083 | 0.5960 | 0.6978 | 0.5309 [0.504, 0.555] [0.480, 0.575] | 0.7693 | 0.0979 | max_depth=5, min_child_weight=3, best_iteration=150 |
| deep_asis | 0.5199 | 0.6239 | 0.5359 | 0.4999 [0.475, 0.527] [0.453, 0.539] | 0.7047 | 0.0483 | seed=42, best_epoch=15, T=1.206 |
| shallow_asis | 0.5211 | 0.6412 | 0.5711 | 0.5317 [0.503, 0.559] [0.456, 0.581] | 0.7186 | 0.0360 | seed=44, best_epoch=5, T=0.991 |
| deep_seedavg | 0.5143 | 0.6028 | 0.6994 | 0.5333 [0.505, 0.557] [0.483, 0.580] | 0.7009 | 0.0852 | ensemble_temperature=0.967, seeds=42/43/44 |
| shallow_seedavg (exploratory) | 0.5097 | 0.6202 | 0.6889 | 0.5441 [0.516, 0.568] [0.492, 0.587] | 0.6963 | 0.0488 | ensemble_temperature=0.935, seeds=42/43/44 |
| ensemble | 0.5149 | 0.6039 | 0.7002 | 0.5342 [0.506, 0.558] [0.484, 0.581] | 0.7124 | 0.0677 | w=0.25, xgb_temperature=1.259 |
| thr_xgb_fitonly (exploratory) | 0.5179 | 0.6239 | 0.6250 | 0.5299 [0.504, 0.555] [0.480, 0.574] | — | — | cw=0.7/1.4/1 |
| thr_deep_asis (exploratory) | 0.5239 | 0.6249 | 0.5351 | 0.4986 [0.473, 0.525] [0.449, 0.540] | — | — | cw=1.4/1.4/1 |
| thr_shallow_asis (exploratory) | 0.5238 | 0.6518 | 0.5738 | 0.5334 [0.505, 0.562] [0.458, 0.585] | — | — | cw=2/2/1 |
| thr_ensemble | 0.5149 | 0.6039 | 0.7002 | 0.5342 [0.506, 0.558] [0.484, 0.581] | — | — | cw=1/1/1 |
| xgb_tertile (exploratory) (tertile labels) | 0.5207 | 0.4940 | 0.4967 | 0.4984 [0.475, 0.520] [0.461, 0.527] | 0.9497 | 0.0335 | max_depth=3, min_child_weight=3, best_iteration=197, edges=0.56/0.73 |

Ensemble weight search on validation:

| w (XGBoost share) | Val macro-F1 | Val NLL |
| ---: | ---: | ---: |
| 0.25 | 0.5149 | 0.7317 |
| 0.5 | 0.5085 | 0.7493 |
| 0.75 | 0.5067 | 0.7699 |

### Paired comparisons (candidate − baseline on the shared bootstrap matrix)

| Baseline → candidate | Δ macro-F1 | row CI | sentence CI | one-sided p | Δ accuracy | acc CI | acc p | Primary |
| --- | ---: | --- | --- | ---: | ---: | --- | ---: | --- |
| deep_asis → deep_seedavg | 0.0333 | [0.011, 0.055] | [0.001, 0.089] | 0.0010 | -0.0210 | [-0.035, -0.007] | 0.0020 | yes (α=0.0125) |
| deep_seedavg → ensemble | 0.0010 | [0.000, 0.002] | [0.000, 0.003] | 0.1449 | 0.0011 | [0.000, 0.003] | 0.4935 | yes (α=0.0125) |
| xgb_fitonly → ensemble | 0.0034 | [-0.005, 0.012] | [-0.013, 0.019] | 0.2308 | 0.0079 | [-0.002, 0.018] | 0.1788 | yes (α=0.0125) |
| ensemble → thr_ensemble | 0.0000 | [0.000, 0.000] | [0.000, 0.000] | 1.0000 | 0.0000 | [0.000, 0.000] | 1.0000 | yes (α=0.0125) |
| shallow_asis → shallow_seedavg | 0.0123 | [-0.016, 0.040] | [-0.026, 0.094] | 0.1838 | -0.0210 | [-0.038, -0.003] | 0.0270 | exploratory |
| xgb_fitonly → xgb_asis | 0.0014 | [-0.003, 0.006] | [-0.003, 0.007] | 0.2697 | 0.0016 | [-0.004, 0.007] | 0.6743 | exploratory |
| xgb_fitonly → thr_xgb_fitonly | -0.0009 | [-0.019, 0.014] | [-0.046, 0.016] | 0.5634 | 0.0279 | [0.016, 0.039] | 0.0010 | exploratory |
| deep_asis → thr_deep_asis | -0.0014 | [-0.006, 0.003] | [-0.005, 0.002] | 0.7373 | 0.0011 | [-0.005, 0.007] | 0.8501 | exploratory |
| shallow_asis → thr_shallow_asis | 0.0016 | [-0.007, 0.009] | [-0.007, 0.012] | 0.3656 | 0.0105 | [0.000, 0.021] | 0.0679 | exploratory |
| deep_asis → shallow_asis | 0.0318 | [0.002, 0.061] | [-0.057, 0.088] | 0.0200 | 0.0174 | [0.001, 0.035] | 0.0480 | exploratory |

### Neural seeds (validation)

| Head | Seed | Best epoch | Val macro-F1 | Val NLL | T |
| --- | ---: | ---: | ---: | ---: | ---: |
| deep | 42 | 15 | 0.5199 | 0.7276 | 1.2058 |
| deep | 43 | 6 | 0.5167 | 0.7106 | 0.9795 |
| deep | 44 | 5 | 0.5147 | 0.7256 | 0.9216 |
| shallow | 42 | 23 | 0.5150 | 0.7260 | 1.1714 |
| shallow | 43 | 6 | 0.5202 | 0.7189 | 0.9930 |
| shallow | 44 | 5 | 0.5211 | 0.7404 | 0.9914 |

### XGBoost validation trials

| Labels | max_depth | min_child_weight | Best iteration | Val macro-F1 | Val balanced acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| main | 3 | 3 | 253 | 0.5029 | 0.6507 |
| main | 3 | 8 | 254 | 0.5066 | 0.6536 |
| main | 5 | 3 | 150 | 0.5083 | 0.6624 |
| main | 5 | 8 | 189 | 0.5077 | 0.6580 |
| tertile | 3 | 3 | 197 | 0.5207 | 0.5157 |
| tertile | 3 | 8 | 287 | 0.5156 | 0.5105 |
| tertile | 5 | 3 | 117 | 0.5145 | 0.5109 |
| tertile | 5 | 8 | 127 | 0.5183 | 0.5154 |

### Sentence-level secondary table (test deduplicated by sentence, label = test majority)

| Variant | Sentences | Accuracy | Balanced acc | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| majority | 476 | 0.5651 | 0.3333 | 0.2407 |
| oracle | 476 | 1.0000 | 1.0000 | 1.0000 |
| xgb_asis | 476 | 0.7395 | 0.4934 | 0.4975 |
| xgb_fitonly | 476 | 0.7374 | 0.4922 | 0.4973 |
| deep_asis | 476 | 0.7542 | 0.5009 | 0.5026 |
| shallow_asis | 476 | 0.7542 | 0.5025 | 0.5030 |
| deep_seedavg | 476 | 0.7626 | 0.5058 | 0.5094 |
| shallow_seedavg | 476 | 0.7668 | 0.5087 | 0.5117 |
| ensemble | 476 | 0.7647 | 0.5075 | 0.5110 |
| thr_xgb_fitonly | 476 | 0.7605 | 0.5050 | 0.5083 |
| thr_deep_asis | 476 | 0.7584 | 0.5017 | 0.5045 |
| thr_shallow_asis | 476 | 0.7710 | 0.5112 | 0.5131 |
| thr_ensemble | 476 | 0.7647 | 0.5075 | 0.5110 |

Self-checks: w1_reproduces_xgb_fitonly=True, w0_reproduces_deep_seedavg=True, all_ones_thresholds_reproduce_argmax=True, y_test_identical_across_sources=True, y_test_sources=['data/splits.npz', 'xgboost/y_test.npy', 'xgboost_tertile/y_test.npy', 'neural/y_test.npy'], models_varying_within_sentence=[], oracle_accuracy_dominates=True, oracle_macro_f1_dominates=False, neural_consistency={'deep': {'selected_seed_matches_result': True, 'temperature_max_abs_diff_vs_info': 0.0}, 'shallow': {'selected_seed_matches_result': True, 'temperature_max_abs_diff_vs_info': 0.0}}.
Missing pieces: none.

## Caveats

- CSVs are template/AI-generated with synthetic scores
- label-noise ceilings (oracle)
- bin edges are a modelling choice
- effective test sizes (csv5k: 86 test sentences; csv10k: 476 test sentences, 500-row sentence pinned to fit)
- validation reuse / multiple comparisons
- MSC rows are a reproduction, not a new result
- nothing in `main/`, `artifacts/importance*` or the shipped reports changed
