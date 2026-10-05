# Neural retention model: deeper head experiment

The optional neural scorer predicts invalid, short-term, or long-term from
[Personal Facts (MSC)](https://huggingface.co/datasets/adugeen/personal-facts-msc).
Its output is a constructed retention proxy,
`P(long_term) + 0.5 × P(short_term)`, not a human-rated importance score.

Backpropagation was already used in the original head: every training batch
called `loss.backward()` followed by an AdamW optimizer step. This experiment
tested whether additional trainable layers improved the model.

## Architecture

Both heads use the same frozen local 384-dimensional BGE embedding and 13
standardized numeric features. The original head had 60,547 parameters:

```text
Text 384 → 128 ─┐
                ├→ 160 → 64 → 3 class logits
Numeric 13 → 32 ┘
```

The deeper head has 131,491 parameters, adding one dense layer to each branch
and one to their fused representation:

```text
Text 384 → 192 → 128 ─┐
                      ├→ 160 → 128 → 64 → 3 class logits
Numeric 13 → 64 → 32 ┘
```

LayerNorm, GELU, and 0.2 dropout are retained in the text and fusion paths.
PyTorch trains the head; the selected head is exported to ONNX for local
inference. The encoder remains frozen. The source's 2,222 usable training facts
were split into 1,777 fit and 445 validation examples. No test examples were
used to choose the architecture or seed.

## Validation selection

Each head was trained with seeds 42, 43, and 44 under the same class-weighted
cross-entropy, AdamW, early-stopping, and preprocessing settings. Selection
used validation macro F1, with lower validation loss as a tie-breaker.

| Head | Parameters | Best validation macro F1 | Mean across seeds |
| --- | ---: | ---: | ---: |
| Original | 60,547 | 0.7175 | 0.7134 |
| Deeper | 131,491 | **0.7291** | **0.7194** |

The selected deeper head was seed 44, epoch 8. Temperature calibration on
validation data reduced its negative log-likelihood from 0.7913 to 0.5385.
The [validation experiment](neural_depth_validation.json) records every seed.

## Published 556-fact test split

| Metric | Deeper neural | Original neural | Frozen XGBoost | Always long-term |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 78.06% | 78.06% | 69.24% | 68.17% |
| Balanced accuracy | **67.78%** | 65.86% | 65.60% | 33.33% |
| Macro F1 | **0.6732** | 0.6504 | 0.6114 | 0.2702 |
| Invalid-fact recall | **42.35%** | 31.76% | 50.59% | 0% |
| Constructed proxy MAE | 0.2160 | **0.1957** | 0.2768 | — |

The deeper head classified more invalid facts correctly: 36 of 85 versus 27
for the original neural head. It still recalled fewer invalid facts than
XGBoost's 43 of 85. Its test negative log-likelihood was 0.5547 versus
0.5345 for the original head. Its 10-bin expected calibration error was
0.0163 versus 0.0428. Warm local inference including BGE was P50 4.2 ms and
P95 5.8 ms across 50 facts on this machine; these are not service latencies.

The deeper head is now the default **optional neural** artifact. The original
artifact remains available at `artifacts/importance_neural_shallow/`, selectable
with `--scorer neural --model-dir artifacts/importance_neural_shallow`.
CogniMem's heuristic remains the default scorer. Neither model is suitable for
automatic deletion: the dataset lacks human numeric importance ratings,
underrepresents greetings and task requests, and has no conversation-disjoint
test split. The published test split was also examined in earlier XGBoost work,
so these results are exploratory rather than an independent generalization claim.

## Reproduce

```bash
.venv/bin/python -m training.compare_neural_depth
.venv/bin/python -m training.train_neural_retention
.venv/bin/python -m unittest discover -s tests -q
```

The [machine-readable result](neural_retention.json) and
[original-head result](neural_retention_shallow.json) preserve the full class
metrics, confusion matrices, calibration, and measured latency.
