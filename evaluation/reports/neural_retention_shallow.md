# Frozen-encoder neural retention proxy

The optional neural model predicts whether an extracted conversational fact is
invalid, short-term, or long-term. It was trained on the same
[Personal Facts (MSC)](https://huggingface.co/datasets/adugeen/personal-facts-msc)
labels used by the frozen XGBoost baseline. The dataset has no human-rated
numeric importance values. The 0–1 output remains a policy proxy:
`P(long_term) + 0.5 × P(short_term)`.

## Architecture and training

Local BGE-small-en-v1.5 supplies a frozen, normalized 384-dimensional passage
embedding. A text branch applies `Linear(384,128) → LayerNorm → GELU → Dropout(0.2)`.
Thirteen numeric features from the existing ML extractor are standardized using
training-only statistics and passed through `Linear(13,32) → LayerNorm → GELU`.
The two branches concatenate to 160 dimensions, followed by
`Linear(160,64) → GELU → Dropout(0.2) → Linear(64,3)`. Softmax probabilities
are temperature-calibrated on validation data. PyTorch trains the head; the
saved ONNX model runs locally through ONNX Runtime.

The published 2,223-row train split supplied 2,222 usable examples; one valid
fact lacked a duration label. A stratified 80/20 training/validation split
selected among seeds 42, 43, and 44 by validation macro F1, then validation
negative log-likelihood. Seed 42, epoch 4 won. Temperature scaling reduced
validation NLL from 0.5644 to 0.5332. The published 556-row test split was
evaluated once after selection. XGBoost figures come from the frozen prior
report on that same published test split; the two native runtimes are evaluated
in separate processes.

## Published test-split comparison

| Metric | Neural | Frozen XGBoost | Always long-term |
| --- | ---: | ---: | ---: |
| Accuracy | 78.06% | 69.24% | 68.17% |
| Balanced accuracy | 65.86% | 65.60% | 33.33% |
| Macro F1 | 0.6504 | 0.6114 | 0.2702 |
| Weighted F1 | 0.7703 | 0.7071 | 0.5526 |
| Constructed proxy MAE | 0.1957 | 0.2768 | — |

| Class | Support | Neural precision | Neural recall | Neural F1 | XGBoost recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Invalid | 85 | 0.6585 | 0.3176 | 0.4286 | 0.5059 |
| Short-term | 92 | 0.5547 | 0.7717 | 0.6455 | 0.7391 |
| Long-term | 379 | 0.8682 | 0.8865 | 0.8773 | 0.7230 |

Neural confusion matrix (rows are gold; columns are predicted):

| Gold / predicted | Invalid | Short-term | Long-term |
| --- | ---: | ---: | ---: |
| Invalid | 27 | 26 | 32 |
| Short-term | 2 | 71 | 19 |
| Long-term | 12 | 31 | 336 |

Neural test negative log-likelihood was 0.5345 and 10-bin expected calibration
error was 0.0428. Warm, single-fact local inference including the BGE encoder
had a measured P50 latency of 5.0 ms and P95 of 6.7 ms across 50 test facts.
These are local machine measurements, not production service latencies.

The neural model improves overall accuracy and macro F1 on this test split, but
recognizes fewer invalid facts than XGBoost: 27/85 versus 43/85. Neither model
is suitable as an automatic deletion decision maker. This is an exploratory
comparison because the test split was already inspected during XGBoost work;
the dataset contains extracted facts, underrepresents greetings and task
requests, and does not provide human importance scores. The heuristic remains
the default, and lifecycle thresholds are unchanged.

## Reproduce and use

```bash
uv pip install --python .venv/bin/python -r requirements-neural.txt
.venv/bin/python -m training.train_neural_retention \
  --architecture frozen-bge-two-branch-v1 \
  --artifact-dir artifacts/importance_neural_shallow \
  --report-name neural_retention_shallow.json
.venv/bin/python -m main process "I prefer concise answers." \
  --user-id demo --session-id demo --scorer neural \
  --model-dir artifacts/importance_neural_shallow --no-save
```

Machine-readable metrics are in
[neural_retention_shallow.json](neural_retention_shallow.json).
The checksum-verified ONNX artifact, training weights, and metadata are in
`artifacts/importance_neural_shallow/`. The deeper head is now the default
optional neural artifact; the heuristic remains CogniMem's default scorer.
