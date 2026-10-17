# Research and training toward 80% answer correctness

The user clarified the target as **80% accuracy**, not 98%. This is an objective, not a guaranteed outcome. No paid training budget was provided, so this experiment uses local CPU compute and existing cached models.

## What the research supports

- **Fix evidence coverage before assuming answer fine-tuning will help.** LongMemEval separates indexing, retrieval, and reading, and evaluates both retrieved and oracle evidence. It studies session decomposition, fact-expanded keys, temporal query expansion, and structured reading. These motivate measuring retrieval failures separately from reader failures. [LongMemEval, ICLR 2025](https://arxiv.org/abs/2410.10813).
- **Train rankers on domain-relevant pairs and hard negatives.** Sentence Transformers describes pointwise binary cross-entropy, contrastive, and ranking-loss training. Its examples evaluate reranking on retrieved candidate pools rather than artificially inserting gold documents at test time. This motivates a small supervised retrieval adapter first, with a full cross-encoder fine-tune as a subsequent controlled experiment. [Official reranker training documentation](https://www.sbert.net/examples/cross_encoder/training/rerankers/README.html).
- **Preserve benchmark integrity.** LoCoMo is a conversational-memory evaluation dataset. Training on its examples means the original whole-dataset score is no longer an independent test. This repository has already inspected all ten conversations, so even conversation-disjoint splits are labeled exploratory. [Original data and evaluation code](https://github.com/snap-research/locomo).
- **Use complementary retrieval signals.** A recent preprint studies lexical/dense fusion and reports gains that differ by query type. It supports testing lexical retrieval alongside dense retrieval, not assuming a single model dominates. Its published results are not CogniMem results and are not directly comparable to our protocol. [Training-Free Lexical-Dense Fusion for Conversational-Memory Retrieval](https://arxiv.org/abs/2606.04194).
- **Gemini tuning needs a different service.** The current Gemini API/AI Studio documentation says it does not offer fine-tuning. Managed supervised tuning is available through Google's enterprise platform, with supported models, dataset upload, and a separate training job. An API key alone does not provision that training setup. No cloud tuning job has been submitted. [Gemini API tuning availability](https://ai.google.dev/gemini-api/docs/model-tuning), [managed supervised tuning](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-use-supervised-tuning).

## Repository diagnosis

The latest completed local extractive benchmark has vector-plus-reranker evidence recall of 55.83%. That is an average labeled-evidence retrieval metric, not an exact ceiling on answer correctness. It nevertheless shows substantial missing context. The hosted assessment/generation/verification path has not completed a representative external evaluation; its failures must not be replaced with successful-abstention labels.

The existing neural/XGBoost retention models predict retention policy. They do not answer questions or train retrieval. Improving their classification accuracy cannot establish an improvement in RAG answer quality.

The 1,527 eligible answerable LoCoMo questions include 21 requiring more than five cited records, with a maximum of 19. The theoretical mean Recall@5 ceiling from annotation cardinality alone is 99.50%; this small effect does not explain the much larger measured recall deficit. Conversely, if exactly five memories are returned, the mean annotation-based Precision@5 ceiling is only **29.48%**, because there are about 1.525 labeled evidence records per question on average. Requiring every percentage metric to exceed 80% would be mathematically inconsistent with those labels and that fixed return count. This is not a claim that every unlabeled memory is semantically irrelevant.

## Implemented local experiment

Run:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m training.train_retrieval_adapter
```

1. Verify the pinned LoCoMo dataset checksum.
2. Exclude adversarial category 5 from this *relevance training* task. Its evidence annotations do not establish answerability.
3. Retrieve up to 60 candidates from keyword and exact-vector channels with reciprocal-rank fusion. Never inject missing gold evidence into validation or test candidates.
4. Score candidates with the frozen MiniLM cross-encoder.
5. Build eight numeric features from query text, retrieved document text, speaker metadata, and the cross-encoder score. No answer string, question category, gold ID, or conversation ID is a model feature.
6. Assign six entire conversations to training, two to validation, and two to test using seed 42. No conversation crosses partitions.
7. Fit a standardized logistic ranking adapter with query weighting and class balancing. Compare four regularization strengths using validation nDCG@5, then Recall@5, then stronger regularization for ties.
8. Evaluate the selected model on the test conversations once and compare it with the frozen cross-encoder using exactly the same candidates.
9. Export JSON numeric weights and a checksum manifest; the optional runtime adapter requires the same frozen base model. No pickle is loaded.

This trains a small supervised adapter. It does **not** fine-tune BGE, MiniLM, or Gemini. The original production default remains selectable and is not silently replaced by an experimental model. The runtime option is:

```bash
.venv/bin/python -m main ask 'Your question' --user-id YOUR_USER \
  --mode hybrid --candidate-limit 60 \
  --ranking-adapter evaluation/reports/retrieval_training/ranker.json
```

The training report's baseline differs from earlier four-arm reports: answerable-only questions, a 60-candidate keyword/vector pool, and direct cross-encoder ranking. Compare adapter versus frozen baseline **within this report**, not against an incompatible previous aggregate.

## Criteria for an 80% claim

- Freeze a representative evaluation split with unseen users/conversations, question categories, and evaluation rubric before model selection.
- Measure actual supported answer correctness, separately on answerable and unanswerable questions; a non-abstention is not automatically correct.
- Report answerable correctness, unanswerable refusal rate, balanced accuracy, grounded claim support, citation completeness, retrieval Recall@K, and latency. Not every metric is an accuracy percentage; latency should decrease.
- Include no-memory, retrieval-only, frozen-model, and trained-model baselines with identical context budgets.
- Report confidence intervals and per-category failures, plus blinded human checks of model-judged answer quality. The generator must not supply its own gold labels.
- Keep provider/validation failures visible as operational failures, not evidence that a question is unanswerable.
- Report full-dataset accuracy as well as selective answered-only accuracy and coverage. Do not reach 80% by refusing almost everything or excluding failed questions.

## Next experiments justified by the outcome

If candidate recall is low, test session-neighbor expansion, multi-key indexing, and time-aware retrieval before enlarging the language model. If gold evidence is present but ranking fails, train a cross-encoder on external conversational training data with mined negatives and evaluate on locked users. If adequate evidence is supplied but the reader fails, collect reviewed complete/partial/unsupported answer examples before LoRA or managed Gemini tuning. Provider reliability must be resolved before running a large hosted evaluation.

No training result here should be called independent generalization or an achieved 80% end-to-end answer accuracy without the corresponding evidence.
