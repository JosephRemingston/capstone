# CogniMem evaluation

This package measures CogniMem end to end: a controlled retrieval and answer ablation on a
fresh benchmark, external evidence retrieval on LoCoMo, evidence retention on LongMemEval,
category/tier classification, and lifecycle behaviour. It was rebuilt from scratch on
12 October 2026 together with a new benchmark, so earlier saved numbers are not comparable.

## Run everything

```bash
.venv/bin/python scripts/measure_all.py --tag baseline --candidate-limit 20 --ann-arms   # saved protocol
.venv/bin/python scripts/measure_all.py --tag final --candidate-limit default --ann-arms # shipped defaults
.venv/bin/python scripts/report_delta.py baseline final --allow-dataset-mismatch
```

`measure_all.py` runs each step as its own process, logs to `docs/reports/logs/`, snapshots
`docs/reports/` to `docs/reports/<tag>/` with a `MANIFEST.json`, and copies the flat summary
to the tracked `evaluation/reports/<tag>_summary.json` (`docs/` is git-ignored).

- Only reports written by the run reach its snapshot. Reports of steps skipped with
  `--skip-locomo` or `--skip-retention` are kept and listed under `reused` in the manifest.
- The manifest records the command line, the git head, a hash of the uncommitted diff, the
  dataset hash, and each step's command and exit code.
- An existing tag is never replaced unless `--overwrite` is given.
- `report_delta.py` refuses to compare snapshots built from different benchmark files unless
  `--allow-dataset-mismatch` is passed, and prints every protocol setting that differs.

Other flags shared by the ablation and LoCoMo runners:

- `--candidate-limit default` uses the library pool: 100 for reranked arms, `max(20, 4×K)` for fusion-only arms.
- `--structure default|none|graph,validity,chain,versions` sets the graph/time-aware reordering after the reranker; `none` keeps the plain reranker order.
- `--abstain-threshold 0.0` turns on the experimental extractive abstention rule (slot rule v2).

Deep result lists for Recall@20 and pool metrics are read with `fusion_limit=K`, so their first K
entries are exactly what a product call with `limit=K` returns.

## Modules

| Command | What it measures | Output |
| --- | --- | --- |
| `python -m evaluation.build_dataset [--check] [--lint]` | Compiles and validates the benchmark seeds | `evaluation/datasets/gold_queries.json` |
| `python -m evaluation.ablation [--split dev\|test] [--candidate-limit N\|default] [--top-k K] [--ann] [--abstain-threshold T]` | Retrieval, pool coverage, grounded answers, answerability, latency for five arms | `docs/reports/ablation/results.{json,md}` |
| `python -m evaluation.locomo [--download] [--candidate-limit N] [--ann]` | External evidence retrieval on all 10 LoCoMo conversations; `held_out_metrics` covers the 7 conversations not used to choose the pool size | `docs/reports/locomo_retrieval.json` |
| `python -m evaluation.retention [--download]` | LongMemEval answer-turn retention through the full pipeline, plus cleanup safety | `docs/reports/longmemeval_retention.json` |
| `python -m evaluation.classification` | 60 category/tier/importance cases | `docs/reports/conversational_evaluation.json` |
| `python -m evaluation.lifecycle` | 16 deterministic lifecycle, task, and supersession checks | `docs/reports/lifecycle.json` |
| `python -m evaluation.report [--retention-only]` | Renders `SUMMARY.md` and `summary.json` from the saved JSON and summarises the importance model from `evaluation/reports/retention_ensemble.json`; `--retention-only` only rewrites `retention_summary.json` | `docs/reports/SUMMARY.md`, `docs/reports/summary.json`, `evaluation/reports/retention_summary.json` |

## Importance model

`python -m training.ensemble.run` trains the model half of the default hybrid importance scorer: the csv5k
"Ensemble + thresholds" (XGBoost blended with three seeds of a deep head on frozen BGE embeddings, with temperatures,
blend weight and class weights chosen on validation only). It reads `data/experiments/csv5k.csv`, writes the
tracked serving artifact `artifacts/importance_ensemble/` and the tracked report
`evaluation/reports/retention_ensemble.{json,md}`, and keeps working files in the git-ignored
`artifacts/ensemble_training/`. On a fresh clone that CSV is missing and the bare command fails; the first run
needs `python -m training.ensemble.run --csv /path/to/cognimem_memory_importance_5000.csv`, which copies it to the
git-ignored `data/experiments/csv5k.csv` (later runs need no flag). Training reproduces the model files bit for
bit (`xgboost.ubj` and the three ONNX files, whose SHA-256 values are listed under `files` in `metadata.json`);
`metadata.json` itself carries a `created_at` timestamp, so its hash (`model_sha256` in provenance and
`/health/ready`) changes on every retrain. On the synthetic csv5k test split
(1,008 rows, 86 sentences) it reaches macro F1 0.8912 against 0.2529 for always the most common class.

- `measure_all.py --scorer-steps-only` runs only retention, classification and lifecycle plus the summary, the
  steps that use the scorer; the ablation and LoCoMo never call it. `--scorer hybrid|heuristic` sets
  `COGNIMEM_SCORER` for every step (default: the environment, else `hybrid`) and the manifest records it.
- Every scorer-dependent runner records which scorer ran (`importance_model` in its report). The classification
  and lifecycle runners pin the clock (`evaluation/clock.py`), so their results do not depend on the day.
- `report_delta.py` also compares cleanup counts (`longmemeval._cleanup.*`). Compare memory counts, not bytes:
  every record now carries its scorer stamp (about 40 bytes for the heuristic, about 300 more for the hybrid),
  so byte totals differ even when cleanup removes the same memories.
- Summaries carry `_protocol.hybrid_scorer` and `_protocol.clock_pinned` markers, so `report_delta.py` warns when
  two snapshots used different scorers or clock handling; isolate a scorer's effect against a same-day run such as
  `heuristic_control`, not against `final`.
- System-level result (16 Oct 2026, snapshots `hybrid` and `heuristic_control`): LongMemEval user-fact retention
  is 90.55% with either scorer, category and tier accuracy (53.3% / 56.7%) and the lifecycle checks (16/16) match,
  the model's score does not separate answer-bearing turns (AUC 0.482 vs 0.582 for the heuristic), and the replay
  takes 581 s instead of 14 s; see [Importance scoring (default)](../README.md#importance-scoring-default).

## The benchmark

- **Authorship.** Twelve personas, 456 memories and 360 questions were written by Claude (AI)
  on 12 October 2026 in `evaluation/datasets/seeds/`. They are not human-validated.
- **One shared corpus.** Every memory belongs to `gold-user`; the persona is the memory's
  speaker and every question names the persona. Cross-persona memories act as distractors,
  so pool size and approximate search have a measurable effect.
- **Unanswerable questions.** The 30 unanswerable questions were rewritten on 12 October 2026
  (abstention rework) to cover 26 attribute types instead of two templates; extractive answers
  without abstention refuse none of them either way.
- **Twelve query types, 30 each.** Semantic, episodic, temporal (current value after a
  change), historical (`as_of` inside an old relation's validity), multi-hop (two memories
  joined by a shared entity), entity relationship, conflict (explicit corrections), preference,
  procedural, recency (latest of several updates), negative/absence, and unanswerable
  (distractor-adjacent: the entity exists but the asked attribute was never stated).
- **Split.** Personas p01–p03 are dev (90 questions) for tuning thresholds; p04–p12 are test
  (270 questions) for every reported number. The whole corpus is indexed in both runs.
- **Loading.** Memories are written with raw persistence, not `ingest()`, so reconciliation
  cannot silently merge or supersede gold evidence across personas. Reconciliation is covered
  by the lifecycle checks instead.
- **Validators** (`build_dataset.py`) check that every answer appears in a supporting memory,
  every historical question has a relation valid at `as_of`, every multi-hop pair shares an
  entity and uses predicates the query cues, no closed relation is contradicted by an open
  claim extracted from the same text, and all graph metadata validates.

## Metrics

- Retrieval: Recall/Hit/nDCG at 5, 10, 20 and MRR over answerable questions.
- Pool coverage: whether gold evidence reached the candidate pool handed to the reranker, and
  reranker recovery (Hit@5 divided by pool hit).
- Grounded correctness (primary answer metric): the answer did not abstain and an accepted
  answer form appears in a supporting memory that reached the context. A different persona's
  snippet with the same word does not count. Top-1 correctness is the strict variant.
- Answerability: answerable questions accepted, unanswerable questions refused, balanced
  accuracy, and the confusion matrix.
- Paired bootstrap 95% intervals and sign-flip randomization p-values between arms.

## Scope cuts in this rebuild

Not rebuilt yet: the bAbI conflict runner, the LongMemEval personalization review packet,
human-review scoring, and the hosted Gemini answer arm. The answer metrics here use the local
extractive generator, which can measure grounding and abstention but not fluent answer quality.

## Tuning disclosure

- The benchmark's dev split (p01–p03) and a dev-only abstention calibration set were used for
  tuning; reported benchmark numbers are from the test split.
- LoCoMo conversations 1–3 were used to choose the reranker pool size; the held-out section
  covers the other seven.
- The retention-classifier change was designed against LongMemEval failures (dev examples),
  then confirmed once on held-out examples.
- The importance model's temperatures, blend weight and class weights were chosen on the csv5k
  validation split (grouped by sentence); its reported numbers are from the test split.

## Limits

All benchmark labels are AI-authored and unreviewed. LoCoMo is generated dialogue, not real
user chats. LongMemEval oracle sessions remove retrieval distractors, so its numbers measure
retention, not question answering. Latency is measured on a single development machine.

The importance model is trained and tested on csv5k, which is synthetic (AI-generated sentences
and scores, not human ratings), and its test split has only 86 distinct sentences. Conversation
turns are far longer than csv5k sentences (at most 13 words; LongMemEval user turns median
42 words), so it runs out of distribution. Its 25% share of the hybrid score is a fixed setting
chosen by the user, not tuned or calibrated; the out-of-distribution evidence is a reason to keep
it small. Before the small-talk penalty it rates greetings mid or high, and its low/mid/high target
differs from the invalid/short-term/long-term target of the removed models.
