# Conversational evaluation and review

`conversational_cases.jsonl` contains 48 developer-authored examples with proposed
category, tier, and importance ranges. It covers greetings, factual statements,
preferences/constraints, one-off/recurring tasks, negation, events, procedures, and
indirect paraphrases. These examples were not used to train an ML model. They are
regression/development checks, not an independent benchmark or human ground truth.

Run `python3 -m evaluation.evaluate` from the repository root. The report includes
all predictions, a category confusion matrix, macro F1, tier accuracy, and agreement
with the proposed importance ranges. Ranges represent a retention policy, not
human-measured continuous scores. Never report range agreement as regression accuracy.

The current run has 48/48 category and tier matches (100%), macro F1 1.0,
and 48/48 importance-range matches. The four previous failures now pass after
extending the preference/event rules. The original labels are unchanged.
These are examples used during development, so the result is regression coverage,
not an estimate of accuracy on unseen conversations. Stateful task workflows,
calendar arithmetic, and ambiguity handling are covered by automated tests.

## Human review

`review_template.jsonl` has blank authoritative labels, blank reviewer/date fields,
and the proposed labels in a separate field. A reviewer should independently
assess each example, fill `expected_category`, `expected_tier`, `importance_range`,
`reviewer`, `reviewed_at`, and set `label_source` to `human_reviewed` only after
actual review. Ambiguous cases should have notes and be adjudicated. Proposed labels
can anchor judgment; for a blind review remove `proposed_labels` before distribution.

Use these definitions:

- semantic: durable factual knowledge about a user or their world;
- preference: standing choices, communication style, or user constraints;
- task: an outstanding action or recurring reminder;
- episodic: something that happened or changed;
- procedural: a repeatable sequence or workflow;
- temporary: transient chatter without useful retained content.

Tier judgments must distinguish importance from duration: one-off tasks are
short-term even when important; standing preferences/facts can be long-term;
greetings are working. Archive decisions need age/access history and cannot be
validated from a timeless sentence alone. Scenario tests cover archive mechanics.
Importance ranges should reflect usefulness for future assistance, not emotional
intensity alone. Reviewers may disagree with the existing policy and should record
that disagreement rather than copy predictions.

```bash
python3 -m evaluation.evaluate --export-review evaluation/review_template.jsonl
python3 -m evaluation.evaluate --labels reviewed.jsonl --reviewed-only \
  --output reports/human_reviewed_evaluation.json
```

The reviewed-only command fails if no eligible reviewed rows exist. No human review
has been performed in this change. For independent evidence, add a fresh blind
set from consented conversations, keep conversation/user groups together, freeze
it before tuning, and report reviewer agreement. Do not tune rules against a test
set and continue calling it held out. Training/calibration splits are not created
from these developer examples.

## External retrieval and memory evaluation

```bash
uv pip install --python .venv/bin/python -r requirements-evaluation.txt
.venv/bin/python -m evaluation.benchmark --download
.venv/bin/python -m evaluation.longmemeval --download
.venv/bin/python -m evaluation.conflicts --download
```

Downloads pin dataset revisions and verify SHA-256. Data/model caches live under
ignored `data/`; reports are committed separately. No hosted model is used by
these three commands. The embedding model downloads once and runs locally.

- **LoCoMo:** all ten external conversations; only text turns, speaker, and dates
  are indexed. QA, observations, and generated summaries are excluded. Compare
  recency, ranked keyword, local semantic, graph, and hybrid retrieval at k=5.
  Report evidence recall/hit rate/MRR/nDCG, per-category results, conversation
  bootstrap intervals, index size, and evidence characters. The preference-word
  subset measures evidence retrieval only, not recommendation personalization.
  Category 5 adversarial questions have reference evidence but are unanswerable;
  both overall reference retrieval and categories 1–4 metrics are reported.
  LoCoMo is a published generated-dialogue benchmark with annotations, not a
  collection of real user chats. No weights/rules were fitted on scored queries.
- **LongMemEval:** all knowledge-update and single-session-preference examples
  from the oracle split. Process original turns through the existing core/store,
  use original timestamps, and measure retention of `has_answer`-labeled turns
  at question time against a retain-everything baseline. Labels never enter the
  memory documents. These results test retention, not generated-answer accuracy;
  oracle sessions omit full-corpus distractors. Six update examples lack labeled
  answer turns and do not enter the retention denominator.
- **bAbI QA1:** all published English test stories. A declared deterministic
  adapter converts movement statements into the core's supported named-person
  location claim form. Gold answers remain unchanged. Compare current memories
  and temporal graph location answers with first/latest-assertion baselines.
  This tests conflict state updates on an external synthetic dataset; it is not
  a claim of natural-language parsing or independent human policy validation.

See [the consolidated results](../reports/retrieval_graph_report.md). Reports
include predictions, dataset fingerprints, exclusions, protocols, and limitations.
LoCoMo raw-turn retrieval bypasses memory lifecycle decisions deliberately to
isolate retrieval; LongMemEval separately exposes end-to-end retention losses.
The poor LongMemEval/graph-coverage results are preserved, not hidden by tuning
on these datasets. A future tuned system needs a fresh held-out split.

Independent human ratings of conflict policy and generated personalization, live
Gemini answers, and comparisons to deployed Mem0/MemGPT/LangMem systems remain
unmeasured. The existing human-review workflow is for category/tier/importance
labels and must not be described as covering those other judgments.
