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

The current run has 44/48 category and tier matches (91.67%), macro F1 0.9150,
and 44/48 importance-range matches. Four failures remain visible: indirect
preferences and the phrase “got called off.” The evaluator does not silently
remove these cases or adjust labels to match the implementation.

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
