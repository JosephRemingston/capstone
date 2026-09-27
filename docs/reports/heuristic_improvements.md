# Conversational heuristic regression experiment

These examples are developer-authored regression checks, not independent human-labeled evaluation data. No model was retrained and these results do not establish benchmark accuracy.

| Sentence | Previous category / score / tier | New category / score / tier |
|---|---|---|
| Hello! | temporary / 0.0402 / working | temporary / 0.0402 / working |
| Thank you! | temporary / 0.0304 / working | temporary / 0.0304 / working |
| I prefer concise Python explanations. | preference / 0.6497 / long_term | preference / 0.6497 / long_term |
| I dislike loud music. | preference / 0.6330 / long_term | preference / 0.6330 / long_term |
| My name is Joseph. | semantic / 0.5030 / long_term | semantic / 0.5030 / long_term |
| I live in Chennai. | semantic / 0.4830 / long_term | semantic / 0.4830 / long_term |
| My project is CogniMem. | semantic / 0.4830 / long_term | semantic / 0.4830 / long_term |
| Remind me to submit the report tomorrow. | task / 0.6867 / long_term | task / 0.6867 / short_term |
| I need to buy groceries today. | task / 0.6745 / long_term | task / 0.6745 / short_term |
| First run the tests, then deploy the application. | procedural / 0.4360 / long_term | procedural / 0.4360 / long_term |
| Yesterday I met my project guide. | episodic / 0.2178 / working | episodic / 0.2978 / short_term |
| I submitted my report today. | episodic / 0.2937 / short_term | episodic / 0.2837 / short_term |
| My favorite book is Dune. | task / 0.6657 / long_term | preference / 0.6657 / long_term |
| Never include peanuts in my meals. | temporary / 0.0245 / working | preference / 0.5978 / long_term |
| The meeting has been cancelled. | temporary / 0.0249 / working | episodic / 0.2997 / short_term |

## Methods

- Whole-word matching and action-position rules replace ambiguous substring matching in the default classifier.
- Recognize explicit user constraints as preferences within the existing category schema.
- Recognize selected completed/cancelled/postponed events before task keyword checks; explicit new requests retain task priority.
- Remove the rule that automatically discards all messages of two words or fewer.
- Require task context for deadline signals; completed events do not get a deadline bonus.
- Raise episodic category weight from 0.12 to 0.20 so fresh recognized events normally clear the working threshold.
- Keep one-off tasks short-term; explicitly recurring tasks can still qualify as long-term.

## Limits

- Short-term expiry is still 14 days from creation, not the parsed task deadline. No job enforces it.
- No conflict resolution is performed: classifying a cancellation does not update a previous meeting record.
- The rules are English-specific and limited; negation, quotations, mixed intents, and unseen paraphrases may still be misclassified.
- Recurrence detection is only a phrase-based signal, not a task scheduler.
- Thresholds are hand-selected policy rules, not statistically calibrated on human labels.
- ML uses the frozen legacy classifier and feature extractor to avoid changing the inputs expected by the existing Hippocorpus artifact. Thus these classifier fixes apply to the default pipeline; the lifecycle task policy applies to both.
- Existing stored records are not migrated or rescored.
