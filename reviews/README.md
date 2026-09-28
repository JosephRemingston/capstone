# Independent human review protocol

Human judgments must be supplied by reviewers who did not implement or tune the
system. Do not fill these labels from model output or developer expectations.

For each frozen case in `template.jsonl`, give every reviewer the user question,
the answer, its citations, the no-memory baseline, and the relevant conflict
decision. Hide system names and randomize answer order. Each case needs at least
two reviewers; disagreements should be adjudicated by a third.

Score answer usefulness, personalization, citation support, and conflict-decision
quality from 1 (poor) to 5 (excellent). Reviewers must record an identifier,
review date, rationale, and whether they saw sensitive information. Report the
mean, median, distribution, inter-rater agreement, abstention rate, and number of
adjudicated cases. A blank review file means evaluation is pending, never zero.

The repository cannot truthfully claim independent human validation until real
reviewers complete and sign this file.
