# Conflict resolution and memory consolidation

## Implemented

The ingestion path now extracts supported assertions and compares them with the
same user's current memories. Assertions retain subject, attribute, normalized
value, polarity, and whether the attribute is exclusive. Different subjects and
attributes cannot conflict. Residence, employer, occupation, name, favorites,
response length, and explicit comparative choices support replacement decisions.
Opposing likes/dislikes and allergy statements conflict for the same object;
liking two different things or having multiple allergies does not conflict.

Automatic precedence is source priority, confidence, observation/confirmation
time, then importance. Exact ties keep an existing assertion. Each group uses
its strongest actual observation, rather than combining one source's confidence
with another's date. A rejected assertion leaves compatible existing facts
current. Caller-assigned priority/confidence are validated, but are not calibrated
probabilities or authenticated source trust.

Superseded records retain their original content and point to the winner.
`resolve_conflict` allows explicit selection/restoration of a supported root with
active contradictions. Confirmation renews retention and observation ordering;
subsequent ingestion still uses the normal precedence policy. Every revision
remains available through the user-scoped history API and CLI.

Repeated equivalent claims or normalized duplicate durable statements consolidate
into one canonical record. It has a faithful summary, evidence IDs, supporting
session IDs, and an observation range. Original observations remain stored as
consolidated records. Repetition refreshes retention/retrieval recency without
artificially raising importance/confidence. Procedural duplicates can consolidate
verbatim; tasks and episodic occurrences retain separate identities. Returning
to a previous value after a conflicting change starts a new evidence episode.

Default list/search omit superseded/consolidated rows. `--include-history` reveals
them; expiry and resolved-task flags remain separate. Search indexes summary text
alongside original content, and session filters recognize supporting sessions.
Bulk `reconcile` works on older/raw-saved records and appends only changed rows.
Repeated bulk runs and duplicate event-ID retries do not add redundant revisions.

## Examples verified by automated scenarios

| Sequence | Result |
| --- | --- |
| Live in Chennai → live in Bengaluru | Bengaluru current; Chennai superseded with a replacement link. |
| Live in Chennai → reside in Chennai | One summary, “User lives in Chennai.”, and two evidence IDs. |
| Like tea + like coffee | Both remain active. |
| Like coffee → no longer like coffee | New negative preference becomes current. |
| Prefer short answers → dislike short answers | Negative response-length preference replaces the positive one. |
| Chennai → Mumbai → Chennai | Three preserved observations; the final Chennai assertion starts a new group. |
| Strong older source + weak newer contradictory source | Strong source stays current; the losing observation remains inspectable. |
| Explicit restoration + equal-quality older late arrival | Confirmation remains current; genuinely newer observations may replace it. |

## Verification

`python3 -m unittest -q`: **83 tests run; 79 passed and 4 optional ML tests skipped**.
The 26 new scenario tests cover conflicts, compatible preferences, negative
facts, named-entity/user separation, uncertainty and compound clauses, all policy
criteria, evidence integrity, out-of-order data, duplicate retries, retention,
manual confirmation, legacy records, bulk idempotency, and CLI integration.

`python3 -m tests.evaluation.evaluate`: existing **48/48 category, tier, and proposed
importance-range checks pass**, macro F1 **1.0**. These remain developer examples,
not an independent estimate of conversational accuracy. No labels were changed.

## Scope

The extractor uses documented deterministic English patterns. It does not infer
preferences from repeated mentions, generate novel facts from repeated events,
resolve arbitrary prose, or recognize city/person aliases. Unsupported statements
stay stored without inferred conflicts. Exact consolidation preserves complex
wording rather than attempting an unsupported abstraction. Unknown/conditional,
reported, historical, and multi-assertion text cannot automatically replace a
supported fact. Use supported independent-clause splitting where appropriate.

History is retained in the existing append-only JSONL log. These changes reduce
duplicate retrieval, not physical disk usage. Transactional storage, writer
locking, compaction, and authentication remain separate work.

See [README usage](../README.md#conflicts-current-memories-and-consolidation).
