export const meta = {
  name: 'f6-abstention-rework',
  description: 'Rework the flawed abstention design: diversify unanswerable questions, build two candidate designs, verify adversarially, judge',
  phases: [
    { title: 'Author', detail: 'diverse unanswerable test queries + dev-only abstention calibration set' },
    { title: 'Design', detail: 'corrected slot rule vs local answerability model' },
    { title: 'Verify', detail: 'adversarial reproduction and external checks' },
    { title: 'Judge', detail: 'shipping decision' },
  ],
}

const R = '/Users/josephremingstonl/Downloads/code/capstone'
const PY = R + '/.venv/bin/python'
const SP = '/private/tmp/claude-501/-Users-josephremingstonl-Downloads-code-capstone/2d91e859-4acd-4fbf-99c1-146f2e2aa197/scratchpad'
const W = SP + '/f6rework'
const SNAP = W + '/snapshot'

const COMMON = `
PROJECT: CogniMem, a local-first memory backend (JSONL revision log, SQLite vector + temporal graph indexes, keyword/semantic/graph hybrid retrieval with RRF, local cross-encoder reranker Xenova/ms-marco-MiniLM-L-6-v2 returning raw logits, extractive or Gemini answer generation in main/retrieval/generation.py).
Python: ${PY}. Run modules with PYTHONPATH=. from inside your own repo copy.
FROZEN SNAPSHOT of the repo to copy from: ${SNAP} (its data/ is a symlink to the main repo data; treat data/ as READ-ONLY except that you may write new files under your own scratch dir).
HARD RULES:
- Never edit anything under ${R}. Work only in your own scratch directory under ${W}. Create your copy with: rsync -a --exclude data ${SNAP}/ <yourdir>/repo/ && ln -s ${R}/data <yourdir>/repo/data
- No hosted/Gemini/network LLM calls. Never read or print .env. Extractive generator only.
- Model downloads (if any) go to a cache under your scratch dir (set HF_HOME / cache_dir there), never into ${R}/data.
- The machine is shared and hot: run heavy jobs one at a time, never more than one LoCoMo/LongMemEval pass concurrently in your own work.
- Benchmark: evaluation/datasets/gold_queries.json is compiled from evaluation/datasets/seeds/p01..p12_*.json by "python -m evaluation.build_dataset" (use --check to verify). One shared corpus of 456 memories under user 'gold-user'; persona = source_metadata.speaker; dev split = personas p01-p03 (90 queries), test = p04-p12 (270). Harness: "python -m evaluation.ablation --split dev|test --candidate-limit 20 --abstain-threshold T --output-dir DIR" (evaluation/systems.answer_extractive forwards abstain_threshold to generation.answer only when it is not None). Metrics in evaluation/metrics.py (answerability: answerable acceptance, unanswerable refusal, balanced accuracy).
`

const BACKGROUND = `
BACKGROUND (why this rework exists):
The previous abstention design ("F6") is at ${SP}/designs/F6_design.json and its adversarial verdict at ${SP}/designs/F6_verdict.json (verdict: FLAWED). Read both in full. In short: F6 abstained when a "slot" asked by the question (name / quantity / frequency / occupation) was not filled in a confident source, plus a logit floor tau=2e-4. It scored 86.3% balanced on the benchmark test split, but the verifier showed (1) 25 of the 30 benchmark unanswerables were just two templates (sibling's job, pet's name) that the slot patterns encode, so a "refuse every slot-pattern question" prior beats it on dev (97.6%); (2) on external data it did not generalise: LongMemEval balanced 54.0% (answerable acceptance 84.7%), LoCoMo balanced 49.1% (category-5 refusal 5/446), worse than not abstaining; (3) bugs: occupation regex dead alternatives, quantity/frequency slots abstain on counting/aggregation/date-arithmetic questions, name slot fires on yes/no and name-given questions, ASCII-only capitalisation, the 'Summary:' line counted as names, tau floor costs answerables and catches almost nothing. The F6 designer's scratch (repo with the rule, scripts collect.py/replay.py) is at ${SP}/design/abstention/ and the verifier's external runners (external.py for LongMemEval and LoCoMo, ext_summary.py) are at ${SP}/design/verify-abstention/ — reuse and adapt them.
The F6 design also contains a REFERENCE diff for main/retrieval/rag.py that adds context['scores'][memory_id] = {rank, reranker (raw logit), fusion, cosine, channels} and context['reranker'] = {status, model, error_type}. That interface WILL ship (exactly those names), so build on it: apply that rag.py reference diff in your copy.
`

const AUTHOR_PROMPT = `${COMMON}
YOUR TASK (author, phase 1 of an abstention rework). You are deliberately NOT told how any abstention rule works; do NOT open ${SP}/designs/ or ${SP}/design/ — author blind.

Directory: ${W}/author. Make your repo copy there as described.

A) Rewrite the 30 benchmark "unanswerable" queries so they are diverse. Today they are almost all "What does <persona>'s sibling do for work?" and "What is the name of <persona>'s pet?" — that is a template artefact and must go. Requirements:
- Keep exactly the same number of unanswerable queries per persona (p01:2, p02:3, p03:2, p04:3, p05:2, p06:3, p07:2, p08:3, p09:2, p10:3, p11:2, p12:3) and keep each query's position in the seed's queries list, so query ids stay stable.
- Across the 30, spread attribute types: at most 3 sibling-occupation and at most 3 pet-name questions in total. Use places/neighbourhoods, schools/universities, scores/grades, prices/salaries/amounts, dates/when, counts/how-many, durations, names of people other than pets, opinions/preferences never stated, yes/no existence ("Has X ever ..."), reasons/why, which-option questions, advice-seeking about a never-mentioned topic, and 2-3 cross-persona traps (asks persona A about something only persona B said).
- Each must be distractor-adjacent: the corpus mentions the entity/topic, but the asked attribute is never stated for that persona anywhere in the corpus. Verify by reading ALL that persona's memories (and grep the whole corpus for the trap ones). Vary phrasing (not all wh- questions; some indirect: "Remind me which ...", "Do you know ...").
- Each query names its persona, is >= 6 words, does not copy memory wording, has no gold, no supporting, no as_of. Leave answer/aliases empty as the existing unanswerables do (check the existing seed format).
- Rebuild with "python -m evaluation.build_dataset" in your copy and make sure "--check" and its validators pass, and run "python -m pytest -q tests/test_evaluation_dataset.py" there.

B) Write a DEV-ONLY abstention calibration set for dev personas p01-p03 ONLY (never p04-p12): ${W}/author/abstention_dev.json, a JSON list of objects {"id","persona","query","answerable":bool,"gold":{"<memory id>":3,...},"answer":"","aliases":[]}. About 45 unanswerable + 45 answerable. Memory ids are the "id" values in gold_queries.json (form "p01-<key>"). Unanswerable: same diversity rules as (A), verified against the corpus. Answerable: diverse forms, deliberately including the hard cases for abstention rules: name questions (wh- and yes/no "Is X called ...?"), how many / how much / how long / how often questions whose answer is stated, counting/aggregation questions over several memories (gold = all needed memories, answer = the count), date-arithmetic/when questions, occupation questions with plain occupation words (nurse, chef, actuary...), advice/preference questions, third-person questions about first-person memories. For answerable rows, the answer (or an alias) must appear verbatim (case-insensitive) in at least one gold memory's content unless it is a computed count/duration; set "computed": true for those. Validate programmatically (write a small checker script, run it, include its output).

Output: return a JSON object per the schema: the seed file paths you changed, the list of the 30 new unanswerable queries with persona and attribute type, the path of abstention_dev.json with counts, the checker and build/test outputs, and caveats. Do not edit anything outside ${W}/author.`

const AUTHOR_SCHEMA = {
  type: 'object',
  properties: {
    changed_seed_files: { type: 'array', items: { type: 'string' } },
    new_unanswerable: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, persona: { type: 'string' }, query: { type: 'string' }, attribute_type: { type: 'string' } }, required: ['persona', 'query', 'attribute_type'] } },
    abstention_dev_path: { type: 'string' },
    abstention_dev_counts: { type: 'string' },
    validation_output: { type: 'string' },
    caveats: { type: 'string' },
  },
  required: ['changed_seed_files', 'new_unanswerable', 'abstention_dev_path', 'abstention_dev_counts', 'validation_output', 'caveats'],
}

const DESIGNS = [
  {
    key: 'slot-rule-v2',
    brief: `DESIGN A — corrected lexical answer-type rule ("slot rule v2"). Start from the F6 rule in ${SP}/design/abstention/repo/main/retrieval/generation.py and fix every issue in the F6 verdict: (a) whole-word occupation lexicon instead of the \\w+ suffix trick, (b) name slot only for wh-questions asking for an unknown name (not yes/no, not when the name is already in the query), (c) Unicode-aware capitalisation and strip speaker/session_date/Summary: lines, (d) do not abstain on counting/aggregation/date-arithmetic questions (how many times / days ago / in total / between ...), or drop quantity/frequency slots if they cannot be made safe, (e) occupation question pattern only when it asks about the person's job, (f) drop the tau floor unless held-out-free evidence supports it. Consider whether the evidence set (best source / sources with logit >= 0) is right. Keep it a pure function of the context; no new dependencies.`,
  },
  {
    key: 'answerability-model',
    brief: `DESIGN B — local answerability model. Find a small, permissively licensed (Apache-2.0/MIT/CC-BY; NOT non-commercial) model with an ONNX export that runs with onnxruntime + tokenizers + huggingface_hub only (all installed; transformers is NOT installed and must not become a runtime dependency; torch must not be needed at runtime). Candidates to investigate: SQuAD2-style extractive QA models with a no-answer head (e.g. ONNX exports of deepset/roberta-base-squad2, deepset/tinyroberta-squad2, deepset/minilm-uncased-squad2, distilbert squad2 variants under Xenova/ or onnx-community/), or NLI cross-encoders (e.g. ONNX exports of cross-encoder/nli-deberta-v3-xsmall/small) used as "does this evidence answer the question". Use the HF hub API to find real repos and files; verify the licence on the model card. Score the top context sources (the ones the extractive generator would answer from) and abstain when the best answerability score is below a threshold tuned on the dev calibration data. Measure added latency (budget: <= 150 ms P50 added per ask on top-5 sources; report P50/P95). Implement as an optional component (new module, e.g. main/retrieval/answerability.py) loaded lazily, with a clear opt-in parameter on generation.answer; the library must work unchanged when the model is unavailable. If no candidate meets licence/latency/quality, say so with numbers.`,
  },
]

const DESIGN_SCHEMA = {
  type: 'object',
  properties: {
    fix_id: { type: 'string' },
    summary: { type: 'string' },
    root_cause: { type: 'string' },
    proposed_changes: { type: 'array', items: { type: 'object', properties: { file: { type: 'string' }, description: { type: 'string' }, unified_diff: { type: 'string' } }, required: ['file', 'description', 'unified_diff'] } },
    full_patch_path: { type: 'string' },
    experiment_scripts: { type: 'array', items: { type: 'string' } },
    dev_results: { type: 'string' },
    final_heldout_check: { type: 'string' },
    numbers_table: { type: 'string' },
    latency: { type: 'string' },
    tests: { type: 'string' },
    risks: { type: 'string' },
    confidence: { type: 'string' },
  },
  required: ['fix_id', 'summary', 'proposed_changes', 'full_patch_path', 'dev_results', 'final_heldout_check', 'numbers_table', 'latency', 'tests', 'risks', 'confidence'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    fix_id: { type: 'string' },
    verdict: { type: 'string', enum: ['sound', 'partially_sound', 'flawed'] },
    reproduced: { type: 'boolean' },
    reproduced_numbers: { type: 'string' },
    issues: { type: 'array', items: { type: 'object', properties: { severity: { type: 'string', enum: ['blocking', 'major', 'minor'] }, issue: { type: 'string' }, evidence: { type: 'string' } }, required: ['severity', 'issue', 'evidence'] } },
    recommended_adjustments: { type: 'string' },
    ship_recommendation: { type: 'string', enum: ['default_on', 'opt_in_only', 'do_not_ship'] },
  },
  required: ['fix_id', 'verdict', 'reproduced', 'reproduced_numbers', 'issues', 'recommended_adjustments', 'ship_recommendation'],
}

const EVAL_PROTOCOL = `
EVALUATION PROTOCOL (mandatory, identical for both designs):
- Tuning data: ONLY the dev calibration set and the benchmark dev split (both with the author's new queries). Never tune on the benchmark test split, LongMemEval or LoCoMo.
- Baselines to report next to your design on every set: (1) no abstention (balanced 50%), (2) the original F6 rule, (3) a "template prior" that refuses every question matching F6's slot question patterns with no evidence check.
- Freeze the design (record sha256 of changed files), THEN compute held-out numbers ONCE: benchmark test split (hybrid_full, pool 20, via evaluation.ablation with the author's new unanswerables), LongMemEval oracle (user turns as memories; *_abs questions = should refuse, others answerable), LoCoMo (category 5 = should refuse, 1-4 answerable). Adapt ${SP}/design/verify-abstention/external.py for the external sets (it writes stores/indexes only under scratch).
- Report per set: answerable acceptance, unanswerable refusal, balanced accuracy, abstention precision, grounded correctness where available, and the counts.
- ADOPTION RULE you are judged against: default-on only if, on all three held-out sets, answerable acceptance >= 95% (benchmark) / >= 90% (externals) AND balanced accuracy beats no-abstention by >= 3 points on BOTH externals. Otherwise it can ship only as opt-in. A design that is worse than no-abstention on any external set should be reported as such, honestly.
- Run the full test suite in your copy (python -m pytest -q -p no:cacheprovider); note that tests/test_conversation_task_extensions.py has a known wall-clock flake under load. Add unit tests for your change.
- Write a single full patch file (git diff format, relative to the snapshot) and give its path.`

phase('Author')
const authored = await agent(AUTHOR_PROMPT, { label: 'author:unanswerable+calibration', phase: 'Author', schema: AUTHOR_SCHEMA })
if (!authored) { log('Author agent failed; stopping.'); return { error: 'author failed' } }
log(`Author done: ${authored.new_unanswerable.length} new unanswerable queries; dev set ${authored.abstention_dev_counts}`)

const AUTHORED = `
AUTHOR OUTPUT (phase 1, use it): new seeds and calibration data are in ${W}/author/repo (changed seed files: ${authored.changed_seed_files.join(', ')}; rebuilt evaluation/datasets/gold_queries.json there) and the dev-only calibration set at ${authored.abstention_dev_path} (${authored.abstention_dev_counts}). After making your repo copy from the snapshot, copy over evaluation/datasets/seeds/ and evaluation/datasets/gold_queries.json from ${W}/author/repo and confirm "python -m evaluation.build_dataset --check" passes. Author caveats: ${authored.caveats}`

const results = await pipeline(
  DESIGNS,
  d => agent(`${COMMON}${BACKGROUND}${AUTHORED}
YOUR TASK: ${d.brief}
Directory: ${W}/${d.key}. ${EVAL_PROTOCOL}
Return the design per the schema with fix_id "F6-${d.key}". The numbers_table must be a markdown table with rows = {no abstention, F6 original, template prior, this design} and columns = acceptance/refusal/balanced for dev-calibration, benchmark-dev, benchmark-test, LongMemEval, LoCoMo.`,
    { label: `design:${d.key}`, phase: 'Design', schema: DESIGN_SCHEMA }),
  (design, d) => design && agent(`${COMMON}${BACKGROUND}${AUTHORED}
YOUR TASK: adversarially VERIFY the abstention design below. Default to skepticism: try to show that its gains are artefacts (template matching, tuning leakage into held-out sets, cherry-picked thresholds, broken baselines), that numbers do not reproduce, that it hurts answerable questions, or that its code is wrong. Directory: ${W}/verify-${d.key} (fresh copy from the snapshot, then copy in the author's seeds/gold as described, then apply the design's patch at ${design.full_patch_path}).
Reproduce its dev and held-out numbers (benchmark test, LongMemEval, LoCoMo) with its frozen files, check sha256s, run the full test suite, write >= 40 of your own adversarial probe questions (answerable and unanswerable, varied phrasing) over a small store, and review the code for bugs. Judge it against the ADOPTION RULE in the protocol. ${EVAL_PROTOCOL}
DESIGN UNDER REVIEW:
${JSON.stringify(design).slice(0, 60000)}`,
    { label: `verify:${d.key}`, phase: 'Verify', schema: VERDICT_SCHEMA }).then(v => ({ design, verdict: v })),
)

phase('Judge')
const valid = results.filter(Boolean)
const judge = await agent(`${COMMON}
YOU ARE THE JUDGE for an abstention rework. Two candidate designs were built and each was adversarially verified. Decide what ships in CogniMem's extractive path.
Options: ship one design default-on; ship one (or both) opt-in only (library default abstain off, CLI/harness flag); ship nothing. Apply the ADOPTION RULE strictly: default-on only if on all three held-out sets (benchmark test, LongMemEval, LoCoMo) answerable acceptance >= 95% (benchmark) / >= 90% (externals) AND balanced accuracy beats no-abstention by >= 3 points on BOTH externals. Weigh the verifiers' reproduced numbers over the designers' claims. You may read the scratch dirs under ${W} to check numbers. List the exact adjustments required before merge (from the verdicts), the honest wording for reports (what the benchmark numbers do and do not show), and the final numbers table (reproduced numbers).
CANDIDATES:
${JSON.stringify(valid).slice(0, 150000)}`,
  { label: 'judge', phase: 'Judge', schema: {
    type: 'object',
    properties: {
      decision: { type: 'string', enum: ['default_on', 'opt_in_only', 'ship_nothing'] },
      chosen_fix_id: { type: 'string' },
      patch_path: { type: 'string' },
      rationale: { type: 'string' },
      required_adjustments: { type: 'string' },
      report_wording: { type: 'string' },
      final_numbers_table: { type: 'string' },
    },
    required: ['decision', 'chosen_fix_id', 'patch_path', 'rationale', 'required_adjustments', 'report_wording', 'final_numbers_table'],
  } })

return { authored, results: valid, judge }
