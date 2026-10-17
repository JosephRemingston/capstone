export const meta = {
  name: 'cognimem-fix-design',
  description: 'Design and measure four CogniMem metric fixes on dev data in scratch copies, then adversarially verify each',
  phases: [
    { title: 'Design', detail: 'four independent designers, each experimenting on a scratch copy with dev-only tuning' },
    { title: 'Verify', detail: 'one adversarial verifier per design re-runs experiments and hunts for leakage or breakage' },
  ],
}

const REPO = '/Users/josephremingstonl/Downloads/code/capstone'
const SCRATCH = '/private/tmp/claude-501/-Users-josephremingstonl-Downloads-code-capstone/2d91e859-4acd-4fbf-99c1-146f2e2aa197/scratchpad/design'

const CONTEXT = `
You are working on CogniMem, a Python 3.13 local-first memory backend for AI assistants, at ${REPO}.
Use the repo virtualenv: ${REPO}/.venv/bin/python (fastembed, torch, xgboost, pytest installed; embedding and reranker models are already cached under ${REPO}/data/rag/models).

HARD RULES
- Do NOT modify, create, or delete anything inside ${REPO}. Read it freely.
- Work only in your own scratch directory (given below). To experiment with code changes, make a scratch copy:
    mkdir -p <your-scratch> && rsync -a --exclude .venv --exclude data --exclude docs --exclude .git ${REPO}/ <your-scratch>/repo/ && ln -s ${REPO}/data <your-scratch>/repo/data
  then edit files in <your-scratch>/repo and run commands from <your-scratch>/repo with ${REPO}/.venv/bin/python (prefix PYTHONPATH=. if needed).
- Never write into ${REPO}/data (it is symlinked read-only in spirit). Write outputs to your scratch dir.
- Tune only on development data. Test/held-out numbers may be computed ONCE at the end, after the design is frozen, and must be labelled as such. Never iterate on test numbers.
- Other agents are running experiments concurrently on this machine, so latency numbers are indicative only; accuracy numbers are what matter.
- Keep the existing unit tests green: from your scratch copy run "${REPO}/.venv/bin/python -m pytest -q -p no:cacheprovider" (baseline: all 150-ish tests pass). Report any test you had to change and why.

SYSTEM FACTS (verified)
- Retrieval: main/retrieval/rag.py MemoryRAG.search() fuses keyword (weight .3), semantic (.5), graph (.2) channels with RRF k=60, keeps a candidate pool (default max(20, 4*limit), cap 200; limits 51-100 crash without candidate_limit), optionally reranks with the local cross-encoder Xenova/ms-marco-MiniLM-L-6-v2 (main/retrieval/reranker.py, raw logits), returns hits with memory_id, text, score(=fusion), fusion_score, reranker_score, reranker{status}, signals{channel:{rank,contribution}}, record (dict), paths (graph paths containing this memory). Graph paths are kept in hybrid mode only if every edge predicate is cued by query words (graph_predicates()). context() builds the answer context (greedy coverage, tie-break on fusion score). main/retrieval/generation.py answer(generator='extractive') abstains only when context['sources'] is empty.
- Vector index: main/storage/indexes.py VectorIndex (exact scan) and ApproximateVectorIndex (LSH 8 tables x 10 bits; multi-probe breaks after radius 0 when any bucket matches; no fallback; _fingerprint hashes only chunk rows). MemoryRAG defaults ann_enabled=True, ann_exact_threshold=256 and rebuilds the index object on every 'vectors' property access.
- Graph: main/graph/temporal.py; explicit source_metadata entities/relations with valid_from/valid_to become edges; relations() filters by valid time (as_of) so closed relations are invisible to current queries; self node keyed by source_metadata.speaker.
- Classification/lifecycle: main/domain/classification.py MemoryClassifier.classify (non-legacy path): uncertain() from main/domain/text_rules.py returns TEMPORARY for any text containing '?', quotes, said/says/asked/maybe/could/might (except polite reminder requests); final fallback returns TEMPORARY. main/domain/lifecycle.py: TEMPORARY or score<0.25 -> working tier (1 hour expiry); short_term 14 days; long_term no expiry. The legacy=True path is frozen feature preprocessing for the trained XGBoost/neural artifacts and must NOT change.

EVALUATION HARNESS (new, in ${REPO}/evaluation)
- evaluation/datasets/gold_queries.json: Claude-authored benchmark, 12 personas, 456 memories (one shared user 'gold-user'; persona = source_metadata.speaker), 360 queries, 30 per type (semantic, episodic, temporal, historical, multi_hop, entity_relationship, conflict, preference, procedural, recency, negative_absence, unanswerable). Split: personas p01-p03 = dev (90 queries), p04-p12 = test (270). Query fields: query_id, query, query_type, persona_id, split, relevant_memory_ids, graded_relevance {id: 1..3}, supporting_memory_ids, expected_answer, answer_aliases, unanswerable, as_of (historical only). Temporal/conflict/recency queries grade the current memory 3 and older ones 1; supporting = current only. "conflict" queries that ask for the ORIGINAL value (contain words like 'before', 'originally', 'first') grade the old memory 3.
- python -m evaluation.ablation --split dev --output-dir <dir> [--candidate-limit N] [--arms a,b] [--abstain-threshold T]: arms vector, vector_reranker, vector_graph, vector_graph_reranker, hybrid_full (keyword+semantic+graph+rerank = shipped path). Writes results.json with summary, by_query_type, rows (per query per arm: predicted top-20 ids, recall_at_5/10, hit_at_5, mrr, pool_hit, correct_grounded, correct_top1, abstain, top_reranker_score, top_fusion_score). Programmatic: evaluation.ablation.run(dataset=..., split='dev', protocol=evaluation.systems.Protocol(candidate_limit=20), arms=[...]). ~60 s for dev with real models.
- python -m evaluation.locomo [--candidate-limit N] [--ann] [--limit-conversations K] --output <file>: external LoCoMo evidence retrieval (data already downloaded at data/evaluation/locomo).
- python -m evaluation.retention [--limit N] --output <file>: LongMemEval oracle replay through MemoryCore + store.ingest with original timestamps; metric = fraction of has_answer turns still visible at question date (memory_retention) vs keep-everything 100%; also visible_fraction. evaluation.retention.replay(row) returns per-example dict. Data at data/evaluation/longmemeval/oracle.json.
- Saved BASELINE reports (pool 20, exact search) in ${REPO}/docs/reports/: ablation/ (test), ablation_dev/ (dev), ablation_ann/, locomo_retrieval.json, locomo_retrieval_ann.json, longmemeval_retention.json, conversational_evaluation.json.

BASELINE NUMBERS
- Fresh benchmark test split, hybrid_full: Recall@5 93.0%, pool hit 99.6%, grounded correctness 89.3%, unanswerable refusal 0% (balanced answerability 50%). correct_top1 by type (hybrid_full): temporal 27.3%, recency 43.5%, conflict 54.5%, historical 52.2%, multi_hop 100%. Recall@5 multi_hop: vector_graph 90.9% but vector_graph_reranker 65.9% (reranker pushes the first-hop partner, e.g. the job memory linking persona->company, out of top-5). For temporal questions the current fact ranks #1 only 6/22 times in hybrid_full, while vector_graph (no reranker) gets temporal top-1 81.8%.
- Unanswerable vs answerable top reranker sigmoid (test, hybrid_full): unanswerable p10 .13 / p50 .83 / p90 .99; answerable p10 .35 / p50 .98 / p90 .999. Unanswerable questions are distractor-adjacent by design (e.g. "What does Priya's brother do for a living?" when a memory mentions the brother but not his job).
- LoCoMo hybrid_reranked (exact, pool 20): Recall@5 55.8%, Hit@5 61.1%, pool hit 69.6%; earlier stage audit: pool 50 -> 84.3% hit, pool 100 -> 91.9% hit; reranker recovers ~88% of what reaches it. With ANN (shipped default): Recall@5 26.9%, pool hit 33.3%.
- LongMemEval retention: 21.0% of answer turns visible at question date (keep-everything 100%); mean visible fraction 18.2%. Diagnosis on 60 examples: 145 of 175 gold user turns expired from the WORKING tier; 150/175 were classified TEMPORARY.
- Classification on 60 Claude-authored cases (evaluation/datasets/conversational_cases.jsonl, held out, never tune on them): category accuracy 50%, tier accuracy 55%.
`

const DESIGN_SCHEMA = {
  type: 'object',
  properties: {
    fix_id: { type: 'string' },
    summary: { type: 'string' },
    root_cause: { type: 'string' },
    proposed_changes: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          file: { type: 'string' },
          description: { type: 'string' },
          unified_diff: { type: 'string', description: 'unified diff against the current repo file, or full replacement code for the function' },
        },
        required: ['file', 'description', 'unified_diff'],
      },
    },
    experiment_scripts: { type: 'array', items: { type: 'string' }, description: 'absolute paths of scripts in your scratch dir that reproduce your numbers' },
    dev_results: { type: 'string', description: 'before/after numbers on development data, per sub-fix, with exact commands' },
    final_heldout_check: { type: 'string', description: 'numbers computed once on test/held-out after freezing, clearly labelled; or "not run"' },
    tests: { type: 'string', description: 'pytest result on the scratch copy with the change applied, and any tests changed with justification' },
    risks: { type: 'string' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
  required: ['fix_id', 'summary', 'root_cause', 'proposed_changes', 'experiment_scripts', 'dev_results', 'final_heldout_check', 'tests', 'risks', 'confidence'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    fix_id: { type: 'string' },
    verdict: { type: 'string', enum: ['sound', 'partially_sound', 'flawed'] },
    reproduced: { type: 'boolean' },
    reproduced_numbers: { type: 'string' },
    issues: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['blocking', 'major', 'minor'] },
          issue: { type: 'string' },
          evidence: { type: 'string' },
        },
        required: ['severity', 'issue', 'evidence'],
      },
    },
    recommended_adjustments: { type: 'string' },
  },
  required: ['fix_id', 'verdict', 'reproduced', 'reproduced_numbers', 'issues', 'recommended_adjustments'],
}

const DESIGNS = [
  {
    id: 'F3-retention',
    label: 'retention-classifier',
    prompt: `Your scratch dir: ${SCRATCH}/retention
FIX F3: LongMemEval retention is 21% because informative user turns are classified TEMPORARY (working tier, 1-hour expiry).
Design and measure a classifier/lifecycle change in the NON-legacy path so that substantive user statements (first-person information, plans, events, facts embedded in questions or requests to the assistant) are not discarded, while genuinely phatic messages (greetings, thanks, acknowledgements, 'ok', 'lol', 'never mind', 'can you say that again?') stay TEMPORARY/working. Assistant-role turns may keep current behaviour unless you show a principled reason.
Development data: LongMemEval examples whose question_id satisfies int(hashlib.sha256(qid.encode()).hexdigest(), 16) % 5 == 0 (about 20%). Tune ONLY on those. Held-out = all other chronology-valid examples; compute held-out memory_retention and visible_fraction ONCE after freezing. Also report (do not tune on) the 60 held-out classification cases via python -m evaluation.classification --output <scratch file>, before and after.
Report retention AND visible_fraction (storage efficiency): a policy that keeps everything is not a fix. Prefer the smallest principled change (for example: questions/hedges should not force TEMPORARY when the text contains first-person informational content; the final fallback for substantive multi-word user text should be EPISODIC/short_term or SEMANTIC rather than TEMPORARY; reconsider whether 14-day short_term expiry or the working threshold drives losses at question dates months later, and whether user-stated facts should be long-term). Quantify each sub-change separately on dev.
Check the legacy=True path is byte-for-byte unchanged in behaviour (the ML artifacts depend on it): run tests/test_ml_importance.py and tests/test_neural_importance.py.
Many existing tests pin classifier behaviour (tests/test_memory_core.py, tests/test_conversational_rules.py, tests/test_conversation_task_extensions.py, tests/test_memory_lifecycle_updates.py, tests/test_memory_reconciliation.py, tests/test_quality_improvements.py, tests/test_cleanup.py). Keep them green; if one encodes the bug itself, say exactly which assertion and why it should change.`,
  },
  {
    id: 'F4-rerank-graph-time',
    label: 'rerank-graph-time',
    prompt: `Your scratch dir: ${SCRATCH}/rerank
FIX F4: the cross-encoder reranker ignores time and graph structure. Evidence: on the fresh benchmark, multi-hop Recall@5 drops from 90.9% (vector_graph) to 65.9% (vector_graph_reranker) because the reranker pushes the first-hop partner of a graph path out of the top 5; temporal correct_top1 drops from 81.8% to 27.3% because superseded statements (memories whose explicit relations are closed with valid_to in the past) outrank current ones; recency/conflict top-1 are 43.5%/54.5% because an older statement on the same topic by the same speaker outranks the later correction or update.
Design changes inside MemoryRAG.search() (after reranking, hybrid mode) such as: (a) evidence-chain completion: when a top-ranked memory belongs to a cued graph path returned by the graph channel, pull the path's other source memories up to sit right after it, within the limit; (b) validity demotion: for non-historical queries (as_of is None), demote memories whose explicit graph relations are ALL closed before now, using record['source_metadata']['relations'] or the graph edges; (c) newer-statement preference: among top candidates from the same speaker that are near-duplicates topically (e.g. dense cosine between the two memory texts above a threshold, or reranker scores within a margin), order the newer memory first unless the query carries an explicit history cue (before, originally, first, previously, used to, earlier, back in, at first, 'in <year>'), in which case keep or prefer the older one. You may propose a different, better mechanism if you can justify it.
Measure each sub-change separately and combined on the DEV split (arms vector_graph_reranker and hybrid_full): Recall@5, MRR, correct_top1 and correct_grounded overall and per query type. Implement by editing main/retrieval/rag.py in your scratch copy (keep the existing signature and result fields; add new result metadata only if useful). Then check LoCoMo for regressions with python -m evaluation.locomo --limit-conversations 3 --modes hybrid_fusion,hybrid_reranked --output <scratch file>, comparing against the same command on an unmodified scratch copy (LoCoMo has no explicit relations, so (a)/(b) should be inert; (c) could move numbers). After freezing, compute the test-split numbers once.
Watch for regressions on historical queries (they pass as_of), on 'conflict' queries asking for the original value, and on episodic questions about older events.`,
  },
  {
    id: 'F6-abstention',
    label: 'abstention',
    prompt: `Your scratch dir: ${SCRATCH}/abstention
FIX F6: the extractive answer path never abstains (unanswerable refusal 0%, balanced answerability 50%). Design an abstention rule for generation.answer(generator='extractive') that maximizes balanced answerability accuracy on the DEV split subject to answerable acceptance >= 95% on dev.
Start from the saved dev rows in ${REPO}/docs/reports/ablation_dev/results.json (each row has top_reranker_score, top_fusion_score, abstain, unanswerable, arm). The primary candidate is a threshold tau on sigmoid(max reranker logit over the context sources). Evaluate alternatives and combinations that could separate distractor-adjacent unanswerable questions from answerable ones, for example: margin between the top two reranker scores; whether the top source mentions every salient query entity/attribute word (query term coverage after removing the persona name and stopwords); answer-type checks (a 'who/where/when/which city/how many' question needs a person/place/date/number token in the top source); requiring agreement between dense and reranker signals. You may also assess, without installing anything, whether a local extractive-QA model with no-answer support (e.g. a SQuAD2-style model) would be worth adding: check whether transformers is importable and estimate cost, but do not install packages.
Also re-run the dev ablation with --candidate-limit 100 (a planned separate fix raises the pool from 20 to 100) and report whether your tau/rule transfers or needs re-tuning at pool 100.
Specify exactly how the rule plugs in: generation.answer(context, generator, abstain_threshold=...) and what context fields it needs (a separate planned change will add context['scores'] = {memory_id: {rank, reranker, fusion, cosine, channels}} and context['reranker'] status to MemoryRAG.context(); design against that interface, and show the code). The rule must be a no-op when no reranker scores are available (reranker disabled or fell back).
After freezing tau/rule on dev, compute the test-split answerable-accept, unanswerable-refusal and balanced accuracy ONCE (you can recompute from ${REPO}/docs/reports/ablation/results.json rows for pool 20, and by running the test ablation at pool 100 if needed).`,
  },
  {
    id: 'F1F2-ann-pool',
    label: 'ann-and-pool',
    prompt: `Your scratch dir: ${SCRATCH}/annpool
FIXES F1+F2: (F1) the approximate vector index is the shipped default for any store with >=256 memories and loses most candidates (LoCoMo hybrid_reranked Recall@5 55.8% exact vs 26.9% ANN; fresh benchmark 93.0% vs 57.5%); (F2) the hybrid candidate pool of 20 contains gold evidence only 69.6% of the time on LoCoMo.
Design concrete code: (F1) MemoryRAG defaults ann_enabled=False and a high ann_exact_threshold (e.g. 5000); cache the vector index object per (approximate, embedder.model_id) and avoid calling store.list() when ANN is off; ApproximateVectorIndex.search keeps probing radii 0->1->2 until it has at least max(limit, min_candidates) candidates and otherwise falls back to exact search, recording diagnostics; include tables/bits/seed in _fingerprint; replace the per-row Python dot product in VectorIndex.search with a numpy matmul that preserves ordering; api/config.py: COGNIMEM_ANN_ENABLED default false, threshold default 5000, reject threshold < 1 at config time (today threshold 0 silently makes the API app None); CLI global --ann flag in main/__main__.py. (F2) default pool formula min(200, max(100, 4*limit)) that also fixes the limit 51-100 crash, a MemoryRAG(candidate_pool=None) override wired to COGNIMEM_CANDIDATE_POOL, reranker threads configurable (COGNIMEM_MODEL_THREADS, default None = let onnxruntime decide) and reranker documents capped at ~600 characters.
Measure in your scratch copy: an LSH recall guard (seeded synthetic 32-d embedder, 2,000 chunks, 50 queries: ANN top-20 overlap with exact >= 0.95, fallback rate reported); exact-scan numpy vs Python ordering equality; LoCoMo hybrid_reranked Recall@5 / Hit@5 / pool hit / P50 latency at candidate_limit 20, 50, 100 on the first 3 conversations (python -m evaluation.locomo --limit-conversations 3 --modes hybrid_fusion,hybrid_reranked --candidate-limit N --output ...), plus the shipped-default path (no --candidate-limit, i.e. --candidate-limit default after your pool change) and the --ann path after your ANN fix; reranker latency with threads=2 vs threads=None at pool 100. Write the new unit tests you would add (as code in the diff) and run the full test suite in the scratch copy; list existing tests that need updating (e.g. tests/test_ann.py, tests/test_reranker.py bounds test) with reasons.`,
  },
]

phase('Design')
const results = await pipeline(
  DESIGNS,
  d => agent(`${CONTEXT}\n\n${d.prompt}\n\nReturn your design via the structured output. Put real measured numbers in dev_results with the exact commands you ran; if something could not be measured, say so explicitly.`,
    { label: `design:${d.label}`, phase: 'Design', schema: DESIGN_SCHEMA }),
  (design, d) => design ? agent(`${CONTEXT}

You are an ADVERSARIAL VERIFIER. Your scratch dir: ${SCRATCH}/verify-${d.label}
A designer proposed the following fix. Try hard to refute it. Default to 'flawed' if you cannot reproduce the key numbers.
1. Re-run the designer's experiment scripts (listed below) or re-implement the change from the diffs in a FRESH scratch copy of the repo, and reproduce the headline dev numbers.
2. Check for leakage: did they tune on test/held-out data, or iterate after looking at it? Are dev and held-out actually disjoint?
3. Check the change does what it claims and nothing else: unintended behaviour changes, edge cases (empty results, reranker disabled or fallback, historical queries with as_of, limit 0, other users), API/CLI behaviour, legacy ML path (for classifier changes).
4. Run the full test suite on your copy with the change applied.
5. Look for cheaper or more principled alternatives, and for over-fitting to the Claude-authored benchmark's specific phrasing.

DESIGN UNDER REVIEW (JSON):
${JSON.stringify(design, null, 2)}`,
    { label: `verify:${d.label}`, phase: 'Verify', schema: VERDICT_SCHEMA }).then(v => ({ design, verdict: v })) : null,
)
return results.filter(Boolean)
