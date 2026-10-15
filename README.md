# CogniMem

**A local-first, temporal memory system for personalized AI assistants.** CogniMem turns conversations into inspectable memories, keeps track of changes, and retrieves cited evidence for later answers. It combines rule-based retention, task and conflict management, a temporal knowledge graph, vector search, local reranking, and retrieval-augmented generation (RAG).

For the project objective, practical uses, implementation layers, data flows, and remaining goals, read [What we are building and why](PROJECT_VISION.md).

> **Research question:** How much does temporal and relational graph retrieval improve vector-based conversational memory retrieval and RAG? The controlled experiment compares vector-only, vector + reranker, vector + graph, and vector + graph + reranker while holding the dataset, prompt, answer policy, context size, and evaluation procedure fixed.

## Why build it?

An ordinary chat model has no reliable, structured account of everything a person said before. Saving every message verbatim is insufficient: greetings and obsolete reminders compete with useful facts; repeated statements waste space; newer facts may contradict older ones; and semantic similarity alone may miss time and relationships. “Where did I live before Bengaluru?” requires history, while “Where is the company I work for based?” may require a two-edge evidence path. A personalized answer should also show which memories support it.

CogniMem separates those responsibilities. It decides what to retain, preserves corrections and provenance, represents relationships with valid and recorded time, searches several evidence channels, and constrains answers to cited context. A personal assistant could remember communication preferences; a project copilot could track decisions, collaborators, and changing facts; a task assistant could retain deadlines and recurring work. The integrating application still owns the chat interface, consent, and any calendar or reminder delivery.

## What works today

| Capability | Implementation |
| --- | --- |
| Intake | Sensitive-pattern redaction, optional statement splitting, six rule-based categories, feature extraction, claims, and typed relation metadata. |
| Retention | Importance score = 25 % csv5k-trained ensemble (XGBoost + frozen-BGE deep head) + 75 % heuristic by default, heuristic alone as an opt-out; working, short-term, long-term, and archive tiers. |
| Updates | Task completion/cancellation/rescheduling and recurrence; supported fact/preference conflict resolution; evidence-preserving consolidation and history. |
| Storage | Local JSONL revision log with SQLite vector/graph indexes, cross-process locks, cleanup preview, guarded physical deletion, and compaction. |
| Retrieval | Ranked keyword, semantic/vector, graph, recency, and hybrid search; rank fusion and local cross-encoder reranking. |
| Answers | Diverse bounded context, extractive answers, optional LangChain/Gemini 2.5 Flash, citation checks, and abstention. |
| Interfaces | Python API, CLI, and FastAPI with authentication, user isolation, export/deletion, health, logs, and Prometheus metrics. |

The implementation targets local research and small deployments. [Limits and next steps](#limits-and-next-steps) distinguish working features from broader research claims.

## System architecture

```mermaid
flowchart TB
    U[Conversation or event] --> I[Python / CLI / HTTP API]
    I --> M[Memory Core: redact, split, classify, extract]
    M --> L[Score and lifecycle policy]
    L --> C[Task updates, conflicts, consolidation]
    C --> J[(JSONL authoritative revision log)]
    J --> X[Rebuildable indexes]
    X --> V[(SQLite vectors)]
    X --> G[(SQLite temporal graph)]
    Q[Question] --> F[User, status, time, history filters]
    J --> F
    F --> K[Keyword channel]
    F --> V
    F --> G
    K --> R[Weighted rank fusion]
    V --> R
    G --> R
    R --> P[Candidate pool]
    P --> E[Local cross-encoder]
    E --> T[Top-K evidence]
    T --> B[Diverse context within character budget]
    B --> A[Extractive or Gemini answer]
    A --> Z[Checked citations or abstention]
```

JSONL is the source of truth; vector and graph data are rebuildable indexes. Hybrid channels contribute candidates independently before fusion. The reranker runs locally and falls back to fusion order if it fails. Memory text is sent to Gemini only when a hosted answer is requested; intake, embeddings, graph search, and reranking run locally.

The three main flows are:

1. **Save:** accept a user statement → redact and classify it → compute features, score, and tier → reconcile supported claims/tasks → append a revision to JSONL → synchronize rebuildable indexes on search or explicit indexing.
2. **Update:** accept a new statement or task operation → find the relevant user-owned memory → revise task/conflict/consolidation state → append the new state without losing earlier revisions → refresh index views.
3. **Retrieve and answer:** filter visible memories by user and requested time → collect independent keyword/vector/graph candidates → fuse and optionally rerank → select diverse Top-K evidence → assemble bounded context → return cited extractive or Gemini statements.

### Layer 1: Understand and structure incoming information

`MemoryCore` receives `MemoryInput` with text, user/session IDs, a role, timestamp, and optional metadata. It redacts supported sensitive patterns before scoring or storage. It classifies each statement as `semantic` (durable fact), `episodic` (event), `procedural` (workflow), `preference`, `task`, or `temporary` (transient chat). Optional segmentation creates separate records for clearly independent clauses. It extracts numeric cues such as deadlines, preferences, recency, and category. Supported factual statements become claims; callers can also provide explicit typed entities and relations for graph indexing.

The default importance score is 75 % the rule-based `ImportanceScorer`, which combines deterministic features into a 0–1 score, and 25 % a trained model's importance class ([details](#importance-scoring-default)). `LifecycleManager` uses that score and category-specific rules to assign `working`, `short_term`, `long_term`, or `archive`. Temporary messages normally remain working memory, one-off tasks remain short-term even when important, and durable facts or preferences can become long-term. Working memory normally expires after one hour, short-term after fourteen days, and dated tasks after their deadline plus 24 hours. Long-term records can appear as archive views after inactivity. These are policy decisions, not human-labeled truths.

| Category | Typical input | Why it matters later |
| --- | --- | --- |
| Semantic | “My supervisor is Priya.” | Stable facts and relationships. |
| Episodic | “I submitted the proposal yesterday.” | Event history and temporal questions. |
| Procedural | “First run tests, then deploy.” | Reusable instructions. |
| Preference | “I prefer brief explanations.” | Personalization and constraints. |
| Task | “Remind me to submit the report Friday.” | Action state, deadlines, recurrence. |
| Temporary | “Thanks!” | Immediate context without durable retention. |

The trained part is the csv5k "Ensemble + thresholds": a fit-only XGBoost model on 525 text features blended with three seeds of a small head on the frozen 384-dimensional BGE encoder, then class weights pick low, mid or high importance. Its quarter share is a fixed choice, not a tuned one; the model learned from short synthetic sentences, which is a reason to keep it small. Small talk gets a quarter of its class value, as in the heuristic. It runs locally through XGBoost and ONNX Runtime.

### Layer 2: Manage change, tasks, and retention

When a supported fact conflicts with an older one, CogniMem identifies a current assertion while preserving the older record and its revision history. “I live in Chennai” followed by “I moved to Bengaluru” can therefore support both current and historical questions. Repeated equivalent memories may be consolidated into a canonical summary with supporting memory IDs, sessions, and observation dates. Ambiguous task references are recorded rather than silently changing an unrelated task; callers can provide an exact task ID.

Tasks support due dates, completion, cancellation, rescheduling, and recurring occurrences. Task state and lifecycle tier are separate fields. Normal retrieval hides expired and resolved items; history controls expose them for inspection. Cleanup previews by default. Its balanced policy protects useful expired items, legal holds, referenced evidence, and unresolved tasks. Applied cleanup removes eligible expired data, compacts only exact duplicate revisions, and rebuilds indexes.

### Layer 3: Persist memories and temporal relationships

`LocalMemoryStore` writes serialized `MemoryRecord` revisions to `memory_store/memories.jsonl` by default. A record contains its category, tier, score, timestamps, task state, claim and conflict status, evidence IDs, and source metadata. Local operations use cross-process locks. SQLite holds chunks and embeddings, graph nodes and edges, aliases, and index state. An LSH vector path supports larger local collections; smaller collections use exact scans.

The graph stores typed relations such as `lives_in`, `works_at`, `based_in`, `likes`, and `has_task`, each linked to its source memory. **Valid time** says when an assertion was true; **recorded time** says when the system learned or revised it. `--as-of` and `--known-at` queries distinguish past truth from past knowledge. Multi-hop retrieval carries the evidence path. Mere graph connectivity is not treated as proof of an unstated fact. Natural-language extraction covers supported patterns and explicit metadata, not arbitrary prose.

### Layer 4: Retrieve and rank evidence

The lightweight `LocalMemoryStore.search()` requires lexical overlap. Its default score weights are keyword relevance **0.65**, category **0.10**, tier **0.05**, recency **0.10**, and importance **0.10**. `MemoryRAG.search()` adds independent vector and graph channels after filtering by user, session, history, task state, expiry, and time.

Hybrid search applies weighted reciprocal-rank fusion to memory IDs, then keeps `min(200, max(100, 4 × Top-K))` candidates by default (`COGNIMEM_CANDIDATE_POOL` overrides; a pool of 20 held LoCoMo gold evidence only 69.6% of the time). Fusion order is computed at the old depth (`max(20, 4 × Top-K)` per channel) and deeper candidates are appended, so the wider pool never reorders results when reranking is off or fails. Up to three representative chunks per candidate go to the local `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder. Each memory takes its best chunk score and keeps that chunk for context; fusion score and ID resolve ties. In hybrid mode a structural step then reorders the reranked list: candidates with a cued, currently valid graph path move up; the other memories of that path follow them; and a statement whose relation has closed yields to its successor when the question asks about that relation. `--no-rerank` exposes the fusion baseline. Model failure returns fused results with fallback metadata instead of failing retrieval. Vector search is exact by default; approximate LSH search is opt-in (`--ann`) and experimental.

### Layer 5: Build context and answer

The context selector keeps the reranked order (fusion-only results still prefer query coverage), removes near-duplicate snippets, and respects a character budget; per-source scores are returned in `context['scores']`, outside the prompt text. Graph paths are included only when all supporting memories fit. `ask --generator extractive` produces local, deterministic evidence text; adding the experimental `--abstain` flag refuses typed questions (name, place, amount, duration, …) that no confidently reranked memory answers. The default `ask` path uses LangChain/Gemini 2.5 Flash after a key is configured. Hosted output must contain source IDs and exact supporting quotes found in the supplied context; invalid citations are rejected, and insufficient evidence should lead to abstention. Quote checking verifies provenance, but it cannot by itself prove every semantic inference in a generated statement.

### Layer 6: Expose and operate the system

The CLI supports ingestion, inspection, search, graph queries, answers, history, reconciliation, and cleanup. FastAPI adds signup/login, bearer-token authentication, role checks, per-user memory endpoints, search, JSON/CSV export, account deletion, `/health/live`, `/health/ready`, `/metrics`, structured request logs, and `/docs`. The web demo is mounted at `/demo` when static assets are present. The CLI caller supplies a user ID; API calls derive it from the authenticated user.

## Setup

Run commands from the repository root. Python 3.13 and [uv](https://docs.astral.sh/uv/) are the development path used here; a standard virtual environment and `pip install -r requirements.txt` can also be used if the pinned packages support your platform.

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m main --help
```

Local embedding and reranker models download on first use into the ignored cache under `data/rag/models/`. Store and index files are local runtime data. The default importance scorer also needs the BGE encoder in that cache, and reads it only from there: run a semantic search once, or `.venv/bin/python -c "from main.retrieval.embeddings import FastEmbedder; FastEmbedder()"`, before the first `process`. Its model files ship in `artifacts/importance_ensemble/`; `COGNIMEM_SCORER=heuristic` runs without both.

For a hosted answer, copy the example environment file and set your key locally. The extractive generator needs no key.

```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY; GEMINI_API_KEY is also accepted.
.venv/bin/python -m main ask "What style of answer do I prefer?" \
  --user-id user_001 --generator gemini
```

The CLI `ask` command loads `.env` by default. The API reads its process environment and also requires a JWT secret of at least 32 bytes:

```bash
export COGNIMEM_JWT_SECRET="$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(48))')"
.venv/bin/python -m api
# Visit http://localhost:8000/docs for the endpoint schemas.
```

`COGNIMEM_STORE`, `COGNIMEM_AUTH_DB`, and `COGNIMEM_INDEX` override local paths. Vector search is exact by default; `COGNIMEM_ANN_ENABLED=true` (CLI: `--ann`) enables LSH for users with at least `COGNIMEM_ANN_EXACT_THRESHOLD` memories (default 5000, must be >= 1). `COGNIMEM_CANDIDATE_POOL` (1-200) and `COGNIMEM_MODEL_THREADS` (default: onnxruntime decides) tune reranking. Keep `.env`, tokens, and generated runtime data out of version control.

## First local workflow

These commands save a preference and a location change, then retrieve the current evidence for the same user. First semantic use may download and initialize local models.

```bash
.venv/bin/python -m main process "I prefer concise technical explanations." \
  --user-id user_001 --session-id session_001 --pretty
.venv/bin/python -m main process "I live in Chennai." \
  --user-id user_001 --session-id session_001
.venv/bin/python -m main process "I moved to Bengaluru." \
  --user-id user_001 --session-id session_002

.venv/bin/python -m main list --user-id user_001 --pretty
.venv/bin/python -m main search "Where do I live now?" \
  --user-id user_001 --mode hybrid --limit 5 --pretty
.venv/bin/python -m main ask "Where do I live now?" \
  --user-id user_001 --generator extractive --pretty
```

`index --user-id user_001` builds indexes explicitly; `graph --user-id user_001 --query Bengaluru` inspects relations; `history RECORD_ID --user-id user_001` shows revisions. Keyword search avoids model downloads. `ask --preview` shows context and prompt without calling Gemini. `--no-rerank` and `--candidate-limit 20` support retrieval comparisons. `cleanup --user-id user_001` previews cleanup; `--apply` executes it. Use `.venv/bin/python -m main COMMAND --help` for flags.

Python applications use the same core and store:

```python
from main import LocalMemoryStore, MemoryCore, MemoryInput
from main.retrieval.rag import MemoryRAG

store = LocalMemoryStore()
record = MemoryCore().process(MemoryInput(
    content="I prefer concise technical explanations.",
    user_id="user_001",
    session_id="session_001",
))
store.save(record)
hits = MemoryRAG(store).search("How should you answer me?", user_id="user_001")
print([(hit["memory_id"], hit["text"]) for hit in hits])
```

`process` uses the hybrid importance scorer by default; pass `--scorer heuristic` (or set `COGNIMEM_SCORER=heuristic`, or use `MemoryCore.with_heuristic()` in Python) for the rule-based score alone. The scorer changes retention scoring, not the retrieval model.

An HTTP integration can call the running API instead. Signup returns a bearer token; use its `data.access_token` value in subsequent requests. The authenticated identity determines which memories the caller can access.

```bash
curl -sS http://localhost:8000/api/v1/auth/signup \
  -H 'Content-Type: application/json' \
  -d '{"email":"demo@example.com","password":"a-long-demo-password"}'

# Copy data.access_token from the response into this shell variable.
export COGNIMEM_DEMO_TOKEN='paste-token-here'
curl -sS http://localhost:8000/api/v1/memories \
  -H "Authorization: Bearer $COGNIMEM_DEMO_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"content":"I prefer concise answers.","session_id":"demo-session"}'
curl -sS http://localhost:8000/api/v1/memories/search \
  -H "Authorization: Bearer $COGNIMEM_DEMO_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"query":"How should you answer me?","mode":"keyword","limit":5}'
```

The `/docs` page documents the other endpoints. Token, store, and index files are local runtime state; do not commit them.

## Evaluation and measured results

**The tables below describe different datasets and tasks. Do not compare their percentages as if they were one experiment.** Retrieval metrics concern evidence selection, answer metrics concern the extractive answer, and retention metrics concern what the lifecycle keeps. Every number comes from a tagged snapshot in [`evaluation/reports/`](evaluation/reports/) (`<tag>_summary.json`), except the importance model's own test metrics, which come from its training report ([`retention_ensemble.json`](evaluation/reports/retention_ensemble.json)). The tags are `baseline` (12 Oct 2026, before the fixes), `e2r` (search fixes only), `final` (all fixes, heuristic scorer), `final_abstain` (`final` plus the opt-in abstention rule), and from 16 Oct `heuristic_control` and `hybrid` (the scorer-dependent measurements with the heuristic alone and with the new default). The [evaluation guide](evaluation/README.md) describes the protocol.

### Before and after

#### What was there before

The metrics saved on 29 Sep 2026 looked strong but did not hold up:

| Saved claim (29 Sep 2026) | Value | Problem found on re-measurement |
| --- | ---: | --- |
| Controlled benchmark Recall@5, vector + graph + reranker | 91.36% | 240 template questions written alongside the rules. Procedural questions scored 5%, because `ingest()` merged 19 of their 20 memories. |
| Abstention accuracy | 91.67% | The extractive answerer refused **0** of the unanswerable questions; the figure counted answerable questions as "correct". |
| LoCoMo hybrid + reranker Recall@5 | 55.83% | Measured with exact search. Approximate (LSH) search, added on 1 Oct, was on by default from 256 memories, so the shipped path was never measured. |
| 48/48 classification cases | 100% | The cases had been used to develop the rules. |

So on 12 Oct 2026 every cache, model and report was deleted and rebuilt from scratch:
- a new harness;
- a new AI-authored benchmark (12 personas, 456 memories, 360 questions, dev 90 / test 270; not human-validated);
- 60 new classification cases;
- retrained retention models, whose metrics reproduced exactly;
- a fresh baseline.

The baseline exposed these problems:

| Problem | Baseline evidence |
| --- | --- |
| Approximate search shipped by default and dropped most candidates | Benchmark Recall@5 93.0% exact vs **57.5%** with LSH; LoCoMo 55.8% vs **26.8%** |
| The reranker saw too few candidates | LoCoMo gold evidence reached the 20-item reranker pool only **69.6%** of the time |
| The reranker ignored time and graph links | Correct memory ranked first: temporal **27.3%**, historical **52.2%**; multi-hop Recall@5 **68.2%** (90.9% before reranking) |
| Users' own statements expired after one hour | LongMemEval user-fact retention **21.0%** (86% of answer turns labelled temporary) |
| No abstention | Unanswerable questions refused **0%** (balanced answerability 50%) |
| `search(mode='recency')` always raised | Crash on every call |

#### What was changed

Each change was designed on dev data, then checked by an independent adversarial review that re-ran the numbers and probed for over-fitting before it was merged. The final code review found 14 further defects, all of them fixed. Changes are listed in the order they were measured, and intervals are paired-bootstrap 95% CIs.

| Change | Files | Mechanism | Measured change |
| --- | --- | --- | --- |
| Keep what users say about themselves | [`main/domain/text_rules.py`](main/domain/text_rules.py), [`classification.py`](main/domain/classification.py), [`lifecycle.py`](main/domain/lifecycle.py) | Questions and hedges that contain first-person facts become episodic, not temporary. Small talk, and questions asking to recall a stored fact, stay temporary. User self-disclosures are kept 90 days instead of 14. | LongMemEval user-fact retention **21.0% → 90.6%**; memories still visible 18.2% → 55.2%. Category accuracy on the 60 cases 50.0% → 53.3%; tier accuracy 55.0% → 53.3%. |
| Exact search by default; corrected approximate search | [`main/storage/indexes.py`](main/storage/indexes.py), [`main/retrieval/rag.py`](main/retrieval/rag.py), [`api/config.py`](api/config.py) | Exact numpy scan by default; LSH is opt-in (`--ann`) and experimental. LSH now probes radii 0–2 until it has enough candidates, falls back to exact search, and rebuilds when its settings change. | Shipped path is exact. With LSH forced on: benchmark Recall@5 **57.5% → 97.5%**; LoCoMo Recall@5 **26.8% → 66.5%** and pool hit 33.3% → 90.6%. |
| Bigger reranker pool | [`main/retrieval/rag.py`](main/retrieval/rag.py), [`reranker.py`](main/retrieval/reranker.py) | The reranker sees 100 fused candidates instead of 20 (`COGNIMEM_CANDIDATE_POOL` overrides). Fusion order without the reranker is unchanged, the 600-character cap is gone, and limits 51–100 no longer crash. | LoCoMo Recall@5 **55.8% → 66.8%** (+11.0 [+9.5, +12.5]); Hit@5 61.1% → 72.3%; pool hit 69.6% → 91.9%; multi-hop 34.7% → 41.8%. On the 7 conversations not used to choose the pool: 55.7% → 66.8%. Benchmark: +0.6 points (not significant). |
| Keep the reranker's order in the answer context | [`main/retrieval/rag.py`](main/retrieval/rag.py) | Context follows the reranked order and exposes per-source scores (`context['scores']`) outside the prompt text. | The first statement of the extractive answer is now the top-ranked memory: 84.2% correct in `final`. |
| Time- and link-aware reordering after the reranker | [`main/retrieval/rag.py`](main/retrieval/rag.py) | Graph-supported candidates and their chain partners move up. A statement whose relation has closed yields to its successor when the question asks about that relation. The "newer of two statements" step is opt-in, because its gains did not survive paraphrase. | Benchmark Recall@5 **93.7% → 97.5%** (+3.8 [+2.2, +5.7]); correct first **77.3% → 84.2%** (+6.9 [+3.6, +10.5]); MRR 0.904 → 0.939; multi-hop Recall@5 65.9% → **100%**. Correct first: temporal 27.3% → 59.1%, historical 52.2% → 91.3%, preference 82.6% → 91.3%. LoCoMo unchanged (it has no stored relations). |
| Opt-in extractive abstention | [`main/retrieval/generation.py`](main/retrieval/generation.py), [`main/__main__.py`](main/__main__.py) | `ask --generator extractive --abstain` refuses when the question asks for a typed answer (name, place, occupation, amount, duration…) that no confidently reranked memory contains. Off by default. | Benchmark test: unanswerable refused **0% → 17.4%**, balanced 50.0% → 58.5%, answerable kept 99.6%. External sets show no useful gain (LongMemEval balanced 52.5%, LoCoMo 50.0%; [details](evaluation/reports/abstention_external.json)). |
| Recency search fix | [`main/retrieval/rag.py`](main/retrieval/rag.py) | The custom-channel check no longer rejects the built-in recency mode. | `search(mode='recency')` works. |

#### Overall: baseline → final

| Metric | Baseline | Final | Change |
| --- | ---: | ---: | --- |
| Benchmark Recall@5 (test, 247 answerable) | 93.0% | **97.5%** | +4.5 [+2.6, +6.5], p = 0.0002 |
| Benchmark correct memory first | 77.3% | **84.2%** | +6.9 [+3.6, +10.5], p = 0.0002 |
| Benchmark MRR / nDCG@10 | 0.904 / 0.879 | **0.939 / 0.933** | MRR +3.5 [+1.6, +5.6] |
| Benchmark grounded correctness (answerable) | 97.6% | 99.2% | +1.6 [+0.4, +3.2], p = 0.13 (not significant) |
| LoCoMo Recall@5 (1,973 questions) | 55.8% | **66.8%** | +11.0 [+9.5, +12.5], p = 0.0002 |
| LoCoMo Hit@5 / Recall@10 | 61.1% / 60.0% | **72.3% / 73.4%** | Hit@5 +11.3 [+9.6, +12.9] |
| LongMemEval user-fact retention | 21.0% | **90.6%** | +69.6 points |
| Unanswerable refused (opt-in abstention) | 0% | 17.4% | answerable kept 99.6% |
| Search latency P50, benchmark / LoCoMo | 85 / 119 ms | 246 / 435 ms | about 3–4× slower; mostly the 5× larger reranker pool |

What did not improve: correction questions ("scratch that…") stay at 54.5% correct first, and latest-update questions moved 43.5% → 39.1% (one question). Abstention barely helps on real conversations. Latency rose: `COGNIMEM_CANDIDATE_POOL=50` keeps most of the LoCoMo gain at about half the reranker cost.

### Current results (snapshot `final`; the importance scorer does not affect retrieval)

#### Controlled benchmark, test split

There are 270 questions: 247 answerable and 23 unanswerable, the latter rewritten to cover 26 attribute types. The run uses exact search, the default pool (100 for reranked arms, 20 for fusion-only arms), top-5 context and the extractive answerer.

| Arm | Recall@5 | Recall@10 | MRR | nDCG@10 | Correct first | Grounded correct (all) | P50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid (keyword + vector + graph) + reranker, shipped | 97.5% | 99.5% | 0.939 | 0.933 | 84.2% | 90.7% | 246 ms |

The [full report](docs/reports/final/ablation/results.md) breaks results down by arm (vector, vector + reranker, vector + graph, vector + graph + reranker), by query type and by number of gold memories, and lists failure cases. It is regenerated locally under the git-ignored `docs/`.

#### External LoCoMo evidence retrieval

These are evidence-retrieval metrics on 1,973 questions over 10 generated conversations, not answer scores. Conversations 1–3 were used to choose the pool size.

| Mode | Recall@5 | Hit@5 | MRR@5 | Pool hit | Multi-hop R@5 | P50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid fusion (no reranker) | 48.2% | 53.1% | 0.374 | 69.6% | 22.3% | 22 ms |
| Hybrid + reranker, shipped | **66.8%** | **72.3%** | 0.589 | 91.9% | 41.8% | 435 ms |
| Hybrid + reranker, held-out 7 conversations | 66.8% | 72.8% | 0.586 | 92.4% | — | — |
| Hybrid + reranker, approximate search (opt-in) | 66.5% | 72.0% | 0.587 | 90.6% | 40.9% | 428 ms |

The graph channel alone still finds almost no LoCoMo evidence (Recall@5 1.2%), because raw dialogue turns carry no explicit relations.

#### Retention and lifecycle

With the hybrid default (snapshot `hybrid`; the [comparison with `final`](#effect-on-the-system-measured-16-oct-2026) is below):

| Check | Result |
| --- | --- |
| LongMemEval oracle (457 examples): answer turns still retrievable when the question is asked | 90.6% (keep-everything: 100%) |
| Mean share of a conversation still visible | 55.2% |
| 60 classification cases (AI-authored): category / tier accuracy | 53.3% / 56.7% (macro F1 0.544) |
| Lifecycle, task and supersession checks | 16/16 |

### Importance scoring (default)

Since 16 Oct 2026 the importance score of every new memory is **25 % a trained model + 75 % the heuristic** ([`main/ensemble.py`](main/ensemble.py)). The model is the csv5k "Ensemble + thresholds". Every other XGBoost and neural retention model, with its training code, data and reports, was removed; the old models are backed up outside the repository, and the dated snapshots keep their numbers as history.

```
model_score = class value: low 0.3102 · mid 0.5792 · high 0.8372    (× 0.25 for small talk, as the heuristic does)
score       = round(0.25 × model_score + 0.75 × heuristic_score, 4)
```

Each class value is the mean csv5k importance of the fit and validation sentences in that class (3,992 rows; the test split is excluded). The lifecycle thresholds are unchanged: working below 0.25, long-term at 0.42 or more for durable categories, and cleanup keeps expired memories scoring 0.65 or more. The record's category still comes from the normal classifier. `COGNIMEM_SCORER=heuristic`, `process --scorer heuristic` or `MemoryCore.with_heuristic()` restores the heuristic alone. Every record stores which scorer ran, the model's class and probabilities, and both part-scores in `source_metadata["importance_model"]`.

#### The model

| Part | Detail |
| --- | --- |
| Training data | `cognimem_memory_importance_5000.csv` ("csv5k"): 5,000 AI-generated memory sentences (443 distinct after removing an " (example N)" suffix) with synthetic importance scores 0.20–1.00. Classes: low < 0.4 ≤ mid < 0.7 ≤ high. Every copy of a sentence stays in one split: 3,200 / 792 / 1,008 rows (286 / 71 / 86 sentences). |
| XGBoost member | 525 features (13 text features + 512 hashed word counts); max depth 3, min child weight 3, 304 rounds, trained on the training split only; temperature-calibrated. |
| Deep-head member | Frozen `BAAI/bge-small-en-v1.5` embedding (384 values) + the 13 text features: text branch 384→192→128, feature branch 13→64→32, joint head 160→128→64→3. Seeds 42, 43, 44, each calibrated, averaged and calibrated again. |
| Blend and thresholds | 0.5 × XGBoost + 0.5 × deep head, then class weights 1 / 0.7 / 1 pick low / mid / high. Temperatures, blend and class weights were chosen on validation only. |
| Files | [`artifacts/importance_ensemble/`](artifacts/importance_ensemble/) (2.5 MB, checksummed, tracked in Git). `python -m training.ensemble.run --csv /path/to/cognimem_memory_importance_5000.csv` rebuilds the model files bit for bit (the CSV is copied to `data/experiments/`, so later runs need no flag; `metadata.json` carries a timestamp); the scorer reads the BGE encoder only from the local cache and never downloads it. |

Test split of csv5k (1,008 rows from 86 sentences; intervals are 95% bootstrap intervals over whole sentences; [full report](evaluation/reports/retention_ensemble.md)):

| Model | Accuracy | Balanced accuracy | Macro F1 [95% CI] | F1 low / mid / high |
| --- | ---: | ---: | --- | --- |
| Always the most common class | 61.11% | 33.33% | 0.2529 [0.220, 0.283] | 0.00 / 0.00 / 0.76 |
| Best possible (oracle) | 91.17% | 92.86% | 0.9230 [0.885, 0.957] | 1.00 / 0.84 / 0.93 |
| **Ensemble + thresholds** | 89.09% | 87.14% | 0.8912 [0.809, 0.949] | 1.00 / 0.76 / 0.92 |

The oracle predicts each test sentence's most common test label; it reads the test labels, so it is a ceiling, not a model.

#### Effect on the system (measured 16 Oct 2026)

Only three measurements use the importance scorer: LongMemEval retention, the 60 classification cases and the 16 lifecycle checks. The benchmark and LoCoMo runners never call it, so their numbers above are unchanged. The same-day heuristic run uses the same code and clock as the hybrid run and reproduces `final` on all 457 LongMemEval examples, so the last two columns isolate the scorer.

| Metric | `final` (heuristic, 12 Oct) | Heuristic, same day | **Hybrid (default)** |
| --- | ---: | ---: | ---: |
| LongMemEval user-fact retention | 90.6% | 90.6% | **90.6%** |
| Mean share of a conversation still visible | 55.2% | 55.2% | **55.2%** |
| Cleanup: expired memories deleted / kept as archive | 1,740 / — | 1,740 / 33 | **1,738 / 38** |
| Classification cases: category / tier accuracy | 53.3% / 53.3% | 53.3% / 56.7% | **53.3% / 56.7%** |
| Classification cases: score inside the labelled range | 50.0% | 50.0% | **58.3%** |
| Lifecycle, task and supersession checks | 16/16 | 16/16 | **16/16** |
| LongMemEval replay time | 13 s | 14 s | **581 s** |

- **Retention did not change.** All 457 of 457 examples keep the same answer turns and the same visible share. Tiers come mostly from the category rules; at 25 % weight the model did not change the visibility of a single answer turn. Cleanup kept 5 more expired memories as archive and deleted 2 fewer.
- **The tier-accuracy rise is the clock, not the model.** The classification runner now pins its clock to the cases' timestamp; with the heuristic alone that gives 56.7%, the same as the hybrid.
- **The range-agreement rise is not better judgement.** A constant score of 0.5 lands inside 83.3% of the hand-labelled ranges, because 40 of the 60 ranges are [0.45, 1].
- **The model does not recognise answer-bearing turns.** As a ranking of answer-bearing against other turns its score has an AUC of 0.482 on user turns (heuristic 0.582) and 0.697 on assistant turns; the blended score's overall AUC is 0.654 against 0.678 for the heuristic. It labels 81.2% of turns "high", and 98.9% of turns have numeric inputs more than 5 standard deviations from the training data: conversation turns are far longer than the at most 13-word training sentences.
- **It costs time.** Each memory is embedded with BGE, about 59 ms per memory on average over the 9,831 replayed turns (long assistant turns cost most), so the LongMemEval replay took 41× longer.

Caveats: csv5k is synthetic (AI-written sentences and scores, not human ratings), and its test split has only 86 distinct sentences. The model predicts low / mid / high importance, a different target from the removed models' invalid / short-term / long-term. Two lifecycle-check texts ("I live in Chennai.", "I live in Bengaluru.") appear verbatim in csv5k, and ten "My name is …" variants do too. Records stored before the switch keep their heuristic score, so an existing store mixes the two scales.

## Reproduce the experiments

After installation, run from the repository root:

```bash
.venv/bin/python -m training.ensemble.run --csv ~/Downloads/cognimem_memory_importance_5000.csv   # importance model (about 30 s)
.venv/bin/python scripts/measure_all.py --tag final --candidate-limit default --ann-arms   # every measurement
.venv/bin/python scripts/measure_all.py --tag hybrid --scorer-steps-only                  # only what the importance scorer affects
.venv/bin/python scripts/measure_all.py --tag heuristic_control --scorer-steps-only --scorer heuristic
.venv/bin/python scripts/report_delta.py baseline final --allow-dataset-mismatch
.venv/bin/python scripts/report_delta.py heuristic_control hybrid
```

`measure_all.py` runs the benchmark ablation (test and dev), LoCoMo, LongMemEval retention, classification and lifecycle checks, and snapshots the reports under the tag; `--scorer-steps-only` runs just the last three, and `--scorer` picks the importance scorer. The first run downloads the models and datasets, and a full run takes about an hour on a laptop. `--abstain-threshold 0.0` measures the opt-in abstention rule, and `--structure none` disables the time- and link-aware reordering. The [evaluation guide](evaluation/README.md) lists every flag and module.

To run the automated tests, install the test runner (`uv pip install --python .venv/bin/python -r requirements-dev.txt`), then run `.venv/bin/python -m pytest -q`.

## Repository map

| Path | Responsibility |
| --- | --- |
| [`main/domain/`](main/domain/) | Models, text rules, classification, scoring, lifecycle, tasks, privacy, claims, reconciliation. |
| [`main/application/`](main/application/) | Memory Core orchestration and maintenance. |
| [`main/storage/`](main/storage/) | JSONL revisions, locks, SQLite index structures, cleanup. |
| [`main/graph/`](main/graph/) | Temporal entities/relations, provenance, historical and multi-hop queries. |
| [`main/retrieval/`](main/retrieval/) | Keyword ranking, BGE embeddings, hybrid search, reranking, context, answers. |
| [`main/ensemble.py`](main/ensemble.py), [`main/ml.py`](main/ml.py), [`training/ensemble/`](training/ensemble/), [`artifacts/importance_ensemble/`](artifacts/importance_ensemble/) | Default hybrid importance scorer, its feature schema, its training, and the trained model. |
| [`api/`](api/), [`static/`](static/) | Authenticated HTTP API, monitoring, privacy operations, demo. |
| [`evaluation/`](evaluation/), [`docs/reports/`](docs/reports/) | Datasets, runners, saved metrics, failure cases, review forms. |
| [`tests/`](tests/) | Automated behavior and integration tests. |

## Limits and next steps

- **Generalization:** the controlled benchmark and classification cases were authored by an AI and are not human-validated, and its dev and test personas share templates. LoCoMo is generated dialogue, and its first three conversations were used to choose the reranker pool. Fresh conversation-disjoint data and independent human ratings are needed for a broader claim.
- **Known gaps:** correction and latest-update questions are still often answered with an older statement; abstention is opt-in and catches only "attribute never stated" questions; category classification is 53% on the 60 cases; the larger reranker pool costs about 3–4× search latency; approximate search remains experimental. One lifecycle check and one task test fail rarely because of a wall-clock race in task updates.
- **Answer quality:** saved RAG scores use extractive answers. Live Gemini correctness, personalization, citation usefulness, and conflict decisions still need independent review. Exact-quote citation validation alone cannot prove semantic truth.
- **Extraction and retention:** graph extraction and indirect contradiction handling cover supported language patterns. The trained quarter of the importance score learned from synthetic sentences, does not recognise answer-bearing conversation turns, and changed no measured retention outcome; human-rated, in-domain importance labels are needed before giving it more weight.
- **Operations:** local JSONL and SQLite are not a production database architecture. Deployment needs backups, retention-policy review, stronger authorization/abuse controls, and workload-specific scale and latency measurements.

For more implementation detail and day-to-day workflows, see [DOCUMENTATION.md](DOCUMENTATION.md). For evaluation protocols, see [evaluation/README.md](evaluation/README.md) and the saved reports linked above.
