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
| Retention | Heuristic scorer by default; optional XGBoost and neural retention-proxy scorers; working, short-term, long-term, and archive tiers. |
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

The default `ImportanceScorer` combines deterministic features into a 0–1 score. `LifecycleManager` uses that score and category-specific rules to assign `working`, `short_term`, `long_term`, or `archive`. Temporary messages normally remain working memory, one-off tasks remain short-term even when important, and durable facts or preferences can become long-term. Working memory normally expires after one hour, short-term after fourteen days, and dated tasks after their deadline plus 24 hours. Long-term records can appear as archive views after inactivity. These are policy decisions, not human-labeled truths.

| Category | Typical input | Why it matters later |
| --- | --- | --- |
| Semantic | “My supervisor is Priya.” | Stable facts and relationships. |
| Episodic | “I submitted the proposal yesterday.” | Event history and temporal questions. |
| Procedural | “First run tests, then deploy.” | Reusable instructions. |
| Preference | “I prefer brief explanations.” | Personalization and constraints. |
| Task | “Remind me to submit the report Friday.” | Action state, deadlines, recurrence. |
| Temporary | “Thanks!” | Immediate context without durable retention. |

The optional XGBoost and neural scorers estimate `P(long_term) + 0.5 × P(short_term)` from extracted conversational facts. They predict a **retention-policy proxy**, not human-rated importance, and never replace the heuristic automatically. The neural path freezes the local 384-dimensional BGE encoder, combines text embeddings with 13 numeric features in a two-branch head, and serves its trained head through ONNX.

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

Local embedding and reranker models download on first use into the ignored cache under `data/rag/models/`. Store and index files are local runtime data. Optional trained retention scorers need their artifacts present or locally trained; see [reproduction](#reproduce-the-experiments).

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

For optional trained retention scoring, select `MemoryCore.with_ml()` (XGBoost) or `MemoryCore.with_neural()`, or pass `--scorer xgboost` / `--scorer neural` to `process`. These options change retention scoring, not the retrieval model.

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

**The tables below describe different datasets and tasks. Do not compare their percentages as if they were one experiment.** Retrieval metrics concern evidence selection, answer metrics concern the extractive answer, and retention metrics concern what the lifecycle keeps. Every number comes from a tagged snapshot in [`evaluation/reports/`](evaluation/reports/) (`<tag>_summary.json`), except the three-dataset retention experiment, which has its own report in [`evaluation/reports/experiments/`](evaluation/reports/experiments/). The tags are `baseline` (12 Oct 2026, before the fixes), `e2r` (search fixes only), `final` (all fixes, shipped defaults) and `final_abstain` (`final` plus the opt-in abstention rule). The [evaluation guide](evaluation/README.md) describes the protocol.

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

### Current results (shipped defaults, snapshot `final`)

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

| Check | Result |
| --- | --- |
| LongMemEval oracle (457 examples): answer turns still retrievable when the question is asked | 90.6% (keep-everything: 100%) |
| Mean share of a conversation still visible | 55.2% |
| 60 classification cases (AI-authored): category / tier accuracy | 53.3% / 53.3% (macro F1 0.544) |
| Lifecycle, task and supersession checks | 16/16 |

### Optional retention models (unchanged)

XGBoost, an original shallow neural head, and a deeper neural head were evaluated on the same **556-fact** published test split from [Personal Facts (MSC)](https://huggingface.co/datasets/adugeen/personal-facts-msc): 85 invalid, 92 short-term, and 379 long-term. The always-long-term baseline exposes class imbalance. Proxy MAE compares a constructed target (`invalid=0`, `short_term=0.5`, `long_term=1`), not human importance ratings.

| Test metric | Deeper neural | Original neural | XGBoost | Always long-term |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 78.06% | 78.06% | 69.24% | 68.17% |
| Balanced accuracy | 67.78% | 65.86% | 65.60% | 33.33% |
| Macro F1 | **0.6732** | 0.6504 | 0.6114 | 0.2702 |
| Weighted F1 | 0.7768 | 0.7703 | 0.7071 | 0.5526 |
| Constructed proxy MAE | 0.2160 | **0.1957** | 0.2768 | — |
| Negative log-likelihood | 0.5547 | **0.5345** | — | — |
| Expected calibration error, 10 bins | **0.0163** | 0.0428 | — | — |
| Warm inference P50 | **4.17 ms** | 5.01 ms | — | — |
| Warm inference P95 | **5.79 ms** | 6.74 ms | — | — |

Per-class F1 (invalid / short-term / long-term) is **0.4932 / 0.6570 / 0.8696** for the deeper head, **0.4286 / 0.6455 / 0.8773** for the original, and **0.4574 / 0.5837 / 0.7931** for XGBoost. Invalid-fact recall is **42.35%** for the deeper head, **31.76%** for the original, and **50.59%** for XGBoost; the deeper model does not win every safety-relevant measure. `—` means unrecorded. Neural latency includes local BGE encoding and ONNX inference after warm-up; XGBoost latency was not measured.

Class-level test metrics (support is the number of gold examples):

| Model | Gold class | Support | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Deeper neural | Invalid | 85 | 0.5902 | 0.4235 | 0.4932 |
| Deeper neural | Short-term | 92 | 0.5913 | 0.7391 | 0.6570 |
| Deeper neural | Long-term | 379 | 0.8684 | 0.8707 | 0.8696 |
| Original neural | Invalid | 85 | 0.6585 | 0.3176 | 0.4286 |
| Original neural | Short-term | 92 | 0.5547 | 0.7717 | 0.6455 |
| Original neural | Long-term | 379 | 0.8682 | 0.8865 | 0.8773 |
| XGBoost | Invalid | 85 | 0.4175 | 0.5059 | 0.4574 |
| XGBoost | Short-term | 92 | 0.4823 | 0.7391 | 0.5837 |
| XGBoost | Long-term | 379 | 0.8782 | 0.7230 | 0.7931 |
| Always long-term | Invalid | 85 | 0 | 0 | 0 |
| Always long-term | Short-term | 92 | 0 | 0 | 0 |
| Always long-term | Long-term | 379 | 0.6817 | 1.0000 | 0.8107 |

Confusion matrices below have gold classes as rows and predicted classes as columns:

| Model | Gold class | Predicted invalid | Predicted short-term | Predicted long-term |
| --- | --- | ---: | ---: | ---: |
| Deeper neural | Invalid | 36 | 18 | 31 |
| Deeper neural | Short-term | 5 | 68 | 19 |
| Deeper neural | Long-term | 20 | 29 | 330 |
| Original neural | Invalid | 27 | 26 | 32 |
| Original neural | Short-term | 2 | 71 | 19 |
| Original neural | Long-term | 12 | 31 | 336 |
| XGBoost | Invalid | 43 | 17 | 25 |
| XGBoost | Short-term | 11 | 68 | 13 |
| XGBoost | Long-term | 49 | 56 | 274 |
| Always long-term | Invalid | 0 | 0 | 85 |
| Always long-term | Short-term | 0 | 0 | 92 |
| Always long-term | Long-term | 0 | 0 | 379 |

The deeper head's mean validation macro F1 over three seeds was **0.7194**, versus **0.7134** for the original; its selected seed reached **0.7291**. The dataset has one annotator, consists of extracted facts, and is not conversation-disjoint. The published test split had already been inspected during XGBoost development. These comparisons are exploratory and do not justify changing the default heuristic or lifecycle thresholds. Detailed class precision/recall, confusion matrices, calibration, and training methods are in the [XGBoost](evaluation/reports/conversational_retention.md), [original neural](evaluation/reports/neural_retention_shallow.md), [deeper neural](evaluation/reports/neural_retention.md), and [depth comparison](evaluation/reports/neural_depth_validation.json) reports.

Model selection used a stratified training/validation split rather than choosing on the test set. XGBoost's four validation trials had macro F1 **0.5934, 0.6022, 0.6002, and 0.6211**; the last configuration (max depth 5, min child weight 8, 396 trees on refit) was selected. The original neural head's three best-checkpoint validation macro F1 values were **0.7175, 0.7057, and 0.7169** for seeds 42–44; the deeper head's were **0.7208, 0.7083, and 0.7291**. The selected original checkpoint used seed 42, epoch 4; the deeper checkpoint used seed 44, epoch 8. Temperature calibration reduced validation negative log-likelihood from **0.5644 to 0.5332** for the original (temperature 1.3946) and from **0.7913 to 0.5385** for the deeper head (temperature 2.4057).

### Retention models on three datasets (experiment, not shipped)

On 16 Oct 2026 the XGBoost model and both neural heads were retrained on three datasets, each trained and tested on its own. Three possible improvements were tested alongside them: averaging the neural heads over seeds, an XGBoost + neural ensemble, and per-class decision thresholds. This is a separate experiment. Nothing in `main/`, the shipped models under `artifacts/importance*` or the reports above changed, and the heuristic remains the default. Every number below comes from [`evaluation/reports/experiments/retention_datasets.md`](evaluation/reports/experiments/retention_datasets.md), which also lists validation scores, every seed and every XGBoost trial. The code is in [`training/experiments/`](training/experiments/), and [Reproduce the experiments](#reproduce-the-experiments) has the command.

#### Datasets

- **MSC**: [Personal Facts (MSC)](https://huggingface.co/datasets/adugeen/personal-facts-msc), with the same pinned files and split as the shipped models above. Each extracted fact is labelled invalid, short-term or long-term by the dataset's single annotator.
- **csv5k**: `cognimem_memory_importance_5000.csv`, 5,000 AI-generated memory sentences, each with a synthetic importance score from 0.20 to 1.00 (mean 0.708). Removing the " (example N)" suffix leaves 443 distinct sentences.
- **csv10k**: `cognimem_10000_memory_importance.csv`, 10,000 AI-generated sentences with scores from 0.30 to 0.99 (mean 0.644) and 2,410 distinct sentences. One sentence appears 500 times, with scores from 0.30 to 0.78.

The CSV scores are cut into three classes: low (below 0.4), mid (0.4 to below 0.7) and high (0.7 and above). Most CSV sentences appear many times, so every copy of a sentence goes into the same split (a 5-fold split grouped by sentence, seed 42); otherwise the test set would repeat training sentences. A sentence holding at least 5% of the rows stays in training, which affects only csv10k's 500-copy sentence. The CSVs are copied to the git-ignored `data/experiments/`.

| | MSC | csv5k | csv10k |
| --- | --- | --- | --- |
| Rows / unique texts | 2,778 / 2,778 | 5,000 / 443 | 10,000 / 2,410 |
| Labels | invalid / short-term / long-term | low / mid / high | low / mid / high |
| Train / validation / test rows | 1,777 / 445 / 556 | 3,200 / 792 / 1,008 | 6,578 / 1,521 / 1,901 |
| Test sentences | 556 | 86 | 476 |
| Test class counts | 85 / 92 / 379 | 120 / 272 / 616 | 86 / 1,017 / 798 |
| Repeated texts whose labels disagree | — | 86 of 410 (21%) | 603 of 1,017 (59%) |
| Best possible macro F1 (oracle) | — (every text unique) | 0.9230 | 0.5264 |

The oracle predicts each test sentence's most common test label. It reads the test labels, so it is a ceiling, not a model. csv10k's ceiling is low because 59% of its repeated sentences carry conflicting labels. The oracle maximises accuracy rather than macro F1, so a model can score slightly above it on macro F1.

#### Architectures

| Name in the tables | What it is |
| --- | --- |
| XGBoost, shipped recipe | Gradient-boosted trees on 525 features (13 text features and 512 hashed word features). A grid over max depth {3, 5} × min child weight {3, 8} is searched on validation, then the winner is refit on train + validation. This is the shipped procedure. |
| XGBoost, no refit | The grid winner trained on the train split only. The ensemble uses it, because the refit model has no unseen validation data left to tune on. |
| Deep neural head | Frozen BAAI/bge-small-en-v1.5 sentence embedding (384 values) plus the 13 text features. Text branch 384→192→128, feature branch 13→64→32, joint head 160→128→64→3, with LayerNorm, GELU and dropout 0.2. The best of seeds 42, 43 and 44 on validation is kept, then temperature-calibrated. Same design as the shipped deeper head. |
| Shallow neural head | Same inputs. Text branch 384→128, feature branch 13→32, joint head 160→64→3. Same design as the original shipped head. |
| 3-seed average | Averages the calibrated probabilities of all three seeds instead of keeping the best one. |
| Ensemble | w × XGBoost (no refit, calibrated) + (1 − w) × the averaged deep head, with w ∈ {0.25, 0.5, 0.75} chosen on validation. |
| + per-class thresholds | Multiplies each class probability by a weight (81 combinations) chosen on validation before taking the top class. |

Every choice is made on the validation split, and each model is scored on the test split once.

#### Macro F1 at a glance

| Model | MSC | csv5k | csv10k |
| --- | ---: | ---: | ---: |
| Always the most common class | 0.2702 | 0.2529 | 0.2324 |
| Best possible (oracle) | — | 0.9230 | 0.5264 |
| XGBoost, shipped recipe | 0.6114 | 0.8846 | 0.5322 |
| XGBoost, no refit | 0.6012 | 0.8654 | 0.5309 |
| Deep neural head | 0.6732 | 0.8850 | 0.4999 |
| Shallow neural head | 0.6504 | 0.8538 | 0.5317 |
| Deep head, 3-seed average | 0.6992 | 0.8699 | 0.5333 |
| Shallow head, 3-seed average | 0.6631 | 0.8678 | **0.5441** |
| Ensemble (XGBoost + averaged deep head) | **0.7132** | 0.8882 | 0.5342 |
| Ensemble + per-class thresholds | **0.7132** | **0.8912** | 0.5342 |

**Bold** marks the best model on each dataset; the oracle is a ceiling, not a model. Thresholds left the MSC ensemble unchanged, so the two ensemble rows tie there.

#### Full test metrics

Macro F1 weights the three classes equally, so the rare classes count as much as the common one. The 95% intervals come from 1,000 bootstrap resamples. MSC resamples test rows; the CSVs resample whole test sentences, because copies of a sentence are not independent. NLL (negative log-likelihood) and ECE (expected calibration error, 10 bins) measure the probabilities, and lower is better for both. XGBoost's MSC NLL and ECE were not recorded in the shipped report; they were computed here.

**MSC**, 556 test facts:

| Model | Accuracy | Balanced accuracy | Macro F1 [95% row CI] | Weighted F1 | F1 invalid / short-term / long-term | NLL | ECE |
| --- | ---: | ---: | --- | ---: | --- | ---: | ---: |
| Always the most common class (long-term) | 68.17% | 33.33% | 0.2702 [0.261, 0.279] | 0.5526 | 0.00 / 0.00 / 0.81 | — | — |
| XGBoost, shipped recipe | 69.24% | 65.60% | 0.6114 [0.562, 0.656] | 0.7071 | 0.46 / 0.58 / 0.79 | 0.7255 | 0.0246 |
| XGBoost, no refit | 68.71% | 64.18% | 0.6012 [0.553, 0.647] | 0.7013 | 0.45 / 0.56 / 0.79 | 0.7313 | 0.0247 |
| Deep neural head | 78.06% | 67.78% | 0.6732 [0.621, 0.720] | 0.7768 | 0.49 / 0.66 / 0.87 | 0.5547 | 0.0163 |
| Shallow neural head | 78.06% | 65.86% | 0.6504 [0.600, 0.700] | 0.7703 | 0.43 / 0.65 / 0.88 | 0.5345 | 0.0428 |
| Deep head, 3-seed average | 80.76% | 68.33% | 0.6992 [0.648, 0.745] | 0.7993 | 0.53 / 0.68 / 0.89 | 0.4979 | 0.0317 |
| Shallow head, 3-seed average | 78.78% | 67.10% | 0.6631 [0.611, 0.712] | 0.7789 | 0.45 / 0.66 / 0.88 | 0.5152 | 0.0538 |
| Ensemble (XGBoost + averaged deep head), w = 0.5 | 80.40% | 71.39% | 0.7132 [0.662, 0.758] | 0.8025 | 0.57 / 0.68 / 0.88 | 0.5399 | 0.1171 |

**csv5k**, 1,008 test rows from 86 sentences:

| Model | Accuracy | Balanced accuracy | Macro F1 [95% sentence CI] | Weighted F1 | F1 low / mid / high | NLL | ECE |
| --- | ---: | ---: | --- | ---: | --- | ---: | ---: |
| Always the most common class (high) | 61.11% | 33.33% | 0.2529 [0.220, 0.283] | 0.4636 | 0.00 / 0.00 / 0.76 | — | — |
| Best possible (oracle) | 91.17% | 92.86% | 0.9230 [0.885, 0.957] | 0.9126 | 1.00 / 0.84 / 0.93 | — | — |
| XGBoost, shipped recipe | 86.71% | 89.05% | 0.8846 [0.824, 0.930] | 0.8687 | 1.00 / 0.76 / 0.89 | 0.2239 | 0.0238 |
| XGBoost, no refit | 85.22% | 86.12% | 0.8654 [0.792, 0.917] | 0.8505 | 1.00 / 0.72 / 0.88 | 0.2358 | 0.0491 |
| Deep neural head | 88.00% | 87.16% | 0.8850 [0.810, 0.941] | 0.8750 | 1.00 / 0.75 / 0.91 | 0.2608 | 0.0421 |
| Shallow neural head | 84.82% | 84.19% | 0.8538 [0.770, 0.926] | 0.8415 | 1.00 / 0.68 / 0.88 | 0.2700 | 0.0584 |
| Deep head, 3-seed average | 87.60% | 84.68% | 0.8699 [0.778, 0.944] | 0.8632 | 1.00 / 0.70 / 0.91 | 0.2131 | 0.0240 |
| Shallow head, 3-seed average | 87.20% | 84.67% | 0.8678 [0.778, 0.941] | 0.8603 | 1.00 / 0.70 / 0.90 | 0.2682 | 0.0431 |
| Ensemble (XGBoost + averaged deep head), w = 0.5 | 88.39% | 87.37% | 0.8882 [0.811, 0.945] | 0.8788 | 1.00 / 0.75 / 0.91 | 0.2193 | 0.0346 |
| XGBoost on equal-thirds labels¹ | 81.55% | 81.58% | 0.8156 [0.747, 0.867] | 0.8140 | 0.83 / 0.74 / 0.88 | 0.3889 | 0.0435 |

**csv10k**, 1,901 test rows from 476 sentences:

| Model | Accuracy | Balanced accuracy | Macro F1 [95% sentence CI] | Weighted F1 | F1 low / mid / high | NLL | ECE |
| --- | ---: | ---: | --- | ---: | --- | ---: | ---: |
| Always the most common class (mid) | 53.50% | 33.33% | 0.2324 [0.217, 0.243] | 0.3729 | 0.00 / 0.70 / 0.00 | — | — |
| Best possible (oracle) | 74.91% | 51.99% | 0.5264 [0.480, 0.609] | 0.7239 | 0.09 / 0.81 / 0.69 | — | — |
| XGBoost, shipped recipe | 59.76% | 69.89% | 0.5322 [0.482, 0.577] | 0.6212 | 0.33 / 0.66 / 0.61 | 0.7850 | 0.1478 |
| XGBoost, no refit | 59.60% | 69.78% | 0.5309 [0.480, 0.575] | 0.6198 | 0.33 / 0.65 / 0.61 | 0.7693 | 0.0979 |
| Deep neural head | 62.39% | 53.59% | 0.4999 [0.453, 0.539] | 0.6320 | 0.21 / 0.70 / 0.59 | 0.7047 | 0.0483 |
| Shallow neural head | 64.12% | 57.11% | 0.5317 [0.456, 0.581] | 0.6458 | 0.28 / 0.71 / 0.61 | 0.7186 | 0.0360 |
| Deep head, 3-seed average | 60.28% | 69.94% | 0.5333 [0.483, 0.580] | 0.6231 | 0.34 / 0.67 / 0.59 | 0.7009 | 0.0852 |
| Shallow head, 3-seed average | 62.02% | 68.89% | 0.5441 [0.492, 0.587] | 0.6346 | 0.35 / 0.69 / 0.59 | 0.6963 | 0.0488 |
| Ensemble (XGBoost + averaged deep head), w = 0.25 | 60.39% | 70.02% | 0.5342 [0.484, 0.581] | 0.6244 | 0.34 / 0.67 / 0.59 | 0.7124 | 0.0677 |
| XGBoost on equal-thirds labels¹ | 49.40% | 49.67% | 0.4984 [0.461, 0.527] | 0.4992 | 0.50 / 0.39 / 0.61 | 0.9497 | 0.0335 |

¹ Uses different labels: the scores are cut into equal thirds (at 0.67 and 0.83 for csv5k, 0.56 and 0.73 for csv10k) instead of at 0.4 and 0.7, on the same split. It shows how much the cut-offs matter and is not directly comparable with the other rows.

Per-class thresholds chosen on validation. The weights follow the class order of each dataset, and 1/1/1 means unchanged:

| Dataset | Model | Chosen class weights | Macro F1 before → after | Accuracy before → after |
| --- | --- | --- | --- | --- |
| MSC | XGBoost, no refit | 0.5/0.5/1 | 0.6012 → 0.6155 | 68.71% → 73.38% |
| MSC | Deep neural head | 1/1/1 | 0.6732 → 0.6732 | 78.06% → 78.06% |
| MSC | Shallow neural head | 1/1/1 | 0.6504 → 0.6504 | 78.06% → 78.06% |
| MSC | Ensemble (XGBoost + averaged deep head) | 1/1/1 | 0.7132 → 0.7132 | 80.40% → 80.40% |
| csv5k | XGBoost, no refit | 1/0.5/1 | 0.8654 → 0.8687 | 85.22% → 87.20% |
| csv5k | Deep neural head | 1/1/1 | 0.8850 → 0.8850 | 88.00% → 88.00% |
| csv5k | Shallow neural head | 1/1/1 | 0.8538 → 0.8538 | 84.82% → 84.82% |
| csv5k | Ensemble (XGBoost + averaged deep head) | 1/0.7/1 | 0.8882 → 0.8912 | 88.39% → 89.09% |
| csv10k | XGBoost, no refit | 0.7/1.4/1 | 0.5309 → 0.5299 | 59.60% → 62.39% |
| csv10k | Deep neural head | 1.4/1.4/1 | 0.4999 → 0.4986 | 62.39% → 62.49% |
| csv10k | Shallow neural head | 2/2/1 | 0.5317 → 0.5334 | 64.12% → 65.18% |
| csv10k | Ensemble (XGBoost + averaged deep head) | 1/1/1 | 0.5342 → 0.5342 | 60.39% → 60.39% |

#### Did the improvements help?

Four comparisons per dataset were fixed before the test run. Each cell gives the change in test macro F1 and the one-sided paired-bootstrap p-value. A change counts as significant when p is below 0.0125 (0.05 split across the four comparisons); for the CSVs it must also hold when whole sentences are resampled.

| Comparison | MSC | csv5k | csv10k |
| --- | --- | --- | --- |
| Seed averaging vs single best seed (deep head) | +0.0260 (p 0.070; not significant) | −0.0151 (p 0.983 rows / 0.637 sentences; not significant) | +0.0333 (p 0.001 rows / 0.011 sentences; **significant**) |
| Ensemble vs averaged deep head | +0.0140 (p 0.241; not significant) | +0.0184 (p 0.008 rows / 0.320 sentences; significant on rows only) | +0.0010 (p 0.145 rows / 0.365 sentences; not significant) |
| Ensemble vs XGBoost (no refit) | +0.1120 (p 0.001; **significant**) | +0.0228 (p 0.007 rows / 0.199 sentences; significant on rows only) | +0.0034 (p 0.231 rows / 0.296 sentences; not significant) |
| Per-class thresholds vs ensemble | 0 (no change) | +0.0030 (p 0.293 rows / 0.185 sentences; not significant) | 0 (no change) |

What this shows:
- **MSC (real data):** the ensemble scores highest, 0.7132 against 0.6732 for the shipped deep head and 0.6114 for shipped XGBoost, and has the best invalid-fact F1 (0.57). Only its gain over XGBoost is significant; its gain over the averaged deep head (+0.0140) is not. Its probabilities are also the least calibrated (ECE 0.1171 against 0.0163 for the deep head). Seed averaging alone reaches 0.6992, with the best accuracy (80.76%) and the lowest NLL (0.4979).
- **csv5k:** every model scores between 0.8538 and 0.8912, close to the 0.9230 ceiling. Every model classifies the low class perfectly, so all errors are mid versus high. With 86 test sentences, the differences between models are within noise.
- **csv10k:** the models score between 0.4999 and 0.5441 against a ceiling of 0.5264, so this dataset cannot tell them apart. Seed averaging raised the deep head's macro F1 by 0.0333 (significant) but lowered accuracy by 2.10 points: it finds 95% of the low-score rows instead of 37%, at the cost of calling more mid and high rows low.
- **Thresholds** stayed at 1/1/1 in half of the cases and never moved macro F1 by more than 0.0143.

#### Checks and caveats

- The experiment code reproduces the shipped MSC models exactly. Scoring the shipped model files through it gives the shipped confusion matrices (macro F1 0.6114, 0.6732 and 0.6504), and retraining from scratch gives the same scores again.
- The CSV texts and scores are AI-generated, so only the MSC results reflect real data. The MSC results repeat the shipped evaluation rather than adding new evidence.
- The 0.4 / 0.7 cut-offs are a modelling choice. Equal thirds lower XGBoost to 0.8156 on csv5k and 0.4984 on csv10k.
- The effective test sets are small (86 sentences for csv5k), and validation was reused for several choices, so validation scores are optimistic. Only the test scores count.
- These results have not had the independent adversarial review that the shipped changes had.

## Reproduce the experiments

After installation, run from the repository root:

```bash
.venv/bin/python -m training.train_conversational_retention          # retention models (separate processes)
.venv/bin/python -m training.train_neural_retention
.venv/bin/python -m training.compare_neural_depth
.venv/bin/python scripts/measure_all.py --tag final --candidate-limit default --ann-arms   # every measurement
.venv/bin/python -m training.experiments.run_all   # optional: retention models on three datasets (not used by the system)
.venv/bin/python scripts/report_delta.py baseline final --allow-dataset-mismatch
```

`measure_all.py` runs the benchmark ablation (test and dev), LoCoMo, LongMemEval retention, classification and lifecycle checks, and snapshots the reports under the tag. The first run downloads the models and datasets, and a full run takes about an hour on a laptop. `--abstain-threshold 0.0` measures the opt-in abstention rule, and `--structure none` disables the time- and link-aware reordering. The [evaluation guide](evaluation/README.md) lists every flag and module.

To run the automated tests, install the test runner (`uv pip install --python .venv/bin/python -r requirements-dev.txt`), then run `.venv/bin/python -m pytest -q`.

## Repository map

| Path | Responsibility |
| --- | --- |
| [`main/domain/`](main/domain/) | Models, text rules, classification, scoring, lifecycle, tasks, privacy, claims, reconciliation. |
| [`main/application/`](main/application/) | Memory Core orchestration and maintenance. |
| [`main/storage/`](main/storage/) | JSONL revisions, locks, SQLite index structures, cleanup. |
| [`main/graph/`](main/graph/) | Temporal entities/relations, provenance, historical and multi-hop queries. |
| [`main/retrieval/`](main/retrieval/) | Keyword ranking, BGE embeddings, hybrid search, reranking, context, answers. |
| [`main/ml.py`](main/ml.py), [`main/neural.py`](main/neural.py), [`training/`](training/) | Optional retention scorers and training. [`training/experiments/`](training/experiments/) holds the three-dataset experiment, which the system does not use. |
| [`api/`](api/), [`static/`](static/) | Authenticated HTTP API, monitoring, privacy operations, demo. |
| [`evaluation/`](evaluation/), [`docs/reports/`](docs/reports/) | Datasets, runners, saved metrics, failure cases, review forms. |
| [`tests/`](tests/) | Automated behavior and integration tests. |

## Limits and next steps

- **Generalization:** the controlled benchmark and classification cases were authored by an AI and are not human-validated, and its dev and test personas share templates. LoCoMo is generated dialogue, and its first three conversations were used to choose the reranker pool. Fresh conversation-disjoint data and independent human ratings are needed for a broader claim.
- **Known gaps:** correction and latest-update questions are still often answered with an older statement; abstention is opt-in and catches only "attribute never stated" questions; category classification is 53% on the 60 cases; the larger reranker pool costs about 3–4× search latency; approximate search remains experimental. One lifecycle check and one task test fail rarely because of a wall-clock race in task updates.
- **Answer quality:** saved RAG scores use extractive answers. Live Gemini correctness, personalization, citation usefulness, and conflict decisions still need independent review. Exact-quote citation validation alone cannot prove semantic truth.
- **Extraction and retention:** graph extraction and indirect contradiction handling cover supported language patterns. Optional retention models learn fact validity/duration rather than human importance; the heuristic remains the default.
- **Operations:** local JSONL and SQLite are not a production database architecture. Deployment needs backups, retention-policy review, stronger authorization/abuse controls, and workload-specific scale and latency measurements.

For more implementation detail and day-to-day workflows, see [DOCUMENTATION.md](DOCUMENTATION.md). For evaluation protocols, see [evaluation/README.md](evaluation/README.md) and the saved reports linked above.
