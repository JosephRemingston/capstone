# Metric-improvement work: progress and how to resume

Paused 12 Oct 2026, ~03:40 IST. Every background job was stopped deliberately: no workflow, measurement run or watcher is running.

Full plan: `~/.claude/plans/ok-now-remove-all-reactive-pinwheel.md`. It has phases A–F and a progress log at the end.
Saved work files: `docs/progress/` (git-ignored, local only):
- `designs/` holds every fix design and its adversarial verdict.
- `abstention_author/` holds the diversified unanswerable seeds and the dev-only abstention set.
- `workflow_scripts/` holds the workflow scripts.
- `reaper.py` is the leftover-process killer.

## Standing decisions

- The `evaluation/` package was rebuilt from scratch. Compare only against tagged snapshots (`evaluation/reports/<tag>_summary.json`).
- No hosted Gemini calls until the user comes back to Gemini. Use the extractive generator only.
- Tune only on the dev split (personas p01–p03). Report every number from the test split (p04–p12) and LoCoMo.
- Run one heavy evaluation job at a time, and kill leftover processes when a task finishes; the Mac overheats.

## Done

| Step | What | Where |
| --- | --- | --- |
| A | Clean slate (294 MB of regenerable caches, models and reports deleted; user documents kept) | `scripts/clean_slate.py` |
| B | New harness, 12-persona benchmark (456 memories, 360 queries, dev 90 / test 270), 60 classification cases | `evaluation/`, `evaluation/datasets/`, `tests/test_evaluation_*.py` |
| C | Retention models retrained; they reproduce exactly (XGBoost macro-F1 0.6114, deep 0.6732, shallow 0.6504) | `artifacts/importance*` |
| D | Baseline snapshot (exact search, pool 20) | `docs/reports/baseline/`, `evaluation/reports/baseline_summary.json` |
| Bug | `search(mode='recency')` always raised | `main/retrieval/rag.py` |
| F3 | Retention classifier: user self-disclosure is kept 90 days. LongMemEval retention 21.0% → 90.6% (held-out 90.4%) | `main/domain/{text_rules,classification,lifecycle}.py` |
| F1 | Exact search by default; LSH ANN corrected and opt-in (experimental); numpy exact scan; index cache; config errors surface | `main/storage/indexes.py`, `main/retrieval/rag.py`, `api/` |
| F2 | Reranker pool 20 → 100 (`default_pool`); fusion-only and fallback order unchanged (fused at the old depth, deeper candidates appended); 600-character cap removed | `main/retrieval/rag.py`, `reranker.py` |
| F5 | Context keeps the reranker order; hits carry `cosine` and `rank`; context has `scores` and `reranker`; new metric `correct_first` | `main/retrieval/rag.py`, `evaluation/metrics.py` |
| F4 | Graph- and time-aware reordering after the reranker: graph promotion, chain partners, and a stale-to-successor swap gated by relevance band plus cued predicate. The `versions` step is opt-in | `main/retrieval/rag.py`, `tests/test_structured_rerank.py` |
| Harness | `--structure default\|none\|<steps>` switch for attribution (ablation, locomo, measure_all) | `evaluation/`, `scripts/measure_all.py` |

Tests: 196 pass (`.venv/bin/python -m pytest -q -p no:cacheprovider`). `tests/test_conversation_task_extensions.py` has a known rare wall-clock flake.

Evidence from the designs and verdicts (dev and held-out runs by the design agents; not yet re-measured with `measure_all`):
- **F2, LoCoMo:** hybrid_reranked Recall@5 .561 → .669 on conversations 1–3, and .554 → .658 on conversations 4–5, which were never used for tuning.
- **F4, benchmark test split (hybrid_full):** Recall@5 93.0 → 96.1, top-1 77.3 → 89.1, multi-hop Recall@5 68.2 → 93.2. These numbers predate the verifier's gating fixes, so re-measure them.

## Remaining, in order

1. **Measure F1+F2+F5 and F4** (this was running when paused):
   ```bash
   .venv/bin/python scripts/measure_all.py --tag e2 --candidate-limit default --structure none --skip-retention
   .venv/bin/python scripts/measure_all.py --tag e4 --candidate-limit default --skip-retention
   .venv/bin/python scripts/report_delta.py baseline e2
   .venv/bin/python scripts/report_delta.py baseline e4
   ```
   Each takes about 35 min. Run them one after the other, never in parallel.
   Optional side arm: `--structure graph,validity,chain,versions` on dev only, to decide on the opt-in `versions` step.
2. **Rework abstention (F6).** The first design was judged FLAWED (`docs/progress/designs/F6_verdict.json`): its gain came from two benchmark templates, and it scored LoCoMo 49.1% and LongMemEval 54.0% balanced.
   - The rework workflow had finished its author stage: `docs/progress/abstention_author/` holds diversified seeds for the 30 unanswerable queries plus `abstention_dev.json`.
   - The rework was stopped during its two design stages (slot rule v2, local ONNX answerability model).
   - To resume, re-run `docs/progress/workflow_scripts/f6-abstention-rework-wf_c1883e15-aea.js`. In a new session, its scratch paths (`/private/tmp/...`) must be recreated from `docs/progress/`, because the old scratch dir may be gone. Re-run the author stage, or point the designers at the saved author output.
   - Adoption rule: default-on only if answerable acceptance is ≥95% on the benchmark and ≥90% on the external sets, and balanced accuracy beats no-abstention by ≥3 points on both LongMemEval and LoCoMo. Otherwise ship it opt-in.
   - Merging the diversified unanswerable seeds changes no baseline number, because extractive never abstains. Rebuild with `python -m evaluation.build_dataset`.
3. **Measure abstention:** `measure_all.py --tag e6 ... --abstain-threshold <tau>`, with the threshold chosen on dev only.
4. **Deferred by the user (12 Oct 2026, 16:10): optional experiments (plan E5–E8).** Not part of this pass: context K 10 and query expansion for multi-hop; a reranker swap (MiniLM-L12, bge-reranker-base); an embedder swap (bge-base, arctic-embed-m); dialogue graph extraction. Keep each only if it passes its adoption rule.
5. **Final adversarial code-review workflow,** then the full test suite.
6. **Docs:**
   - README "before and after" section (user request): original and baseline metrics, then one row per fix with the file, the mechanism and the measured delta. Every number comes from `evaluation/reports/<tag>_summary.json`.
   - Also update DOCUMENTATION, PROJECT_VISION and system_design, and de-duplicate `requirements.txt`.
   - Known gap to report: category classification is 53.3% on the 60 cases.
7. Commit only when the user asks.

## Update 12 Oct 2026, ~17:15 IST

- Measured: `e2` and `e4` with the earlier harness. Both were superseded because the final code review found that pool-100 deep lists were fused at a different depth than product calls.
- Abstention rework judged: slot rule v2 ships opt-in (`ask --abstain`, library default off); the QNLI answerability model is not shipped (see `docs/progress/designs/F6_judge.json`). Merged with the judge's adjustments, plus the reviewer's fixes for "is called <Name>", `--preview` warnings, a hermetic test and an expected-failure limitation. The diversified unanswerable seeds were merged; gold sha256 is `a93ca70f…0f25`.
- Final code review (`docs/progress/final_review.json`): 14 confirmed findings, all fixed.
  - Snapshots now contain only this run's outputs, record reused reports, argv, diff hash and step commands, and refuse to overwrite a tag.
  - `fusion_limit` makes harness deep lists match product order.
  - Successor injection checks the truncated pool.
  - Abstention: stemmer, "how much time", non-amount money nouns.
  - Recall questions are not disclosures; 90-day retention requires self-disclosure.
  - Per-arm pools, vector path counts, LoCoMo held-out split, and `report_delta` dataset/protocol checks.
  - Stale ANN and retention sections were removed from the `e2`/`e4`/`e6` tracked summaries.
- `requirements.txt` de-duplicated; evaluation/README updated. 232 tests pass (1 expected failure).
- Running (17:01): `measure_all` tags `final` (all defaults, `--ann-arms`, retention), then `e2r` (`--structure none`), then `final_abstain` (`--abstain-threshold 0.0`). Log: `scratchpad/measure_final.log`.
- Left after that: README before/after section and results rewrite, DOCUMENTATION/PROJECT_VISION/system_design touch-ups, and quiet-machine latency.

## Completed 12 Oct 2026, ~19:30 IST

- Final snapshots: `final` (all fixes, shipped defaults, `--ann-arms`, retention), `e2r` (search fixes only; LoCoMo and retention reused from `final`, recorded in the manifest), and `final_abstain` (opt-in abstention). Its lifecycle step hit the known rare wall-clock flake once; it passed in `final` and in 6 re-runs.
- README: a new "Before and after" section plus current results; architecture text and limits updated. DOCUMENTATION, PROJECT_VISION and system_design were updated and their dead links removed. `evaluation/reports/abstention_external.json` holds the external abstention numbers; `baseline_summary.json` gained the derived LoCoMo held-out section.
- 232 tests pass (1 expected failure); dataset check passes; nothing is running.
- Not done, by user decision: optional experiments E5–E8, the Gemini evaluation and the QNLI answerability model (not shipped). Nothing is committed yet.
