// Metric definitions for the Evaluation page: which snapshot keys make up each layer's table, and the snapshot comparison.
// Values are read from evaluation/reports/<tag>_summary.json through the API; nothing is typed in by hand.

export type Flat = Record<string, number | null | undefined>
export type Format = 'pct' | 'num' | 'count' | 'ms' | 'ratio'

export interface MetricRow {
  label: string
  keys: string[]                       // first key present wins (e.g. a metric renamed between snapshots)
  format: Format
  /** Snapshot the Final column reads from (default 'final'), e.g. 'final_abstain' for the opt-in abstention rows. */
  finalTag?: string
  /** Snapshot the Final column falls back to when `final` did not record the metric; shown with an asterisk. */
  finalFallback?: string
  /** Second key for "a / b" rows such as lifecycle checks passed / total. */
  of?: string
  note?: (flats: Record<string, Flat>) => string | null
}

export interface LayerTable {
  id: string
  title: string
  description: string
  heading?: string
  rows: MetricRow[]
  groups?: { heading: string; rows: MetricRow[] }[]
}

const pct = (v: number | null | undefined) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)

export function format(value: number | null | undefined, kind: Format): string {
  if (value == null || Number.isNaN(value)) return '—'
  switch (kind) {
    case 'pct': return `${(value * 100).toFixed(1)}%`
    case 'num': return value.toFixed(3)
    case 'count': return Math.round(value).toLocaleString()
    case 'ms': return `${Math.round(value)} ms`
    case 'ratio': return String(value)
  }
}

export function pick(flat: Flat | undefined, keys: string[]): number | null {
  if (!flat) return null
  for (const key of keys) {
    const value = flat[key]
    if (value !== undefined && value !== null) return value
  }
  return null
}

export interface Cells { baseline: string; final: string; current: string; same: boolean }

/**
 * The three columns of a layer table. Current is the `hybrid` snapshot (shipped default, 16 Oct 2026) when it measured
 * the metric; the benchmark and LoCoMo runners never call the importance scorer, so for those Current equals Final.
 */
export function cells(row: MetricRow, flats: Record<string, Flat>): Cells {
  const show = (tag: string) => {
    const value = pick(flats[tag], row.keys)
    if (value == null) return '—'
    if (row.of) {
      const total = pick(flats[tag], [row.of])
      return `${format(value, 'count')}/${format(total, 'count')}`
    }
    return format(value, row.format)
  }
  let final = show(row.finalTag ?? 'final')
  if (final === '—' && row.finalFallback && pick(flats[row.finalFallback], row.keys) != null) final = `${show(row.finalFallback)}*`
  const measured = pick(flats.hybrid, row.keys) != null && !row.finalTag
  return { baseline: show('baseline'), final, current: measured ? show('hybrid') : final, same: !measured }
}

const H = 'ablation.hybrid_full'
const L = 'locomo_retrieval.hybrid_reranked'

export const LAYERS: LayerTable[] = [
  {
    id: 'layer1', title: 'Layer 1 · Understand and structure', description: 'Redaction, splitting, six rule-based categories, features and claims.',
    heading: '60 classification cases (AI-authored)',
    rows: [
      { label: 'Category accuracy', keys: ['classification.category_accuracy'], format: 'pct' },
      { label: 'Category macro F1', keys: ['classification.category_macro_f1'], format: 'num' },
    ],
  },
  {
    id: 'layer2', title: 'Layer 2 · Score, retain, manage change',
    description: 'Importance = 25 % csv5k ensemble + 75 % rules; tiers, expiry, conflicts, tasks and cleanup.',
    heading: 'Retention and lifecycle',
    rows: [
      { label: 'LongMemEval user-fact retention (457 examples)', keys: ['longmemeval.all.memory_retention'], format: 'pct' },
      { label: 'Mean share of a conversation still visible', keys: ['longmemeval.all.mean_visible_fraction'], format: 'pct' },
      { label: 'Cleanup: expired memories deleted', keys: ['longmemeval._cleanup.removed_memories'], format: 'count' },
      {
        label: 'Cleanup: expired memories kept as archive', keys: ['longmemeval._cleanup.archived_memories'], format: 'count',
        finalFallback: 'heuristic_control',
        note: () => '* Not recorded on 12 Oct; taken from the same-day rule-based rerun (heuristic_control), which reproduces final on all 457 examples.',
      },
      {
        label: 'Classification cases: tier accuracy', keys: ['classification.tier_accuracy'], format: 'pct',
        note: f => (f.heuristic_control ? `The rise is the runner's clock pin, not the model: the same-day rule-based run also gives ${pct(pick(f.heuristic_control, ['classification.tier_accuracy']))}.` : null),
      },
      {
        label: 'Classification cases: score inside the labelled range', keys: ['classification.importance_range_agreement'], format: 'pct',
        note: f => {
          const constant = pick(f.hybrid, ['classification._range_baselines.constant.0.5'])
          return constant == null ? null : `Not better judgement: a constant score of 0.5 lands inside ${pct(constant)} of the labelled ranges.`
        },
      },
      { label: 'Lifecycle, task and supersession checks', keys: ['lifecycle.passed'], of: 'lifecycle.checks', format: 'count' },
    ],
  },
  {
    id: 'layer3', title: 'Layer 3 · Persist memories and time', description: 'JSONL revision log, SQLite vectors and temporal graph. Exact search is the default; these rows force approximate (LSH) search on.',
    heading: 'Approximate search forced on',
    rows: [
      { label: 'Benchmark Recall@5', keys: ['ablation_ann.hybrid_full.recall_at_5'], format: 'pct' },
      { label: 'Benchmark: gold memory in the reranker pool', keys: ['ablation_ann.hybrid_full.pool_hit'], format: 'pct' },
      { label: 'LoCoMo Recall@5', keys: ['locomo_retrieval_ann.hybrid_reranked.recall_at_5'], format: 'pct' },
      { label: 'LoCoMo: gold evidence in the reranker pool', keys: ['locomo_retrieval_ann.hybrid_reranked.pool_hit'], format: 'pct' },
    ],
  },
  {
    id: 'layer4', title: 'Layer 4 · Retrieve and rank', description: 'Keyword, vector and graph channels, rank fusion, cross-encoder reranker and time/link-aware reordering.',
    rows: [],
    groups: [
      {
        heading: 'Benchmark test split · Recall@5 by arm',
        rows: [
          { label: 'Vector', keys: ['ablation.vector.recall_at_5'], format: 'pct' },
          { label: 'Vector + reranker', keys: ['ablation.vector_reranker.recall_at_5'], format: 'pct' },
          { label: 'Vector + graph', keys: ['ablation.vector_graph.recall_at_5'], format: 'pct' },
          { label: 'Vector + graph + reranker', keys: ['ablation.vector_graph_reranker.recall_at_5'], format: 'pct' },
          { label: 'Hybrid + reranker (shipped)', keys: [`${H}.recall_at_5`], format: 'pct' },
        ],
      },
      {
        heading: 'Benchmark test split · hybrid + reranker (247 answerable)',
        rows: [
          { label: 'Recall@10', keys: [`${H}.recall_at_10`], format: 'pct' },
          { label: 'MRR', keys: [`${H}.mrr`], format: 'num' },
          { label: 'nDCG@10', keys: [`${H}.ndcg_at_10`], format: 'num' },
          { label: 'Correct memory first', keys: [`${H}.correct_first_answerable`, `${H}.correct_top1_answerable`], format: 'pct' },
          { label: 'Gold memory in the reranker pool', keys: [`${H}.pool_hit`], format: 'pct' },
          { label: 'Search latency P50', keys: [`${H}.p50_ms`], format: 'ms' },
        ],
      },
      {
        heading: 'LoCoMo evidence retrieval (1,973 questions)',
        rows: [
          { label: 'Recall@5', keys: [`${L}.recall_at_5`], format: 'pct' },
          { label: 'Recall@5 on held-out conversations', keys: ['locomo_retrieval_held_out.hybrid_reranked.recall_at_5'], format: 'pct' },
          { label: 'Hit@5', keys: [`${L}.hit_at_5`], format: 'pct' },
          { label: 'Recall@10', keys: [`${L}.recall_at_10`], format: 'pct' },
          { label: 'MRR@5', keys: [`${L}.mrr_at_5`], format: 'num' },
          { label: 'Gold evidence in the reranker pool', keys: [`${L}.pool_hit`], format: 'pct' },
          { label: 'Multi-hop Recall@5', keys: [`${L}.multi-hop_recall_at_5`], format: 'pct' },
          { label: 'Graph channel alone, Recall@5', keys: ['locomo_retrieval.graph.recall_at_5'], format: 'pct' },
          { label: 'Search latency P50', keys: [`${L}.p50_ms`], format: 'ms' },
        ],
      },
    ],
  },
  {
    id: 'layer5', title: 'Layer 5 · Build context and answer', description: 'Bounded context in reranked order, extractive or Gemini answers, citation checks and opt-in abstention.',
    heading: 'Benchmark test split · hybrid + reranker',
    rows: [
      { label: 'Grounded correctness, answerable questions', keys: [`${H}.correct_grounded_answerable`], format: 'pct' },
      { label: 'Grounded correctness, all questions', keys: [`${H}.correct_grounded`], format: 'pct' },
      { label: 'Unanswerable refused (opt-in abstention)', keys: [`${H}.unanswerable_refusal_rate`], format: 'pct', finalTag: 'final_abstain' },
      { label: 'Answerable still answered (opt-in abstention)', keys: [`${H}.answerable_accept_rate`], format: 'pct', finalTag: 'final_abstain' },
      { label: 'Balanced answerability (opt-in abstention)', keys: [`${H}.balanced_answerability`], format: 'pct', finalTag: 'final_abstain' },
    ],
  },
]

export const LAYER_TAGS = ['baseline', 'final', 'final_abstain', 'hybrid', 'heuristic_control']

export function allRows(layer: LayerTable): MetricRow[] {
  return [...layer.rows, ...(layer.groups ?? []).flatMap(group => group.rows)]
}

// --- snapshot comparison (the same rules as scripts/report_delta.py) ---------------------------------------------------

const NOT_METRICS = ['._protocol.', '._vector_paths.']
export const isMetric = (key: string) => !key.startsWith('_scorers.') && !NOT_METRICS.some(marker => key.includes(marker))

export interface DeltaRow { key: string; before: number; after: number; delta: number }
export interface Comparison { rows: DeltaRow[]; protocol: { key: string; before: number | null | undefined; after: number | null | undefined }[]; onlyOne: number }

export function compare(before: Flat, after: Flat, filter = '', changedOnly = false): Comparison {
  const terms = filter.toLowerCase().split(/\s+/).filter(Boolean)
  const matches = (key: string) => terms.every(term => key.toLowerCase().includes(term))
  const numeric = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
  const keys = new Set([...Object.keys(before), ...Object.keys(after)])
  const rows: DeltaRow[] = []
  let onlyOne = 0
  for (const key of [...keys].sort()) {
    if (!isMetric(key) || !matches(key)) continue
    const a = before[key], b = after[key]
    if (numeric(a) && numeric(b)) {
      const delta = b - a
      if (!changedOnly || Math.abs(delta) > 1e-12) rows.push({ key, before: a, after: b, delta })
    } else if ((key in before) !== (key in after)) {
      onlyOne += 1
    }
  }
  const sections = new Set(rows.map(row => row.key.split('.')[0]))
  const protocol = [...keys].sort()
    .filter(key => key.includes('._protocol.') && before[key] !== after[key] && (!terms.length || sections.has(key.split('.')[0])))
    .map(key => ({ key, before: before[key], after: after[key] }))
  return { rows, protocol, onlyOne }
}

/** "ablation: resolved_pool; locomo_retrieval: candidate_limit" — which settings differ, grouped by section. */
export function settingsSummary(protocol: Comparison['protocol']): string {
  const bySection = new Map<string, Set<string>>()
  for (const { key } of protocol) {
    const [section, , setting] = key.split('.')
    if (!bySection.has(section)) bySection.set(section, new Set())
    bySection.get(section)!.add(setting)
  }
  return [...bySection].map(([section, settings]) => `${section}: ${[...settings].join(', ')}`).join('; ')
}

/** +1 when a higher value is better, -1 when lower is better (latency, storage), 0 for plain counts. */
export function direction(key: string): -1 | 0 | 1 {
  const last = key.split('.').pop() ?? ''
  if (/_ms$|^bytes_|latency/.test(last)) return -1
  if (/^(queries|answerable_queries|questions|examples|with_gold_turns|cases|checks|turns|removed_memories|archived_memories|visible_state_preserved|pairs)$/.test(last)) return 0
  return 1
}
