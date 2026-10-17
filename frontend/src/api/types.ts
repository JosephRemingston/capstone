// Shapes of the CogniMem HTTP API (api/app.py), captured from live responses.

export type Category = 'semantic' | 'episodic' | 'procedural' | 'preference' | 'task' | 'temporary'
export type Tier = 'working' | 'short_term' | 'long_term' | 'archive'
export const CATEGORIES: Category[] = ['semantic', 'episodic', 'procedural', 'preference', 'task', 'temporary']
export const TIERS: Tier[] = ['working', 'short_term', 'long_term', 'archive']

export interface User { id: string; email: string; role: string; created_at: string }
export interface AuthResult { access_token: string; token_type: 'bearer'; expires_in: number; user: User }

/** Provenance of the importance score: `source_metadata.importance_model`. */
export interface ImportanceModel {
  type: 'hybrid' | 'heuristic' | string
  model_weight?: number
  model_class?: 'low' | 'mid' | 'high'
  model_probabilities?: [number, number, number]
  model_score?: number
  small_talk_penalty?: boolean
  heuristic_score?: number
  model_sha256?: string
  encoder?: string
}

export interface Claim { subject: string; predicate: string; value: string; positive: boolean; display_value: string; display_subject: string }

export interface MemoryRecord {
  id: string
  content: string
  user_id: string
  session_id: string
  role: string
  category: Category
  tier: Tier
  importance_score: number
  confidence: number
  created_at: string
  updated_at: string | null
  recorded_at: string | null
  expires_at: string | null
  archive_after: string | null
  memory_status: 'active' | 'superseded' | 'consolidated' | string
  superseded_by: string | null
  consolidated_into: string | null
  conflict_ids: string[]
  conflict_resolution: { reason?: string; [key: string]: unknown } | null
  claim: Claim | null
  evidence_ids: string[]
  task_status: string | null
  due_at: string | null
  next_due_at: string | null
  recurrence: Record<string, unknown> | null
  summary: string | null
  source_metadata: { importance_model?: ImportanceModel; privacy?: { redacted: boolean; types: string[] }; [key: string]: unknown }
  features: Record<string, number>
}

export type SearchMode = 'keyword' | 'semantic' | 'graph' | 'hybrid' | 'recency'

/** A RAG hit (non-keyword modes). Keyword mode returns plain MemoryRecords. */
export interface SearchHit {
  memory_id: string
  text: string
  rank: number
  score: number
  fusion_score?: number
  reranker_score?: number | null
  cosine?: number | null
  channels?: string[]
  signals?: Record<string, { rank: number; contribution: number }>
  reranker?: { status: string; model: string | null; error_type: string | null; score?: number | null }
  timings_ms?: Record<string, number>
  paths?: unknown[]
  record?: MemoryRecord
}

export interface AskStatement { text: string; evidence: { source_id: string; quote: string }[] }
export interface AskResult {
  answer: string
  abstain: boolean
  reason?: string
  statements: AskStatement[]
  sources: Record<string, { id: string; observed_at: string; text: string }>
  generator: 'extractive' | 'gemini'
  reranker?: { status: string; model: string | null } | null
}

export interface GraphRelation {
  id: string
  subject: string
  predicate: string
  object: string
  positive: boolean
  source_id: string
  valid_from: string
  valid_to: string | null
  known_from: string
  known_to: string | null
  confidence: number
  subject_label: string
  subject_type: string
  object_label: string
  object_type: string
  evidence_ids: string[]
}
export interface GraphEntity { id: string; type: string; label: string }
export interface GraphResult { relations: GraphRelation[]; entities?: GraphEntity[]; paths?: { nodes: string[]; edges: GraphRelation[] }[] }

export interface Health { status: string; checks: Record<string, string>; importance_scorer?: { type: string; model_sha256?: string } }

export interface SnapshotInfo { tag: string; created_at: string | null; git_head: string | null; scorer: string | null; sections: string[]; metrics: number }
export interface Snapshot {
  tag: string
  manifest: { tag?: string; created_at?: string; git_head?: string; dirty_tree?: boolean; dataset_sha256?: string; scorer?: string | null;
    steps?: { step: string; exit_code: number; seconds: number }[]; reused?: Record<string, unknown> }
  flat: Record<string, number | null>
}
export interface ClassificationReport {
  accuracy: number; balanced_accuracy: number; macro_f1: number; weighted_f1: number
  confusion_matrix: number[][]; per_class: Record<string, { precision: number; recall: number; f1: number; support: number }>
}
export interface ModelEvaluation { test: ClassificationReport; row_ci95: [number, number]; sentence_ci95: [number, number] }
export interface ModelReport {
  generated_at: string
  model: string
  class_names: string[]
  class_values: number[]
  resamples: number
  dataset: { name: string; n_rows: number; bin_edges: number[]; split_sizes: Record<string, number>; split_sentences: Record<string, number>;
    diagnostics: { unique_sentences: number; score_range: [number, number]; words_per_text: { mean: number; max: number } } }
  tuned: { w_xgb: number; class_weights: number[]; xgb_temperature: number; ensemble_temperature: number; per_seed_temperature: Record<string, number>;
    val_macro_f1_before: number; val_macro_f1_after: number }
  ensemble_thresholds: ModelEvaluation & { validation_macro_f1: number }
  majority: ModelEvaluation & { class_name: string }
  oracle: ModelEvaluation
  serving_checks: Record<string, { rows?: number; class_mismatches: number; max_abs_probability_diff: number }>
  caveats: string[]
}
