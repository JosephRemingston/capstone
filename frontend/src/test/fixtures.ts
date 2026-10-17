import type { AskResult, MemoryRecord, SearchHit } from '@/api/types'

export const record = (overrides: Partial<MemoryRecord> = {}): MemoryRecord => ({
  id: '11111111-aaaa-bbbb-cccc-000000000001', content: 'I prefer concise technical explanations.', user_id: 'u', session_id: 's', role: 'user',
  category: 'preference', tier: 'long_term', importance_score: 0.6846, confidence: 1, created_at: '2026-10-17T20:00:00+00:00',
  updated_at: null, recorded_at: null, expires_at: null, archive_after: '2027-01-15T20:00:00+00:00', memory_status: 'active',
  superseded_by: null, consolidated_into: null, conflict_ids: [], conflict_resolution: null, claim: null, evidence_ids: [],
  task_status: null, due_at: null, next_due_at: null, recurrence: null, summary: null, features: {},
  source_metadata: { importance_model: { type: 'hybrid', model_weight: 0.25, model_class: 'high', model_probabilities: [0.0, 0.014, 0.986],
    model_score: 0.8372, small_talk_penalty: false, heuristic_score: 0.6337, model_sha256: 'abc', encoder: 'bge' } },
  ...overrides,
})

// Captured from POST /api/v1/memories/search (hybrid) on 18 Oct 2026, trimmed.
export const hit: SearchHit = {
  memory_id: '032104fa-4d72-4ca5-bf5f-c71b2b480d33', text: 'I moved to Bengaluru.', rank: 1, score: 0.0204, fusion_score: 0.0204,
  reranker_score: -2.997, cosine: 0.549, channels: ['keyword', 'semantic', 'graph'],
  signals: { keyword: { rank: 1, contribution: 0.0049 }, semantic: { rank: 1, contribution: 0.0082 } },
  reranker: { status: 'applied', model: 'Xenova/ms-marco-MiniLM-L-6-v2', error_type: null, score: -2.997 },
  timings_ms: { filter: 0.2, keyword: 0.1, vector: 359.9, graph: 3.9, fusion: 0.6, reranker: 83.2, total: 447.8 },
}

export const answer: AskResult = {
  answer: 'I moved to Bengaluru.', abstain: false, generator: 'extractive', reranker: { status: 'applied', model: 'x' },
  statements: [{ text: 'I moved to Bengaluru.', evidence: [{ source_id: '032104fa-4d72-4ca5-bf5f-c71b2b480d33', quote: 'I moved to Bengaluru.' }] }],
  sources: { '032104fa-4d72-4ca5-bf5f-c71b2b480d33': { id: '032104fa-4d72-4ca5-bf5f-c71b2b480d33', observed_at: '2026-10-17T20:16:38+00:00', text: 'I moved to Bengaluru.' } },
}
