import type { MemoryRecord } from '@/api/types'

/** Recomputes the stored score from its parts: 0.25 × model class value + 0.75 × rule score (main/ensemble.py). */
export function importanceParts(record: MemoryRecord) {
  const stamp = record.source_metadata.importance_model
  if (!stamp || stamp.type !== 'hybrid' || stamp.model_score == null || stamp.heuristic_score == null) return null
  const weight = stamp.model_weight ?? 0.25
  const recomputed = Math.round((weight * stamp.model_score + (1 - weight) * stamp.heuristic_score) * 1e4) / 1e4
  return { stamp, weight, recomputed, matches: Math.abs(recomputed - record.importance_score) < 1e-4 }
}
