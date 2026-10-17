import { describe, expect, it } from 'vitest'
import { importanceParts } from './importance'
import { record } from '@/test/fixtures'

describe('importanceParts', () => {
  it('recomputes 0.25 × model + 0.75 × rules and matches the stored score', () => {
    const parts = importanceParts(record())!
    expect(parts.recomputed).toBe(Math.round((0.25 * 0.8372 + 0.75 * 0.6337) * 1e4) / 1e4)
    expect(parts.matches).toBe(true)
  })
  it('flags a stored score that does not match its parts', () => {
    expect(importanceParts(record({ importance_score: 0.9 }))!.matches).toBe(false)
  })
  it('returns null for the heuristic scorer', () => {
    expect(importanceParts(record({ source_metadata: { importance_model: { type: 'heuristic' } } }))).toBeNull()
  })
})
