import { describe, expect, it } from 'vitest'
import { metricTotal } from './metrics'

const TEXT = `# HELP cognimem_requests_total Requests
# TYPE cognimem_requests_total counter
cognimem_requests_total{method="GET",status="200"} 3.0
cognimem_requests_total{method="POST",status="201"} 2.0
cognimem_requests_total_created 1.7e9
cognimem_memories_created_total 5.0
`
describe('metricTotal', () => {
  it('sums labelled samples of one family only', () => {
    expect(metricTotal(TEXT, 'cognimem_requests_total')).toBe(5)
    expect(metricTotal(TEXT, 'cognimem_memories_created_total')).toBe(5)
  })
  it('returns null when absent', () => {
    expect(metricTotal(TEXT, 'cognimem_missing_total')).toBeNull()
    expect(metricTotal(undefined, 'x')).toBeNull()
  })
})
