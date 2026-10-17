import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import { LAYERS, LAYER_TAGS, allRows, cells, compare, direction, pick, settingsSummary, type Flat } from './evaluation'

// The tracked snapshot files the API serves: the page must find every metric it lists.
const REPORTS = path.resolve(import.meta.dirname, '../../../evaluation/reports')
const flats: Record<string, Flat> = Object.fromEntries(LAYER_TAGS.map(tag =>
  [tag, JSON.parse(fs.readFileSync(path.join(REPORTS, `${tag}_summary.json`), 'utf8')).flat as Flat]))

describe('layer tables against the real snapshots', () => {
  it('every row resolves in its Final snapshot', () => {
    const missing = LAYERS.flatMap(allRows)
      .filter(row => pick(flats[row.finalTag ?? 'final'], row.keys) == null && pick(flats[row.finalFallback ?? ''], row.keys) == null)
      .map(row => row.label)
    expect(missing).toEqual([])
  })

  it('every scorer-dependent row (Layers 1-2) is measured by the current hybrid snapshot', () => {
    const rows = LAYERS.filter(l => l.id === 'layer1' || l.id === 'layer2').flatMap(allRows)
    expect(rows.filter(row => pick(flats.hybrid, row.keys) == null).map(r => r.label)).toEqual([])
  })

  it('reproduces the README numbers', () => {
    const row = (label: string) => LAYERS.flatMap(allRows).find(r => r.label.startsWith(label))!
    expect(cells(row('LongMemEval user-fact retention'), flats)).toEqual({ baseline: '21.0%', final: '90.6%', current: '90.6%', same: false })
    expect(cells(row('Cleanup: expired memories kept'), flats)).toEqual({ baseline: '—', final: '33*', current: '38', same: false })
    expect(cells(row('Lifecycle'), flats)).toMatchObject({ baseline: '16/16', final: '16/16', current: '16/16' })
    expect(cells(row('Hybrid + reranker (shipped)'), flats)).toEqual({ baseline: '93.0%', final: '97.5%', current: '97.5%', same: true })
    expect(cells(row('Correct memory first'), flats)).toMatchObject({ baseline: '77.3%', final: '84.2%' })
    expect(cells(row('Unanswerable refused'), flats)).toMatchObject({ baseline: '0.0%', final: '17.4%', same: true })
    expect(cells(row('Recall@5'), flats)).toMatchObject({ baseline: '55.8%', final: '66.8%' })
  })

  it('notes cite the same-day rule-based run', () => {
    const tier = LAYERS.flatMap(allRows).find(r => r.label.includes('tier accuracy'))!
    expect(tier.note!(flats)).toContain('56.7%')
  })
})

describe('compare', () => {
  it('lists common metrics with deltas, skips settings and counts one-sided keys', () => {
    const before = { 'a.x': 0.5, 'a._protocol.top_k': 5, 'b.y': 1, 'only.before': 2 }
    const after = { 'a.x': 0.75, 'a._protocol.top_k': 10, 'b.y': 1 }
    const result = compare(before, after)
    expect(result.rows).toEqual([{ key: 'a.x', before: 0.5, after: 0.75, delta: 0.25 }, { key: 'b.y', before: 1, after: 1, delta: 0 }])
    expect(result.protocol).toEqual([{ key: 'a._protocol.top_k', before: 5, after: 10 }])
    expect(result.onlyOne).toBe(1)
  })

  it('filters by all terms, keeps settings warnings for shown sections, and can hide unchanged rows', () => {
    const before = { 'a.x': 0.5, 'a._protocol.top_k': 5, 'b.y': 1 }
    const after = { 'a.x': 0.75, 'a._protocol.top_k': 10, 'b.y': 1 }
    expect(compare(before, after, 'b').protocol).toEqual([])
    expect(compare(before, after, 'a x').rows.map(r => r.key)).toEqual(['a.x'])
    expect(compare(before, after, 'a x').protocol).toHaveLength(1)
    expect(compare(before, after, '', true).rows.map(r => r.key)).toEqual(['a.x'])
  })

  it('matches report_delta on final -> hybrid LongMemEval retention', () => {
    const rows = compare(flats.final, flats.hybrid, 'longmemeval all memory_retention').rows
    expect(rows).toHaveLength(1)
    expect(rows[0].delta).toBe(0)
  })
})

describe('settingsSummary', () => {
  it('groups setting differences by section', () => {
    const protocol = compare(flats.baseline, flats.final).protocol
    expect(protocol.length).toBeGreaterThan(5)
    expect(settingsSummary(protocol)).toBe('ablation: resolved_pool; ablation_ann: resolved_pool; ablation_dev: resolved_pool')
  })
})

describe('direction', () => {
  it('knows which way is better', () => {
    expect(direction('ablation.hybrid_full.recall_at_5')).toBe(1)
    expect(direction('ablation.hybrid_full.p50_ms')).toBe(-1)
    expect(direction('longmemeval._cleanup.bytes_after')).toBe(-1)
    expect(direction('longmemeval._cleanup.removed_memories')).toBe(0)
    expect(direction('ablation.hybrid_full.queries')).toBe(0)
  })
})
