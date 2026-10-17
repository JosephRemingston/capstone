import type { MemoryRecord } from '@/api/types'
import { num, pct } from '@/lib/format'
import { importanceParts } from '@/lib/importance'

const CLASSES = ['low', 'mid', 'high'] as const

function Bar({ value, className }: { value: number; className: string }) {
  return (
    <div className="h-1.5 w-full rounded-full bg-muted">
      <div className={`h-1.5 rounded-full ${className}`} style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
    </div>
  )
}

export function ImportanceBreakdown({ record }: { record: MemoryRecord }) {
  const parts = importanceParts(record)
  return (
    <div className="space-y-2 text-sm" data-testid="importance">
      <div className="flex items-baseline justify-between">
        <span className="font-medium">Importance</span>
        <span className="font-mono text-base">{num(record.importance_score, 4)}</span>
      </div>
      <Bar value={record.importance_score} className="bg-primary" />
      {parts ? (
        <div className="space-y-1.5 text-xs text-muted-foreground">
          <div className="flex justify-between">
            <span>
              {parts.weight} × model <b className="text-foreground">{parts.stamp.model_class}</b>
              {parts.stamp.small_talk_penalty && ' (small talk × 0.25)'}
            </span>
            <span className="font-mono">{num(parts.stamp.model_score, 4)}</span>
          </div>
          <div className="flex justify-between">
            <span>{1 - parts.weight} × rules</span>
            <span className="font-mono">{num(parts.stamp.heuristic_score, 4)}</span>
          </div>
          {parts.stamp.model_probabilities && (
            <div className="grid grid-cols-3 gap-2 pt-1">
              {CLASSES.map((name, i) => (
                <div key={name}>
                  <div className="flex justify-between"><span>{name}</span><span>{pct(parts.stamp.model_probabilities![i])}</span></div>
                  <Bar value={parts.stamp.model_probabilities![i]} className={parts.stamp.model_class === name ? 'bg-primary' : 'bg-muted-foreground/40'} />
                </div>
              ))}
            </div>
          )}
          {!parts.matches && <p className="text-destructive">Parts do not add up to the stored score ({parts.recomputed}).</p>}
        </div>
      ) : (
        <p className="text-xs text-muted-foreground">Rule-based score only (heuristic scorer).</p>
      )}
    </div>
  )
}
