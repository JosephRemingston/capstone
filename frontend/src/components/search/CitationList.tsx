import { Link } from 'react-router-dom'
import type { AskResult } from '@/api/types'
import { formatDate } from '@/lib/format'

/** Each answer statement with the exact quotes and source memories that support it. */
export function CitationList({ result }: { result: AskResult }) {
  if (result.abstain) {
    return (
      <div className="rounded-lg border border-dashed p-4 text-sm" data-testid="abstained">
        <p className="font-medium">No answer: the memories do not contain enough evidence.</p>
        {result.reason && <p className="text-muted-foreground">{result.reason}</p>}
      </div>
    )
  }
  return (
    <ol className="space-y-3">
      {result.statements.map((statement, index) => (
        <li key={index} className="rounded-lg border p-3 text-sm">
          <p>{statement.text}</p>
          <ul className="mt-2 space-y-1">
            {statement.evidence.map(evidence => {
              const source = result.sources[evidence.source_id]
              return (
                <li key={evidence.source_id} className="text-xs text-muted-foreground">
                  “{evidence.quote}” —{' '}
                  <Link to={`/memories/${evidence.source_id}`} className="underline">source {evidence.source_id.slice(0, 8)}</Link>
                  {source && <>, observed {formatDate(source.observed_at)}</>}
                </li>
              )
            })}
          </ul>
        </li>
      ))}
    </ol>
  )
}
