import { Link } from 'react-router-dom'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { CategoryBadge, StatusBadge, TierBadge } from './Badges'
import { ImportanceBreakdown } from './ImportanceBreakdown'
import { formatDate, label, relative } from '@/lib/format'
import type { MemoryRecord } from '@/api/types'

export function MemoryCard({ record, showImportance = true }: { record: MemoryRecord; showImportance?: boolean }) {
  const redacted = record.source_metadata.privacy?.redacted
  return (
    <Card size="sm">
      <CardHeader className="space-y-2">
        <Link to={`/memories/${record.id}`} className="font-medium hover:underline">{record.content}</Link>
        <div className="flex flex-wrap items-center gap-1.5">
          <CategoryBadge category={record.category} />
          <TierBadge tier={record.tier} />
          <StatusBadge status={record.memory_status} />
          {record.task_status && <span className="text-xs text-muted-foreground">task {label(record.task_status)}{record.due_at && `, due ${formatDate(record.due_at)}`}</span>}
          {redacted && <span className="text-xs text-muted-foreground">redacted: {record.source_metadata.privacy!.types.join(', ')}</span>}
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-muted-foreground">
          {record.expires_at ? <>Expires {relative(record.expires_at)} ({formatDate(record.expires_at)})</> : 'Does not expire'}
          {record.conflict_ids.length > 0 && <> · resolved a conflict with {record.conflict_ids.map(id => (
            <Link key={id} to={`/memories/${id}`} className="underline">an earlier memory</Link>))}</>}
        </p>
        {showImportance && <ImportanceBreakdown record={record} />}
      </CardContent>
    </Card>
  )
}
