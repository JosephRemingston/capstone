import { Link, useParams } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { ErrorBanner } from '@/components/ErrorBanner'
import { CategoryBadge, StatusBadge, TierBadge } from '@/components/memory/Badges'
import { ImportanceBreakdown } from '@/components/memory/ImportanceBreakdown'
import { useHistory, useMemory } from '@/api/hooks'
import { formatDate, label } from '@/lib/format'

function Field({ name, children }: { name: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[10rem_1fr] gap-3 py-1.5 text-sm">
      <dt className="text-muted-foreground">{name}</dt><dd className="min-w-0 break-words">{children}</dd>
    </div>
  )
}

const MemoryLink = ({ id }: { id: string }) => <Link to={`/memories/${id}`} className="font-mono text-xs underline">{id.slice(0, 8)}</Link>

export function MemoryDetail() {
  const { id = '' } = useParams()
  const memory = useMemory(id)
  const history = useHistory(id)
  const record = memory.data

  return (
    <div className="space-y-4">
      <Link to="/memories" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"><ArrowLeft className="size-4" />Memories</Link>
      <ErrorBanner error={memory.error} />
      {memory.isLoading && <Skeleton className="h-48 w-full" />}
      {record && (
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">{record.content}</CardTitle>
              <div className="flex flex-wrap gap-1.5"><CategoryBadge category={record.category} /><TierBadge tier={record.tier} /><StatusBadge status={record.memory_status} /></div>
            </CardHeader>
            <CardContent>
              <dl className="divide-y">
                <Field name="Created">{formatDate(record.created_at)}</Field>
                <Field name="Expires">{record.expires_at ? formatDate(record.expires_at) : 'never'}</Field>
                <Field name="Archive after">{formatDate(record.archive_after)}</Field>
                <Field name="Session · role">{record.session_id} · {record.role}</Field>
                {record.claim && <Field name="Claim">{record.claim.display_subject} {label(record.claim.predicate)}{record.claim.positive ? '' : ' (not)'} {record.claim.display_value}</Field>}
                {record.task_status && <Field name="Task">{label(record.task_status)}{record.due_at && `, due ${formatDate(record.due_at)}`}{record.next_due_at && `, next ${formatDate(record.next_due_at)}`}</Field>}
                {record.superseded_by && <Field name="Superseded by"><MemoryLink id={record.superseded_by} /></Field>}
                {record.conflict_ids.length > 0 && <Field name="Conflicts with"><span className="flex gap-2">{record.conflict_ids.map(c => <MemoryLink key={c} id={c} />)}</span>{record.conflict_resolution?.reason && <span className="ml-2 text-muted-foreground">resolved by {label(String(record.conflict_resolution.reason))}</span>}</Field>}
                {record.evidence_ids.length > 0 && <Field name="Evidence"><span className="flex flex-wrap gap-2">{record.evidence_ids.map(e => <MemoryLink key={e} id={e} />)}</span></Field>}
                <Field name="ID"><span className="font-mono text-xs">{record.id}</span></Field>
              </dl>
            </CardContent>
          </Card>
          <div className="space-y-4">
            <Card><CardContent className="pt-4"><ImportanceBreakdown record={record} /></CardContent></Card>
            <Card>
              <CardHeader><CardTitle className="text-base">Revision history</CardTitle></CardHeader>
              <CardContent>
                <ErrorBanner error={history.error} />
                <ol className="space-y-3 border-l pl-4">
                  {history.data?.map((revision, index) => (
                    <li key={index} className="text-sm">
                      <p className="font-medium">{label(revision.memory_status)} · {label(revision.tier)}{revision.task_status && ` · task ${label(revision.task_status)}`}</p>
                      <p className="text-xs text-muted-foreground">{formatDate(revision.updated_at ?? revision.recorded_at ?? revision.created_at)}</p>
                    </li>
                  ))}
                </ol>
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  )
}
