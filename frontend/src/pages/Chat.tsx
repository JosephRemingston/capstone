import { useState, type FormEvent } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { ErrorBanner, EmptyState } from '@/components/ErrorBanner'
import { MemoryCard } from '@/components/memory/MemoryCard'
import { useCreateMemory } from '@/api/hooks'
import type { MemoryRecord } from '@/api/types'

const EXAMPLES = ['I prefer concise technical explanations.', 'I live in Chennai.', 'I moved to Bengaluru.', 'Remind me to submit the report tomorrow.', 'hello']

export function Chat() {
  const create = useCreateMemory()
  const [content, setContent] = useState('')
  const [session, setSession] = useState('web')
  const [log, setLog] = useState<MemoryRecord[]>([])

  const send = (text: string) => {
    const trimmed = text.trim()
    if (!trimmed) return
    create.mutate({ content: trimmed, session_id: session.trim() || 'web' }, {
      onSuccess: record => { setLog(previous => [record, ...previous]); setContent('') },
    })
  }
  const submit = (event: FormEvent) => { event.preventDefault(); send(content) }

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
      <section className="space-y-4">
        <div>
          <h1 className="text-xl font-semibold">Tell CogniMem something</h1>
          <p className="text-sm text-muted-foreground">Each message is redacted, classified, scored and given a lifetime. The card on the right shows how it was understood.</p>
        </div>
        <form onSubmit={submit} className="space-y-3">
          <Textarea aria-label="Message" rows={4} placeholder="I prefer concise technical explanations." value={content}
            onChange={e => setContent(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) send(content) }} />
          <div className="flex items-end gap-3">
            <div className="flex-1 space-y-1.5">
              <Label htmlFor="session">Session</Label>
              <Input id="session" value={session} onChange={e => setSession(e.target.value)} />
            </div>
            <Button type="submit" disabled={create.isPending || !content.trim()}>{create.isPending ? 'Saving…' : 'Remember'}</Button>
          </div>
          <ErrorBanner error={create.error} />
        </form>
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map(text => (
            <Button key={text} type="button" variant="outline" size="sm" disabled={create.isPending} onClick={() => send(text)}>{text}</Button>
          ))}
        </div>
      </section>
      <section className="space-y-3" aria-live="polite">
        {log.length === 0
          ? <EmptyState title="Nothing stored in this visit yet">Try one of the examples, such as “I live in Chennai.” followed by “I moved to Bengaluru.”</EmptyState>
          : log.map(record => <MemoryCard key={`${record.id}-${record.updated_at}`} record={record} />)}
      </section>
    </div>
  )
}
