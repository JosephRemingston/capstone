import { useState, type FormEvent } from 'react'
import { Link } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { NativeSelect } from '@/components/NativeSelect'
import { ErrorBanner, EmptyState } from '@/components/ErrorBanner'
import { CategoryBadge, TierBadge } from '@/components/memory/Badges'
import { ChannelSignals } from '@/components/search/ChannelSignals'
import { CitationList } from '@/components/search/CitationList'
import { useAsk, useSearch, type AskParams } from '@/api/hooks'
import type { MemoryRecord, SearchHit, SearchMode } from '@/api/types'
import { num } from '@/lib/format'

const MODES: SearchMode[] = ['hybrid', 'keyword', 'semantic', 'graph', 'recency']

function isHit(item: SearchHit | MemoryRecord): item is SearchHit { return 'memory_id' in item }

function Timings({ timings }: { timings?: Record<string, number> }) {
  if (!timings) return null
  const { total, ...stages } = timings
  return (
    <p className="text-xs text-muted-foreground">
      {Object.entries(stages).map(([stage, ms]) => `${stage} ${ms.toFixed(0)} ms`).join(' · ')}{total != null && ` · total ${total.toFixed(0)} ms`}
    </p>
  )
}

function SearchPanel() {
  const search = useSearch()
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<SearchMode>('hybrid')
  const [rerank, setRerank] = useState(true)
  const submit = (event: FormEvent) => {
    event.preventDefault()
    if (query.trim()) search.mutate({ query: query.trim(), mode, limit: 10, rerank: mode === 'keyword' ? null : rerank })
  }
  const results = search.data ?? []
  const first = results.find(isHit)
  return (
    <div className="space-y-4">
      <form onSubmit={submit} className="flex flex-wrap items-end gap-3">
        <div className="min-w-60 flex-1 space-y-1.5">
          <Label htmlFor="search-query">Query</Label>
          <Input id="search-query" placeholder="Where do I live now?" value={query} onChange={e => setQuery(e.target.value)} />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="search-mode">Mode</Label>
          <NativeSelect id="search-mode" value={mode} onChange={e => setMode(e.target.value as SearchMode)}>
            {MODES.map(m => <option key={m} value={m}>{m}</option>)}
          </NativeSelect>
        </div>
        <label className="flex items-center gap-2 pb-1.5 text-sm">
          <Switch checked={rerank} disabled={mode === 'keyword'} onCheckedChange={setRerank} aria-label="Rerank" />Rerank
        </label>
        <Button type="submit" disabled={search.isPending || !query.trim()}>{search.isPending ? 'Searching…' : 'Search'}</Button>
      </form>
      <ErrorBanner error={search.error} />
      {first && <Timings timings={first.timings_ms} />}
      {search.isSuccess && results.length === 0 && <EmptyState title="No matching memories" />}
      <ol className="space-y-2">
        {results.map((item, index) => {
          const hit = isHit(item) ? item : null
          const record = hit ? hit.record : (item as MemoryRecord)
          const id = hit ? hit.memory_id : (item as MemoryRecord).id
          return (
            <li key={id}>
              <Card size="sm">
                <CardContent className="space-y-2 pt-3">
                  <div className="flex items-start justify-between gap-3">
                    <Link to={`/memories/${id}`} className="font-medium hover:underline">{index + 1}. {hit ? hit.text : record?.content}</Link>
                    {record && <span className="flex shrink-0 gap-1"><CategoryBadge category={record.category} /><TierBadge tier={record.tier} /></span>}
                  </div>
                  {hit && (
                    <>
                      <ChannelSignals hit={hit} />
                      <p className="font-mono text-xs text-muted-foreground">
                        fusion {num(hit.fusion_score, 4)} · reranker {hit.reranker?.status === 'applied' ? num(hit.reranker_score, 3) : hit.reranker?.status ?? '—'} · cosine {num(hit.cosine, 3)}
                      </p>
                    </>
                  )}
                </CardContent>
              </Card>
            </li>
          )
        })}
      </ol>
    </div>
  )
}

export function AskPanel() {
  const ask = useAsk()
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<AskParams['mode']>('hybrid')
  const [gemini, setGemini] = useState(false)
  const [confirming, setConfirming] = useState(false)
  const [abstain, setAbstain] = useState(false)
  const [asOf, setAsOf] = useState('')
  const [knownAt, setKnownAt] = useState('')

  const iso = (value: string) => (value ? new Date(value).toISOString() : undefined)
  const submit = (event: FormEvent) => {
    event.preventDefault()
    if (!query.trim()) return
    ask.mutate({ query: query.trim(), mode, generator: gemini ? 'gemini' : 'extractive', abstain: gemini ? false : abstain, as_of: iso(asOf), known_at: iso(knownAt) })
  }
  // Turning Gemini on sends memory text to Google: ask first.
  const toggleGemini = (checked: boolean) => (checked ? setConfirming(true) : setGemini(false))

  return (
    <div className="space-y-4">
      <form onSubmit={submit} className="space-y-3">
        <div className="flex flex-wrap items-end gap-3">
          <div className="min-w-60 flex-1 space-y-1.5">
            <Label htmlFor="ask-query">Question</Label>
            <Input id="ask-query" placeholder="Where do I live now?" value={query} onChange={e => setQuery(e.target.value)} />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="ask-mode">Retrieval</Label>
            <NativeSelect id="ask-mode" value={mode} onChange={e => setMode(e.target.value as AskParams['mode'])}>
              {MODES.filter(m => m !== 'recency').map(m => <option key={m} value={m}>{m}</option>)}
            </NativeSelect>
          </div>
          <Button type="submit" disabled={ask.isPending || !query.trim()}>{ask.isPending ? 'Answering…' : 'Ask'}</Button>
        </div>
        <div className="flex flex-wrap items-end gap-5 text-sm">
          <label className="flex items-center gap-2"><Switch checked={gemini} onCheckedChange={toggleGemini} aria-label="Use Gemini" />Use Gemini</label>
          <label className="flex items-center gap-2"><Switch checked={abstain && !gemini} disabled={gemini} onCheckedChange={setAbstain} aria-label="Abstain" />Refuse when unsure</label>
          <div className="space-y-1"><Label htmlFor="as-of" className="text-xs">True as of</Label><Input id="as-of" type="datetime-local" className="h-8 w-52" value={asOf} onChange={e => setAsOf(e.target.value)} /></div>
          <div className="space-y-1"><Label htmlFor="known-at" className="text-xs">As known at</Label><Input id="known-at" type="datetime-local" className="h-8 w-52" value={knownAt} onChange={e => setKnownAt(e.target.value)} /></div>
        </div>
        <p className="text-xs text-muted-foreground">
          {gemini ? 'Gemini writes the answer from the retrieved memories; their text is sent to Google.' : 'Answered locally: the cited memories themselves, nothing leaves this machine.'}
        </p>
      </form>
      <ErrorBanner error={ask.error} />
      {ask.data && <CitationList result={ask.data} />}
      <Dialog open={confirming} onOpenChange={setConfirming}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Send memories to Gemini?</DialogTitle>
            <DialogDescription>The memories retrieved for each question will be sent to Google's Gemini API to write the answer. Local answers never leave this machine.</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirming(false)}>Keep local</Button>
            <Button onClick={() => { setGemini(true); setConfirming(false) }}>Use Gemini</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export function Search() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">Search & Ask</h1>
        <p className="text-sm text-muted-foreground">Search shows how each channel and the reranker ranked your memories. Ask returns an answer with citations.</p>
      </div>
      <Tabs defaultValue="ask">
        <TabsList><TabsTrigger value="ask">Ask</TabsTrigger><TabsTrigger value="search">Search</TabsTrigger></TabsList>
        <TabsContent value="ask" className="pt-4"><AskPanel /></TabsContent>
        <TabsContent value="search" className="pt-4"><SearchPanel /></TabsContent>
      </Tabs>
    </div>
  )
}
