import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react'
import { Link } from 'react-router-dom'
import ForceGraph2D from 'react-force-graph-2d'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Skeleton } from '@/components/ui/skeleton'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { NativeSelect } from '@/components/NativeSelect'
import { ErrorBanner, EmptyState } from '@/components/ErrorBanner'
import { useGraph, type GraphParams } from '@/api/hooks'
import type { GraphRelation } from '@/api/types'
import { formatDate, label } from '@/lib/format'

interface Node { id: string; label: string; type: string; highlighted: boolean }
interface Link_ { source: string; target: string; relation: GraphRelation }

const TYPE_COLOR: Record<string, string> = { person: '#6366f1', location: '#10b981', organization: '#f59e0b', task: '#ef4444' }

function useSize<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(600)
  useEffect(() => {
    if (!ref.current) return
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width))
    observer.observe(ref.current)
    return () => observer.disconnect()
  }, [])
  return [ref, width] as const
}

export function Graph() {
  const [draft, setDraft] = useState<GraphParams>({ query: '', predicate: '', as_of: '', known_at: '', hops: 2 })
  const [params, setParams] = useState<GraphParams>({})
  const [asTable, setAsTable] = useState(false)
  const [selected, setSelected] = useState<GraphRelation | null>(null)
  const graph = useGraph(params)
  const [box, width] = useSize<HTMLDivElement>()

  const iso = (value?: string) => (value ? new Date(value).toISOString() : undefined)
  const submit = (event: FormEvent) => {
    event.preventDefault()
    setParams({ query: draft.query?.trim() || undefined, predicate: draft.predicate || undefined, as_of: iso(draft.as_of), known_at: iso(draft.known_at), hops: draft.hops })
  }

  const relations = useMemo(() => graph.data?.relations ?? [], [graph.data])
  const predicates = useMemo(() => [...new Set(relations.map(r => r.predicate))].sort(), [relations])
  const data = useMemo(() => {
    const matched = new Set((graph.data?.entities ?? []).map(e => e.id))
    const nodes = new Map<string, Node>()
    for (const r of relations) {
      nodes.set(r.subject, { id: r.subject, label: r.subject_label, type: r.subject_type, highlighted: matched.has(r.subject) })
      nodes.set(r.object, { id: r.object, label: r.object_label, type: r.object_type, highlighted: matched.has(r.object) })
    }
    const links: Link_[] = relations.map(r => ({ source: r.subject, target: r.object, relation: r }))
    return { nodes: [...nodes.values()], links }
  }, [relations, graph.data])

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">Knowledge graph</h1>
        <p className="text-sm text-muted-foreground">Relations extracted from your memories, each with when it was true (valid time) and when it was learned (recorded time).</p>
      </div>
      <form onSubmit={submit} className="flex flex-wrap items-end gap-3">
        <div className="min-w-48 flex-1 space-y-1.5"><Label htmlFor="g-query">Highlight entity</Label><Input id="g-query" placeholder="Bengaluru" value={draft.query} onChange={e => setDraft({ ...draft, query: e.target.value })} /></div>
        <div className="space-y-1.5"><Label htmlFor="g-pred">Relation</Label>
          <NativeSelect id="g-pred" value={draft.predicate} onChange={e => setDraft({ ...draft, predicate: e.target.value })}>
            <option value="">All</option>{predicates.map(p => <option key={p} value={p}>{label(p)}</option>)}
          </NativeSelect></div>
        <div className="space-y-1.5"><Label htmlFor="g-asof">True as of</Label><Input id="g-asof" type="datetime-local" className="w-52" value={draft.as_of} onChange={e => setDraft({ ...draft, as_of: e.target.value })} /></div>
        <div className="space-y-1.5"><Label htmlFor="g-known">As known at</Label><Input id="g-known" type="datetime-local" className="w-52" value={draft.known_at} onChange={e => setDraft({ ...draft, known_at: e.target.value })} /></div>
        <div className="space-y-1.5"><Label htmlFor="g-hops">Hops</Label>
          <NativeSelect id="g-hops" value={draft.hops} onChange={e => setDraft({ ...draft, hops: Number(e.target.value) })}>{[1, 2, 3].map(h => <option key={h}>{h}</option>)}</NativeSelect></div>
        <Button type="submit">Apply</Button>
        <label className="flex items-center gap-2 pb-1.5 text-sm"><Switch checked={asTable} onCheckedChange={setAsTable} aria-label="Table view" />Table</label>
      </form>
      <ErrorBanner error={graph.error} />
      {graph.isLoading ? <Skeleton className="h-96 w-full" /> : relations.length === 0 ? (
        <EmptyState title="No relations yet">Facts like “I live in Chennai.” or “I work at Acme.” become relations here.</EmptyState>
      ) : asTable ? (
        <Table>
          <TableHeader><TableRow><TableHead>Subject</TableHead><TableHead>Relation</TableHead><TableHead>Object</TableHead><TableHead>Valid from</TableHead><TableHead>Recorded</TableHead><TableHead>Source</TableHead></TableRow></TableHeader>
          <TableBody>{relations.map(r => (
            <TableRow key={r.id}>
              <TableCell>{r.subject_label}</TableCell><TableCell>{r.positive ? '' : 'not '}{label(r.predicate)}</TableCell><TableCell>{r.object_label}</TableCell>
              <TableCell>{formatDate(r.valid_from)}</TableCell><TableCell>{formatDate(r.known_from)}</TableCell>
              <TableCell><Link to={`/memories/${r.source_id}`} className="font-mono text-xs underline">{r.source_id.slice(0, 8)}</Link></TableCell>
            </TableRow>))}
          </TableBody>
        </Table>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_18rem]">
          <div ref={box} className="overflow-hidden rounded-lg border" data-testid="graph-canvas">
            <ForceGraph2D
              graphData={data} width={width} height={460} backgroundColor="transparent" cooldownTicks={80}
              nodeLabel={(n: Node) => `${n.label} (${n.type})`}
              linkLabel={(l: Link_) => label(l.relation.predicate)}
              linkDirectionalArrowLength={4} linkDirectionalArrowRelPos={0.9}
              linkColor={() => 'rgba(120,120,140,0.6)'}
              onLinkClick={(l: Link_) => setSelected(l.relation)}
              nodeCanvasObject={(n: Node & { x?: number; y?: number }, ctx, scale) => {
                const radius = n.highlighted ? 7 : 5
                ctx.beginPath(); ctx.arc(n.x ?? 0, n.y ?? 0, radius, 0, 2 * Math.PI)
                ctx.fillStyle = TYPE_COLOR[n.type] ?? '#64748b'; ctx.fill()
                if (n.highlighted) { ctx.lineWidth = 2 / scale; ctx.strokeStyle = '#ef4444'; ctx.stroke() }
                ctx.font = `${12 / scale}px system-ui`; ctx.fillStyle = getComputedStyle(document.documentElement).color
                ctx.textAlign = 'center'; ctx.fillText(n.label, n.x ?? 0, (n.y ?? 0) + radius + 12 / scale)
              }}
            />
          </div>
          <Card size="sm">
            <CardHeader><CardTitle className="text-base">{selected ? 'Relation' : 'Legend'}</CardTitle></CardHeader>
            <CardContent className="space-y-2 text-sm">
              {selected ? (
                <>
                  <p><b>{selected.subject_label}</b> {selected.positive ? '' : 'not '}{label(selected.predicate)} <b>{selected.object_label}</b></p>
                  <p className="text-muted-foreground">True from {formatDate(selected.valid_from)}{selected.valid_to && ` until ${formatDate(selected.valid_to)}`}</p>
                  <p className="text-muted-foreground">Recorded {formatDate(selected.known_from)}</p>
                  <Link to={`/memories/${selected.source_id}`} className="underline">Open source memory</Link>
                </>
              ) : (
                <>
                  {Object.entries(TYPE_COLOR).map(([type, color]) => (
                    <p key={type} className="flex items-center gap-2"><span className="size-3 rounded-full" style={{ background: color }} />{type}</p>
                  ))}
                  <p className="text-muted-foreground">Click a relation to see when it was true and where it came from.</p>
                </>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
