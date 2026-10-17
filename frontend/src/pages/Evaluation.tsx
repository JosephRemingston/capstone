import { useMemo, useState } from 'react'
import { useQueries } from '@tanstack/react-query'
import { AlertTriangle } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Skeleton } from '@/components/ui/skeleton'
import { Switch } from '@/components/ui/switch'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { NativeSelect } from '@/components/NativeSelect'
import { ErrorBanner, EmptyState } from '@/components/ErrorBanner'
import { api } from '@/api/client'
import { useModelReport, useSnapshot, useSnapshots } from '@/api/hooks'
import type { ModelEvaluation, Snapshot } from '@/api/types'
import { LAYERS, LAYER_TAGS, cells, compare, direction, settingsSummary, type Flat, type MetricRow } from '@/lib/evaluation'
import { formatDate } from '@/lib/format'
import { cn } from '@/lib/utils'

function MetricTable({ rows, flats }: { rows: MetricRow[]; flats: Record<string, Flat> }) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Metric</TableHead>
          <TableHead className="w-24 text-right">Baseline</TableHead>
          <TableHead className="w-24 text-right">Final</TableHead>
          <TableHead className="w-24 text-right">Current</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map(row => {
          const c = cells(row, flats)
          const note = row.note?.(flats)
          return (
            <TableRow key={row.label}>
              <TableCell className="whitespace-normal">
                {row.label}
                {note && <span className="block text-xs text-muted-foreground">{note}</span>}
              </TableCell>
              <TableCell className="text-right font-mono">{c.baseline}</TableCell>
              <TableCell className="text-right font-mono">{c.final}</TableCell>
              <TableCell className={cn('text-right font-mono', !c.same && 'font-semibold')}
                title={c.same ? 'This measurement does not use the importance scorer, so Current equals Final' : undefined}>
                {c.current}{c.same && <span className="ml-1 text-muted-foreground">=</span>}
              </TableCell>
            </TableRow>
          )
        })}
      </TableBody>
    </Table>
  )
}

function ByLayer() {
  const results = useQueries({
    queries: LAYER_TAGS.map(tag => ({ queryKey: ['snapshot', tag], staleTime: 60_000, queryFn: () => api.get<Snapshot>(`/api/v1/evaluation/snapshots/${tag}`) })),
  })
  if (results.some(r => r.isLoading)) return <Skeleton className="h-96 w-full" />
  const error = results.find(r => r.error)?.error
  const flats: Record<string, Flat> = Object.fromEntries(LAYER_TAGS.map((tag, i) => [tag, results[i].data?.flat ?? {}]))
  return (
    <div className="space-y-4">
      <ErrorBanner error={error} />
      <Card size="sm">
        <CardContent className="grid gap-3 pt-3 text-sm sm:grid-cols-3">
          <p><b>Baseline</b> · 12 Oct 2026, the rebuilt harness before any fix.</p>
          <p><b>Final</b> · 12 Oct 2026, every fix, rule-based importance score.</p>
          <p><b>Current</b> · 16 Oct 2026, the shipped default: importance = 25 % model + 75 % rules. <span className="text-muted-foreground">“=” marks measurements that never use the importance score, so Current equals Final.</span></p>
        </CardContent>
      </Card>
      {LAYERS.map(layer => (
        <Card key={layer.id}>
          <CardHeader>
            <CardTitle>{layer.title}</CardTitle>
            <CardDescription>{layer.description}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            {layer.rows.length > 0 && (
              <section className="space-y-2">
                {layer.heading && <h3 className="text-sm font-medium text-muted-foreground">{layer.heading}</h3>}
                <MetricTable rows={layer.rows} flats={flats} />
              </section>
            )}
            {layer.groups?.map(group => (
              <section key={group.heading} className="space-y-2">
                <h3 className="text-sm font-medium text-muted-foreground">{group.heading}</h3>
                <MetricTable rows={group.rows} flats={flats} />
              </section>
            ))}
          </CardContent>
        </Card>
      ))}
      <Card size="sm">
        <CardContent className="pt-3 text-sm text-muted-foreground">
          Layer 6 (CLI, API, web app) has no measured metrics. Different datasets measure different things: do not compare percentages across tables.
          The README explains each layer's problems and fixes; per-question-type breakdowns are in the local benchmark report.
        </CardContent>
      </Card>
    </div>
  )
}

const f4 = (v: number) => v.toFixed(4)
const pct2 = (v: number) => `${(v * 100).toFixed(2)}%`

function ModelTab() {
  const report = useModelReport()
  if (report.isLoading) return <Skeleton className="h-96 w-full" />
  if (report.error) return <ErrorBanner error={report.error} />
  const r = report.data!
  const names = r.class_names
  const rows: [string, ModelEvaluation][] = [
    [`Always the most common class (${r.majority.class_name})`, r.majority],
    ['Best possible (oracle)', r.oracle],
    ['Ensemble + thresholds (shipped)', r.ensemble_thresholds],
  ]
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Importance model · csv5k test split</CardTitle>
          <CardDescription>{r.model}. {r.dataset.split_sizes.test.toLocaleString()} test rows from {r.dataset.split_sentences.test} sentences; 95% intervals resample whole sentences ({r.resamples.toLocaleString()} resamples).</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model</TableHead><TableHead className="text-right">Accuracy</TableHead><TableHead className="text-right">Balanced accuracy</TableHead>
                <TableHead className="text-right">Macro F1</TableHead><TableHead>95% CI</TableHead><TableHead>F1 {names.join(' / ')}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map(([label, m]) => (
                <TableRow key={label} className={label.startsWith('Ensemble') ? 'font-semibold' : ''}>
                  <TableCell className="whitespace-normal">{label}</TableCell>
                  <TableCell className="text-right font-mono">{pct2(m.test.accuracy)}</TableCell>
                  <TableCell className="text-right font-mono">{pct2(m.test.balanced_accuracy)}</TableCell>
                  <TableCell className="text-right font-mono">{f4(m.test.macro_f1)}</TableCell>
                  <TableCell className="font-mono">[{m.sentence_ci95[0].toFixed(3)}, {m.sentence_ci95[1].toFixed(3)}]</TableCell>
                  <TableCell className="font-mono">{names.map(n => m.test.per_class[n].f1.toFixed(2)).join(' / ')}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <p className="mt-2 text-xs text-muted-foreground">The oracle predicts each test sentence's most common test label: a ceiling that reads the test labels, not a model.</p>
        </CardContent>
      </Card>
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader><CardTitle className="text-base">How it scores</CardTitle></CardHeader>
          <CardContent className="space-y-1 text-sm">
            <p>Class values (its 25 % share of the importance score): {names.map((n, i) => `${n} ${r.class_values[i]}`).join(' · ')}</p>
            <p>Blend: {r.tuned.w_xgb} × XGBoost + {1 - r.tuned.w_xgb} × deep head; class weights {r.tuned.class_weights.join(' / ')}</p>
            <p className="text-muted-foreground">Validation macro F1 {f4(r.tuned.val_macro_f1_before)} → {f4(r.tuned.val_macro_f1_after)} with the class weights (all tuning on validation only).</p>
            <p className="text-muted-foreground">Data: {r.dataset.n_rows.toLocaleString()} AI-generated sentences ({r.dataset.diagnostics.unique_sentences} distinct, at most {r.dataset.diagnostics.words_per_text.max} words); classes low &lt; {r.dataset.bin_edges[0]} ≤ mid &lt; {r.dataset.bin_edges[1]} ≤ high.</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle className="text-base">Serving matches training</CardTitle></CardHeader>
          <CardContent className="space-y-1 text-sm">
            {Object.entries(r.serving_checks).map(([name, check]) => (
              <p key={name}>{name.replace(/_/g, ' ')}: {check.class_mismatches} class mismatches, max probability difference {check.max_abs_probability_diff.toExponential(1)}</p>
            ))}
          </CardContent>
        </Card>
      </div>
      <Card size="sm">
        <CardHeader><CardTitle className="text-base">Caveats</CardTitle></CardHeader>
        <CardContent><ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">{r.caveats.map(c => <li key={c}>{c}</li>)}</ul></CardContent>
      </Card>
    </div>
  )
}

function tone(key: string, delta: number) {
  const better = Math.sign(delta) * direction(key)
  return better > 0 ? 'text-emerald-700 dark:text-emerald-400' : better < 0 ? 'text-red-700 dark:text-red-400' : ''
}

function CompareTab() {
  const snapshots = useSnapshots()
  const [before, setBefore] = useState('baseline')
  const [after, setAfter] = useState('final')
  const [filter, setFilter] = useState('')
  const [changedOnly, setChangedOnly] = useState(true)
  const left = useSnapshot(before)
  const right = useSnapshot(after)
  const result = useMemo(
    () => (left.data && right.data ? compare(left.data.flat, right.data.flat, filter, changedOnly) : null),
    [left.data, right.data, filter, changedOnly],
  )
  const tags = snapshots.data ?? []
  const describe = (s?: Snapshot) => s && `${formatDate(s.manifest.created_at)} · ${s.manifest.scorer ?? 'heuristic'} scorer · git ${s.manifest.git_head ?? '—'}`

  return (
    <div className="space-y-4">
      <ErrorBanner error={snapshots.error ?? left.error ?? right.error} />
      <div className="flex flex-wrap items-end gap-3">
        <div className="space-y-1.5"><Label htmlFor="before">Before</Label>
          <NativeSelect id="before" value={before} onChange={e => setBefore(e.target.value)}>{tags.map(t => <option key={t.tag} value={t.tag}>{t.tag}</option>)}</NativeSelect></div>
        <div className="space-y-1.5"><Label htmlFor="after">After</Label>
          <NativeSelect id="after" value={after} onChange={e => setAfter(e.target.value)}>{tags.map(t => <option key={t.tag} value={t.tag}>{t.tag}</option>)}</NativeSelect></div>
        <div className="min-w-56 flex-1 space-y-1.5"><Label htmlFor="metric-filter">Filter</Label>
          <Input id="metric-filter" placeholder="e.g. locomo recall_at_5" value={filter} onChange={e => setFilter(e.target.value)} /></div>
        <label className="flex items-center gap-2 pb-1.5 text-sm"><Switch checked={changedOnly} onCheckedChange={setChangedOnly} aria-label="Changed only" />Changed only</label>
      </div>
      <p className="text-xs text-muted-foreground">{describe(left.data)}{left.data && right.data && '  →  '}{describe(right.data)}</p>
      {result && result.protocol.length > 0 && (
        <Alert>
          <AlertTriangle />
          <AlertTitle>The two snapshots were measured with different settings</AlertTitle>
          <AlertDescription>
            <p>{settingsSummary(result.protocol)}. Differences in these sections may come from the settings, not the code.</p>
            <details className="mt-1">
              <summary className="cursor-pointer text-xs">Show all {result.protocol.length} setting differences</summary>
              {result.protocol.map(p => <span key={p.key} className="block font-mono text-xs">{p.key}: {String(p.before ?? '—')} → {String(p.after ?? '—')}</span>)}
            </details>
          </AlertDescription>
        </Alert>
      )}
      {(left.isLoading || right.isLoading) && <Skeleton className="h-64 w-full" />}
      {result && (result.rows.length === 0 ? <EmptyState title="No common metrics match">Try another filter, turn off “Changed only”, or pick snapshots that measured the same sections.</EmptyState> : (
        <Table>
          <TableHeader>
            <TableRow><TableHead>Metric</TableHead><TableHead className="text-right">{before}</TableHead><TableHead className="text-right">{after}</TableHead><TableHead className="text-right">Change</TableHead></TableRow>
          </TableHeader>
          <TableBody>
            {result.rows.map(row => (
              <TableRow key={row.key}>
                <TableCell className="font-mono text-xs whitespace-normal break-all">{row.key}</TableCell>
                <TableCell className="text-right font-mono">{f4(row.before)}</TableCell>
                <TableCell className="text-right font-mono">{f4(row.after)}</TableCell>
                <TableCell className={cn('text-right font-mono', tone(row.key, row.delta))}>
                  {row.delta > 0 ? '+' : ''}{f4(row.delta)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      ))}
      {result && <p className="text-xs text-muted-foreground">{result.rows.length} metrics shown{result.onlyOne > 0 && `; ${result.onlyOne} exist in only one snapshot`}. Green is better and red is worse; for latency and storage lower is better, and plain counts stay uncoloured.</p>}
    </div>
  )
}

export function Evaluation() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">Evaluation</h1>
        <p className="text-sm text-muted-foreground">Every number comes from the tracked snapshots in evaluation/reports, the same files behind the README.</p>
      </div>
      <Tabs defaultValue="layers">
        <TabsList>
          <TabsTrigger value="layers">By layer</TabsTrigger>
          <TabsTrigger value="model">Importance model</TabsTrigger>
          <TabsTrigger value="compare">Compare snapshots</TabsTrigger>
        </TabsList>
        <TabsContent value="layers" className="pt-4"><ByLayer /></TabsContent>
        <TabsContent value="model" className="pt-4"><ModelTab /></TabsContent>
        <TabsContent value="compare" className="pt-4"><CompareTab /></TabsContent>
      </Tabs>
    </div>
  )
}
