import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Download } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Skeleton } from '@/components/ui/skeleton'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { NativeSelect } from '@/components/NativeSelect'
import { ErrorBanner, EmptyState } from '@/components/ErrorBanner'
import { CategoryBadge, StatusBadge, TierBadge } from '@/components/memory/Badges'
import { useMemories, type MemoryFilters } from '@/api/hooks'
import { download } from '@/api/client'
import { CATEGORIES, TIERS, type Category, type Tier } from '@/api/types'
import { label, num, relative } from '@/lib/format'

export function Memories() {
  const [filters, setFilters] = useState<MemoryFilters>({ tier: '', category: '', session_id: '', include_history: false })
  const memories = useMemories(filters)
  const [exportError, setExportError] = useState<unknown>(null)
  const set = (patch: Partial<MemoryFilters>) => setFilters(previous => ({ ...previous, ...patch }))
  const exportAs = (format: 'json' | 'csv') =>
    download(`/api/v1/me/export?format=${format}`, `cognimem-export.${format}`).catch(setExportError)

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">Memories</h1>
          <p className="text-sm text-muted-foreground">What is visible now. Expired and superseded memories are hidden unless you include history.</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => exportAs('json')}><Download />JSON</Button>
          <Button variant="outline" size="sm" onClick={() => exportAs('csv')}><Download />CSV</Button>
        </div>
      </div>
      <div className="flex flex-wrap items-end gap-3">
        <div className="space-y-1.5">
          <Label htmlFor="tier">Tier</Label>
          <NativeSelect id="tier" value={filters.tier} onChange={e => set({ tier: e.target.value as Tier | '' })}>
            <option value="">All tiers</option>{TIERS.map(t => <option key={t} value={t}>{label(t)}</option>)}
          </NativeSelect>
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="category">Category</Label>
          <NativeSelect id="category" value={filters.category} onChange={e => set({ category: e.target.value as Category | '' })}>
            <option value="">All categories</option>{CATEGORIES.map(c => <option key={c} value={c}>{c}</option>)}
          </NativeSelect>
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="session-filter">Session</Label>
          <Input id="session-filter" className="w-40" placeholder="any" value={filters.session_id} onChange={e => set({ session_id: e.target.value })} />
        </div>
        <label className="flex items-center gap-2 pb-1.5 text-sm">
          <Switch checked={!!filters.include_history} onCheckedChange={checked => set({ include_history: checked })} aria-label="Include history" />
          Include history
        </label>
      </div>
      <ErrorBanner error={memories.error ?? exportError} />
      {memories.isLoading ? <Skeleton className="h-40 w-full" /> : !memories.data?.length ? (
        <EmptyState title="No memories match">Add some on the Chat page.</EmptyState>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Memory</TableHead><TableHead>Category</TableHead><TableHead>Tier</TableHead>
              <TableHead className="text-right">Importance</TableHead><TableHead>Expires</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {memories.data.map(record => (
              <TableRow key={`${record.id}-${record.updated_at}`} className={record.memory_status !== 'active' ? 'opacity-60' : ''}>
                <TableCell className="max-w-md whitespace-normal">
                  <Link to={`/memories/${record.id}`} className="hover:underline">{record.content}</Link>{' '}
                  <StatusBadge status={record.memory_status} />
                </TableCell>
                <TableCell><CategoryBadge category={record.category} /></TableCell>
                <TableCell><TierBadge tier={record.tier} /></TableCell>
                <TableCell className="text-right font-mono">{num(record.importance_score, 3)}</TableCell>
                <TableCell className="text-muted-foreground">{record.expires_at ? relative(record.expires_at) : 'never'}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  )
}
