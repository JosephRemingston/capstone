import { Badge } from '@/components/ui/badge'
import { label } from '@/lib/format'
import type { Category, Tier } from '@/api/types'
import { cn } from '@/lib/utils'

const TIER_STYLE: Record<Tier, string> = {
  working: 'bg-zinc-500/15 text-zinc-700 dark:text-zinc-300',
  short_term: 'bg-amber-500/15 text-amber-800 dark:text-amber-300',
  long_term: 'bg-emerald-500/15 text-emerald-800 dark:text-emerald-300',
  archive: 'bg-slate-500/15 text-slate-700 dark:text-slate-300',
}

export function TierBadge({ tier }: { tier: Tier }) {
  return <Badge variant="outline" className={cn('border-transparent', TIER_STYLE[tier])}>{label(tier)}</Badge>
}

export function CategoryBadge({ category }: { category: Category }) {
  return <Badge variant="outline">{category}</Badge>
}

export function StatusBadge({ status }: { status: string }) {
  if (status === 'active') return null
  return <Badge variant="destructive">{status}</Badge>
}
