export function formatDate(value: string | null | undefined) {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export function relative(value: string | null | undefined, now = Date.now()) {
  if (!value) return 'never'
  const ms = new Date(value).getTime() - now
  const abs = Math.abs(ms)
  const [unit, size] = abs >= 86_400_000 ? ['day', 86_400_000] : abs >= 3_600_000 ? ['hour', 3_600_000] : ['minute', 60_000]
  return new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' }).format(Math.round(ms / size), unit as Intl.RelativeTimeFormatUnit)
}

export const pct = (value: number | null | undefined, digits = 0) => (value == null ? '—' : `${(value * 100).toFixed(digits)}%`)
export const num = (value: number | null | undefined, digits = 3) => (value == null ? '—' : value.toFixed(digits))
export const label = (value: string) => value.replace(/_/g, ' ')
