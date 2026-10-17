/** Sums the samples of one Prometheus metric family, e.g. cognimem_memories_created_total. */
export function metricTotal(text: string | undefined, name: string): number | null {
  if (!text) return null
  let total = 0, found = false
  for (const line of text.split('\n')) {
    if (line.startsWith('#') || !line.startsWith(name)) continue
    const rest = line.slice(name.length)
    if (rest && rest[0] !== '{' && rest[0] !== ' ') continue
    const value = Number(line.trim().split(/\s+/).pop())
    if (Number.isFinite(value)) { total += value; found = true }
  }
  return found ? total : null
}
