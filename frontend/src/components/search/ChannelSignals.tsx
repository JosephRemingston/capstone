import type { SearchHit } from '@/api/types'
import { num } from '@/lib/format'

const CHANNELS = ['keyword', 'semantic', 'graph'] as const

/** Which channels found this memory, at what rank, and how much each contributed to the fused score. */
export function ChannelSignals({ hit }: { hit: SearchHit }) {
  return (
    <div className="flex flex-wrap gap-1.5" data-testid="channel-signals">
      {CHANNELS.map(channel => {
        const signal = hit.signals?.[channel]
        return (
          <span key={channel} className={`rounded-md border px-2 py-0.5 text-xs ${signal ? '' : 'opacity-40'}`}
            title={signal ? `contribution ${num(signal.contribution, 4)}` : 'not found by this channel'}>
            {channel} {signal ? `#${signal.rank}` : '—'}
          </span>
        )
      })}
    </div>
  )
}
