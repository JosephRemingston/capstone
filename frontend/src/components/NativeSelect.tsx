import { cn } from '@/lib/utils'

/** A styled native <select>: accessible and dependable for small filter lists. */
export function NativeSelect({ className, ...props }: React.ComponentProps<'select'>) {
  return (
    <select
      className={cn('h-8 rounded-lg border border-input bg-transparent px-2.5 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30', className)}
      {...props}
    />
  )
}
