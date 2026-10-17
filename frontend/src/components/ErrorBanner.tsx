import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { ApiError } from '@/api/client'

export function ErrorBanner({ error }: { error: unknown }) {
  if (!error) return null
  const api = error instanceof ApiError ? error : null
  return (
    <Alert variant="destructive" role="alert">
      <AlertTitle>{api ? `Request failed (${api.status})` : 'Something went wrong'}</AlertTitle>
      <AlertDescription>
        {error instanceof Error ? error.message : String(error)}
        {api?.requestId && <span className="block text-xs opacity-70">request {api.requestId}</span>}
      </AlertDescription>
    </Alert>
  )
}

export function EmptyState({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">
      <p className="font-medium text-foreground">{title}</p>
      {children}
    </div>
  )
}
