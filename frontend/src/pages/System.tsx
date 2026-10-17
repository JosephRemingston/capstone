import { useState } from 'react'
import { CheckCircle2, XCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { ErrorBanner } from '@/components/ErrorBanner'
import { useDeleteAccount, useHealth, useMetrics } from '@/api/hooks'
import { useAuth } from '@/auth/AuthContext'
import { label } from '@/lib/format'
import { metricTotal } from '@/lib/metrics'

export function System() {
  const health = useHealth()
  const metrics = useMetrics()
  const { user, signOut } = useAuth()
  const remove = useDeleteAccount()
  const [open, setOpen] = useState(false)
  const [typed, setTyped] = useState('')
  const scorer = health.data?.importance_scorer
  const families = [...new Set((metrics.data ?? '').split('\n').filter(l => l.startsWith('# TYPE cognimem')).map(l => l.split(' ')[2]))]

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card>
        <CardHeader><CardTitle>Health</CardTitle><CardDescription>/health/ready, refreshed every 30 seconds.</CardDescription></CardHeader>
        <CardContent className="space-y-2 text-sm">
          <ErrorBanner error={health.error} />
          {health.data && Object.entries(health.data.checks).map(([check, state]) => (
            <p key={check} className="flex items-center gap-2">
              {state === 'ok' ? <CheckCircle2 className="size-4 text-emerald-600" /> : <XCircle className="size-4 text-destructive" />}{label(check)}: {state}
            </p>
          ))}
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>Importance scorer</CardTitle><CardDescription>Chosen at start-up with COGNIMEM_SCORER.</CardDescription></CardHeader>
        <CardContent className="space-y-1 text-sm">
          <p className="text-lg font-medium">{scorer?.type === 'hybrid' ? 'Hybrid: 25 % model + 75 % rules' : scorer?.type === 'heuristic' ? 'Rules only (heuristic)' : '—'}</p>
          {scorer?.model_sha256 && <p className="font-mono text-xs text-muted-foreground">model {scorer.model_sha256.slice(0, 16)}</p>}
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>Metrics</CardTitle><CardDescription>Totals from /metrics (Prometheus) since the server started.</CardDescription></CardHeader>
        <CardContent className="grid grid-cols-2 gap-3 text-sm">
          {families.length === 0 && <p className="text-muted-foreground">No metrics yet.</p>}
          {families.map(name => (
            <div key={name}><p className="text-muted-foreground">{label(name.replace(/^cognimem_/, '').replace(/_total$/, ''))}</p><p className="text-lg font-medium">{metricTotal(metrics.data, name)?.toLocaleString() ?? '—'}</p></div>
          ))}
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>Account</CardTitle><CardDescription>{user?.email}</CardDescription></CardHeader>
        <CardContent>
          <Button variant="destructive" onClick={() => { setTyped(''); setOpen(true) }}>Delete account and all memories</Button>
        </CardContent>
      </Card>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete your account?</DialogTitle>
            <DialogDescription>This permanently removes your account and every memory. Type your email to confirm.</DialogDescription>
          </DialogHeader>
          <Label htmlFor="confirm-email" className="sr-only">Email</Label>
          <Input id="confirm-email" value={typed} onChange={e => setTyped(e.target.value)} placeholder={user?.email} />
          <ErrorBanner error={remove.error} />
          <DialogFooter>
            <Button variant="outline" onClick={() => setOpen(false)}>Cancel</Button>
            <Button variant="destructive" disabled={typed !== user?.email || remove.isPending}
              onClick={() => remove.mutate(undefined, { onSuccess: () => signOut('Your account was deleted.') })}>Delete permanently</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
