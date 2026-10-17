import { useState, type FormEvent } from 'react'
import { Navigate } from 'react-router-dom'
import { Brain } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { ErrorBanner } from '@/components/ErrorBanner'
import { useLogin, useSignup } from '@/api/hooks'
import { useAuth } from '@/auth/AuthContext'

function AuthForm({ mode }: { mode: 'login' | 'signup' }) {
  const { signIn } = useAuth()
  const login = useLogin()
  const signup = useSignup()
  const mutation = mode === 'login' ? login : signup
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const submit = (event: FormEvent) => {
    event.preventDefault()
    mutation.mutate({ email, password }, { onSuccess: signIn })
  }
  return (
    <form onSubmit={submit} className="space-y-4">
      <div className="space-y-1.5">
        <Label htmlFor={`${mode}-email`}>Email</Label>
        <Input id={`${mode}-email`} type="email" autoComplete="email" required value={email} onChange={e => setEmail(e.target.value)} />
      </div>
      <div className="space-y-1.5">
        <Label htmlFor={`${mode}-password`}>Password</Label>
        <Input id={`${mode}-password`} type="password" required minLength={mode === 'signup' ? 12 : 1}
          autoComplete={mode === 'signup' ? 'new-password' : 'current-password'} value={password} onChange={e => setPassword(e.target.value)} />
        {mode === 'signup' && <p className="text-xs text-muted-foreground">At least 12 characters.</p>}
      </div>
      <ErrorBanner error={mutation.error} />
      <Button type="submit" className="w-full" disabled={mutation.isPending}>
        {mutation.isPending ? 'Please wait…' : mode === 'login' ? 'Sign in' : 'Create account'}
      </Button>
    </form>
  )
}

export function Login() {
  const { user, notice } = useAuth()
  if (user) return <Navigate to="/" replace />
  return (
    <div className="grid min-h-svh place-items-center bg-muted/40 px-4">
      <Card className="w-full max-w-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Brain className="size-5" />CogniMem</CardTitle>
          <CardDescription>A local-first memory for your assistant.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {notice && <Alert><AlertDescription>{notice}</AlertDescription></Alert>}
          <Tabs defaultValue="login">
            <TabsList className="w-full">
              <TabsTrigger value="login">Sign in</TabsTrigger>
              <TabsTrigger value="signup">Sign up</TabsTrigger>
            </TabsList>
            <TabsContent value="login" className="pt-4"><AuthForm mode="login" /></TabsContent>
            <TabsContent value="signup" className="pt-4"><AuthForm mode="signup" /></TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  )
}
