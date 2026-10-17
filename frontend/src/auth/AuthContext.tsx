import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { setToken, setUnauthorizedHandler } from '@/api/client'
import type { AuthResult, User } from '@/api/types'

// Token in sessionStorage, as the /demo page does: gone when the tab closes, never in localStorage.
const KEY = 'cognimem_session'

interface Session { token: string; user: User; expiresAt: number }
interface AuthValue { user: User | null; signIn: (result: AuthResult) => void; signOut: (reason?: string) => void; notice: string | null }

const AuthContext = createContext<AuthValue | null>(null)

function load(): Session | null {
  try {
    const session = JSON.parse(sessionStorage.getItem(KEY) ?? 'null') as Session | null
    return session && session.expiresAt > Date.now() ? session : null
  } catch {
    return null
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const client = useQueryClient()
  const [session, setSession] = useState<Session | null>(() => { const s = load(); setToken(s?.token ?? null); return s })
  const [notice, setNotice] = useState<string | null>(null)

  const signOut = useCallback((reason?: string) => {
    try { sessionStorage.removeItem(KEY) } catch { /* storage unavailable */ }
    setToken(null)
    setSession(null)
    client.clear()
    setNotice(reason ?? null)
  }, [client])

  const signIn = useCallback((result: AuthResult) => {
    const next = { token: result.access_token, user: result.user, expiresAt: Date.now() + result.expires_in * 1000 }
    try { sessionStorage.setItem(KEY, JSON.stringify(next)) } catch { /* storage unavailable */ }
    setToken(next.token)
    setSession(next)
    setNotice(null)
  }, [])

  // The API has no refresh endpoint: when the token expires (or any call returns 401) the user signs in again.
  useEffect(() => {
    setUnauthorizedHandler(() => signOut('Your session expired. Please sign in again.'))
    return () => setUnauthorizedHandler(null)
  }, [signOut])
  useEffect(() => {
    if (!session) return
    const timer = window.setTimeout(() => signOut('Your session expired. Please sign in again.'), session.expiresAt - Date.now())
    return () => window.clearTimeout(timer)
  }, [session, signOut])

  const value = useMemo(() => ({ user: session?.user ?? null, signIn, signOut, notice }), [session, signIn, signOut, notice])
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const value = useContext(AuthContext)
  if (!value) throw new Error('useAuth must be used inside AuthProvider')
  return value
}
