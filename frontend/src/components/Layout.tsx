import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { Brain, Database, MessageSquare, Moon, Network, Search, Sun, Activity, LogOut, BarChart3 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useAuth } from '@/auth/AuthContext'
import { cn } from '@/lib/utils'

const NAV = [
  { to: '/', label: 'Chat', icon: MessageSquare, end: true },
  { to: '/memories', label: 'Memories', icon: Database },
  { to: '/search', label: 'Search & Ask', icon: Search },
  { to: '/graph', label: 'Graph', icon: Network },
  { to: '/evaluation', label: 'Evaluation', icon: BarChart3 },
  { to: '/system', label: 'System', icon: Activity },
]

function useTheme() {
  const [dark, setDark] = useState(() => {
    try { const saved = localStorage.getItem('cognimem_theme'); if (saved) return saved === 'dark' } catch { /* ignore */ }
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false
  })
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    try { localStorage.setItem('cognimem_theme', dark ? 'dark' : 'light') } catch { /* ignore */ }
  }, [dark])
  return [dark, setDark] as const
}

export function Layout() {
  const { user, signOut } = useAuth()
  const [dark, setDark] = useTheme()
  return (
    <div className="min-h-svh bg-background text-foreground">
      <header className="sticky top-0 z-10 border-b bg-background/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3">
          <span className="flex items-center gap-2 font-semibold"><Brain className="size-5" />CogniMem</span>
          <nav className="order-last flex w-full gap-1 overflow-x-auto md:order-none md:w-auto md:flex-1" aria-label="Main">
            {NAV.map(({ to, label, icon: Icon, end }) => (
              <NavLink key={to} to={to} end={end} className={({ isActive }) => cn(
                'flex shrink-0 items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm whitespace-nowrap text-muted-foreground hover:bg-muted hover:text-foreground',
                isActive && 'bg-muted text-foreground')}>
                <Icon className="size-4" />{label}
              </NavLink>
            ))}
          </nav>
          <span className="ml-auto hidden text-sm text-muted-foreground sm:inline md:ml-0">{user?.email}</span>
          <Button variant="ghost" size="icon" className="ml-auto sm:ml-0" aria-label={dark ? 'Light theme' : 'Dark theme'} onClick={() => setDark(!dark)}>
            {dark ? <Sun /> : <Moon />}
          </Button>
          <Button variant="ghost" size="icon" aria-label="Sign out" onClick={() => signOut()}><LogOut /></Button>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-6"><Outlet /></main>
    </div>
  )
}
