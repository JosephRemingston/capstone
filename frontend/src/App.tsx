import { lazy, Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { Skeleton } from '@/components/ui/skeleton'
import { Layout } from '@/components/Layout'
import { useAuth } from '@/auth/AuthContext'
import { Chat } from '@/pages/Chat'
import { Evaluation } from '@/pages/Evaluation'
import { Login } from '@/pages/Login'
import { Memories } from '@/pages/Memories'
import { MemoryDetail } from '@/pages/MemoryDetail'
import { Search } from '@/pages/Search'
import { System } from '@/pages/System'

// The force-directed graph library is large; load it only when the Graph page opens.
const Graph = lazy(() => import('@/pages/Graph').then(m => ({ default: m.Graph })))

function RequireAuth({ children }: { children: React.ReactNode }) {
  const { user } = useAuth()
  return user ? children : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route element={<RequireAuth><Layout /></RequireAuth>}>
        <Route index element={<Chat />} />
        <Route path="memories" element={<Memories />} />
        <Route path="memories/:id" element={<MemoryDetail />} />
        <Route path="search" element={<Search />} />
        <Route path="graph" element={<Suspense fallback={<Skeleton className="h-96 w-full" />}><Graph /></Suspense>} />
        <Route path="evaluation" element={<Evaluation />} />
        <Route path="system" element={<System />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
