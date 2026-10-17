import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ApiError } from '@/api/client'
import { AuthProvider } from '@/auth/AuthContext'
import App from './App'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    // Client errors (validation, auth, not found) are not retried; network or server errors once.
    queries: { retry: (count, error) => !(error instanceof ApiError && error.status < 500) && count < 1, staleTime: 10_000 },
  },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter basename={import.meta.env.BASE_URL.replace(/\/$/, '')}>
        <AuthProvider><App /></AuthProvider>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
)
