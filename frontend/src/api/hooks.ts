import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from './client'
import type { AskResult, AuthResult, Category, GraphResult, Health, MemoryRecord, ModelReport, SearchHit, SearchMode, Snapshot, SnapshotInfo, Tier } from './types'

const V1 = '/api/v1'

export interface MemoryFilters { tier?: Tier | ''; category?: Category | ''; session_id?: string; include_history?: boolean; limit?: number }

function query(params: Record<string, string | number | boolean | undefined | null>) {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) if (value !== undefined && value !== null && value !== '') search.set(key, String(value))
  const text = search.toString()
  return text ? `?${text}` : ''
}

export const useSignup = () => useMutation({ mutationFn: (b: { email: string; password: string }) => api.post<AuthResult>(`${V1}/auth/signup`, b) })
export const useLogin = () => useMutation({ mutationFn: (b: { email: string; password: string }) => api.post<AuthResult>(`${V1}/auth/login`, b) })

export const useMemories = (filters: MemoryFilters) =>
  useQuery({ queryKey: ['memories', filters], queryFn: () => api.get<MemoryRecord[]>(`${V1}/memories${query({ limit: 100, ...filters })}`) })

export const useMemory = (id: string) => useQuery({ queryKey: ['memory', id], queryFn: () => api.get<MemoryRecord>(`${V1}/memories/${id}`) })
export const useHistory = (id: string) => useQuery({ queryKey: ['history', id], queryFn: () => api.get<MemoryRecord[]>(`${V1}/memories/${id}/history`) })

export function useCreateMemory() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: (b: { content: string; session_id: string }) => api.post<MemoryRecord>(`${V1}/memories`, b),
    onSuccess: () => { client.invalidateQueries({ queryKey: ['memories'] }); client.invalidateQueries({ queryKey: ['graph'] }) },
  })
}

export interface SearchParams { query: string; mode: SearchMode; limit: number; rerank?: boolean | null }
export const useSearch = () => useMutation({ mutationFn: (b: SearchParams) => api.post<(SearchHit | MemoryRecord)[]>(`${V1}/memories/search`, b) })

export interface AskParams { query: string; mode: Exclude<SearchMode, 'recency'>; generator: 'extractive' | 'gemini'; abstain: boolean; as_of?: string; known_at?: string }
export const useAsk = () => useMutation({ mutationFn: (b: AskParams) => api.post<AskResult>(`${V1}/memories/ask`, b) })

export interface GraphParams { query?: string; as_of?: string; known_at?: string; predicate?: string; hops?: number }
export const useGraph = (p: GraphParams) => useQuery({ queryKey: ['graph', p], queryFn: () => api.get<GraphResult>(`${V1}/graph${query({ ...p })}`) })

export const useHealth = () => useQuery({ queryKey: ['health'], queryFn: () => api.get<Health>('/health/ready'), retry: false, refetchInterval: 30_000 })
export const useMetrics = () => useQuery({
  queryKey: ['metrics'],
  queryFn: async () => (await fetch('/metrics')).text(),
  refetchInterval: 30_000,
})
export const useDeleteAccount = () => useMutation({ mutationFn: () => api.delete<unknown>(`${V1}/me`) })

export const useSnapshots = () => useQuery({ queryKey: ['snapshots'], queryFn: () => api.get<SnapshotInfo[]>(`${V1}/evaluation/snapshots`), staleTime: 60_000 })
export const useSnapshot = (tag: string | undefined) => useQuery({
  queryKey: ['snapshot', tag], enabled: !!tag, staleTime: 60_000,
  queryFn: () => api.get<Snapshot>(`${V1}/evaluation/snapshots/${encodeURIComponent(tag!)}`),
})
export const useModelReport = () => useQuery({ queryKey: ['model-report'], queryFn: () => api.get<ModelReport>(`${V1}/evaluation/model`), staleTime: 60_000 })
