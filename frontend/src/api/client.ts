// Fetch wrapper for the CogniMem API: Bearer token, `{success, data, error}` envelope, typed errors.

export class ApiError extends Error {
  status: number
  code: string
  requestId?: string
  details?: unknown
  constructor(status: number, code: string, message: string, requestId?: string, details?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.requestId = requestId
    this.details = details
  }
}

let token: string | null = null
let onUnauthorized: (() => void) | null = null

export function setToken(value: string | null) { token = value }
export function getToken() { return token }
export function setUnauthorizedHandler(handler: (() => void) | null) { onUnauthorized = handler }

function describe(details: unknown): string | null {
  // FastAPI validation errors: [{loc: [...,'field'], msg}] -> "field: msg"
  if (!Array.isArray(details) || !details.length) return null
  return details.map((d: { loc?: unknown[]; msg?: string }) => {
    const field = Array.isArray(d.loc) ? d.loc.filter(p => p !== 'body' && p !== 'query').join('.') : ''
    return field ? `${field}: ${d.msg}` : d.msg ?? ''
  }).join('; ')
}

export async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  if (init.body !== undefined && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
  if (token) headers.set('Authorization', `Bearer ${token}`)
  const response = await fetch(path, { ...init, headers })
  const body = await response.json().catch(() => null)
  if (!response.ok || body?.success === false) {
    if (response.status === 401 && token) onUnauthorized?.()
    const error = body?.error ?? {}
    const message = describe(error.details) ?? error.message ?? body?.detail ?? `HTTP ${response.status}`
    throw new ApiError(response.status, error.code ?? 'http_error', message, body?.request_id, error.details)
  }
  return body?.data as T
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, data: unknown) => request<T>(path, { method: 'POST', body: JSON.stringify(data) }),
  delete: <T>(path: string) => request<T>(path, { method: 'DELETE' }),
}

/** Downloads an authenticated file response (the export endpoint is not an envelope). */
export async function download(path: string, filename: string) {
  const response = await fetch(path, { headers: token ? { Authorization: `Bearer ${token}` } : {} })
  if (!response.ok) throw new ApiError(response.status, 'http_error', `Download failed (HTTP ${response.status})`)
  const url = URL.createObjectURL(await response.blob())
  const link = Object.assign(document.createElement('a'), { href: url, download: filename })
  link.click()
  URL.revokeObjectURL(url)
}
