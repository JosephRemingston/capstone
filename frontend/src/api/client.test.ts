import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, request, setToken, setUnauthorizedHandler } from './client'

const respond = (status: number, body: unknown) =>
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }))

afterEach(() => { vi.restoreAllMocks(); setToken(null); setUnauthorizedHandler(null) })

describe('request', () => {
  it('unwraps the success envelope and sends the bearer token', async () => {
    const fetch = respond(200, { success: true, data: { id: 1 }, error: null, request_id: 'r1' })
    setToken('tok')
    await expect(request('/api/v1/me')).resolves.toEqual({ id: 1 })
    expect(new Headers(fetch.mock.calls[0][1]!.headers).get('Authorization')).toBe('Bearer tok')
  })

  it('raises ApiError with the API message and request id', async () => {
    respond(422, { success: false, data: null, error: { code: 'http_error', message: 'limit must be between 1 and 100' }, request_id: 'r2' })
    const error = await request('/x').catch(e => e)
    expect(error).toBeInstanceOf(ApiError)
    expect(error).toMatchObject({ status: 422, code: 'http_error', message: 'limit must be between 1 and 100', requestId: 'r2' })
  })

  it('turns FastAPI validation details into field messages', async () => {
    respond(422, { success: false, error: { code: 'validation_error', message: 'Request validation failed',
      details: [{ loc: ['body', 'password'], msg: 'String should have at least 12 characters' }] }, request_id: 'r3' })
    await expect(request('/x', { method: 'POST', body: '{}' })).rejects.toThrow('password: String should have at least 12 characters')
  })

  it('signs out on 401 only when a session exists', async () => {
    const handler = vi.fn()
    setUnauthorizedHandler(handler)
    respond(401, { success: false, error: { code: 'http_error', message: 'Invalid token' }, request_id: 'r4' })
    await request('/x').catch(() => {})
    expect(handler).not.toHaveBeenCalled()   // e.g. a failed login is not an expired session
    setToken('tok')
    await request('/x').catch(() => {})
    expect(handler).toHaveBeenCalledOnce()
  })
})
