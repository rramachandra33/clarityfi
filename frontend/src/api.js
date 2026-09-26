import { useCallback, useEffect, useRef, useState } from 'react'
import { apiFetch } from './http'

export class ApiError extends Error {
  constructor(message, status, code) {
    super(message)
    this.status = status
    this.code = code
  }
}

// Every endpoint returns {detail, code} on failure; 429 also carries Retry-After.
export async function api(path, { method = 'GET', body, signal } = {}) {
  let response
  try {
    response = await apiFetch(path, {
      method,
      signal,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  } catch (error) {
    if (error?.name === 'AbortError') throw error
    throw new ApiError('Could not reach the ClarityFI service. Is the backend running?', 0, 'offline')
  }
  if (response.status === 204) return null
  let data = null
  try { data = await response.json() } catch { /* non-JSON error body */ }
  if (!response.ok) {
    let message = typeof data?.detail === 'string' ? data.detail : null
    if (!message && Array.isArray(data?.detail)) message = data.detail.map((d) => d.msg).join('; ')
    if (response.status === 429) {
      const wait = response.headers?.get?.('Retry-After')
      message = `Too many requests. Please wait ${wait || 'a few'} seconds and try again.`
    }
    throw new ApiError(message || `Request failed (${response.status})`, response.status, data?.code)
  }
  return data
}

// Loads `path` (null = skip) and reloads when `path` or `nonce` changes. Stale responses are ignored.
export function useApi(path, nonce = 0) {
  const [state, setState] = useState({ loading: Boolean(path), data: null, error: '' })
  useEffect(() => {
    if (!path) { setState({ loading: false, data: null, error: '' }); return undefined }
    const controller = new AbortController()
    setState((current) => ({ ...current, loading: true, error: '' }))
    api(path, { signal: controller.signal })
      .then((data) => setState({ loading: false, data, error: '' }))
      .catch((error) => { if (error?.name !== 'AbortError') setState({ loading: false, data: null, error: error.message }) })
    return () => controller.abort()
  }, [path, nonce])
  return state
}

// Runs an async action, tracking pending/error, ignoring results after unmount.
export function useAction() {
  const [state, setState] = useState({ pending: false, error: '' })
  const alive = useRef(true)
  // Set true on every mount: React StrictMode mounts, unmounts and re-mounts in development.
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])
  const run = useCallback(async (fn) => {
    setState({ pending: true, error: '' })
    try {
      const result = await fn()
      if (alive.current) setState({ pending: false, error: '' })
      return result
    } catch (error) {
      if (alive.current) setState({ pending: false, error: error.message })
      return undefined
    }
  }, [])
  return [state, run]
}

export const money = (value, digits = 0) =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: digits }).format(Number(value) || 0)

export const pct = (value, digits = 1) => (value === null || value === undefined ? '—' : `${Number(value).toFixed(digits)}%`)
