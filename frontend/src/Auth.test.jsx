import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { apiFetch } from './http'
import { USER } from './test/helpers'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

const res = (data, status = 200, headers = {}) => ({ ok: status < 400, status, json: async () => data, headers: { get: (k) => headers[k] ?? null } })

function backend({ me = 401, login = () => res(USER), register = () => res(USER, 201), config } = {}) {
  const calls = []
  vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => {
    calls.push({ url: String(url), method: options.method || 'GET', headers: options.headers || {}, body: options.body, credentials: options.credentials })
    if (url === '/api/auth/me') return me === 200 ? res(USER) : res({ detail: 'Please sign in.', code: 'not_authenticated' }, 401)
    if (url === '/api/auth/config') return res(config || { min_password_length: 8, demo_login: true, demo_users: ['learner-101', 'learner-105'], demo_password: 'clarity-demo' })
    if (url === '/api/auth/login') return login(options)
    if (url === '/api/auth/register') return register(options)
    if (url === '/api/auth/logout') return res(null, 204)
    if (url === '/api/health') return res({ openai_ready: true, knowledge_chunks: 5 })
    return res({}, 200)
  }))
  return calls
}

const fill = async (id, password) => {
  fireEvent.change(await screen.findByLabelText(/user id/i), { target: { value: id } })
  fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: password } })
}

describe('sign in', () => {
  it('shows the sign-in form when there is no session, then enters the workspace', async () => {
    const calls = backend()
    render(<App />)
    await fill('learner-204', 'a-decent-password')
    fireEvent.click(screen.getByRole('button', { name: /^enter clarityfi/i }))
    await screen.findByRole('heading', { name: 'Finance Q&A' })
    const call = calls.find((c) => c.url === '/api/auth/login')
    expect(JSON.parse(call.body)).toEqual({ user_id: 'learner-204', password: 'a-decent-password' })
    expect(call.headers['X-Clarity-Client']).toBe('web')
    expect(call.credentials).toBe('same-origin')
    expect(screen.getByText(/hello, learner-204/i)).toBeInTheDocument()
  })

  it('goes straight to the workspace when a session already exists (reload)', async () => {
    backend({ me: 200 })
    render(<App />)
    await screen.findByRole('heading', { name: 'Finance Q&A' })
    expect(screen.queryByLabelText(/^password/i)).not.toBeInTheDocument()
  })

  it('validates before sending and shows the server error without clearing the form', async () => {
    const calls = backend({ login: () => res({ detail: 'Incorrect user id or password.', code: 'invalid_credentials' }, 401) })
    render(<App />)
    fireEvent.click(await screen.findByRole('button', { name: /^enter clarityfi/i }))
    expect(screen.getByText(/at least 3 characters/i)).toBeInTheDocument()
    expect(screen.getByText(/enter your password/i)).toBeInTheDocument()
    expect(calls.some((c) => c.url === '/api/auth/login')).toBe(false)
    await fill('learner-204', 'wrong-password')
    fireEvent.click(screen.getByRole('button', { name: /^enter clarityfi/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Incorrect user id or password.')
    expect(screen.getByLabelText(/user id/i)).toHaveValue('learner-204')
  })

  it('shows the wait time when sign-in is throttled', async () => {
    backend({ login: () => res({ detail: 'Too many sign-in attempts. Please wait 300 seconds and try again.', code: 'too_many_attempts' }, 429, { 'Retry-After': '300' }) })
    render(<App />)
    await fill('learner-204', 'whatever-pass')
    fireEvent.click(screen.getByRole('button', { name: /^enter clarityfi/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/wait 300 seconds/)
  })

  it('creates an account with matching passwords of the required length', async () => {
    const calls = backend()
    render(<App />)
    fireEvent.click(await screen.findByRole('button', { name: /create an account/i }))
    await fill('brand-new', 'short')
    fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: 'different' } })
    fireEvent.click(screen.getByRole('button', { name: /^create account/i }))
    expect(screen.getByText(/use at least 8 characters/i)).toBeInTheDocument()
    expect(screen.getByText(/do not match/i)).toBeInTheDocument()
    expect(calls.some((c) => c.url === '/api/auth/register')).toBe(false)
    fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: 'a-decent-password' } })
    fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: 'a-decent-password' } })
    fireEvent.click(screen.getByRole('button', { name: /^create account/i }))
    await screen.findByRole('heading', { name: 'Finance Q&A' })
    expect(JSON.parse(calls.find((c) => c.url === '/api/auth/register').body)).toEqual({ user_id: 'brand-new', password: 'a-decent-password' })
  })

  it('offers demo accounts that fill in the documented demo password', async () => {
    backend()
    render(<App />)
    fireEvent.click(await screen.findByRole('button', { name: 'learner-105' }))
    expect(screen.getByLabelText(/user id/i)).toHaveValue('learner-105')
    expect(screen.getByLabelText(/^password/i)).toHaveValue('clarity-demo')
  })

  it('hides the demo box when demo login is off', async () => {
    backend({ config: { min_password_length: 8, demo_login: false, demo_users: [], demo_password: null } })
    render(<App />)
    await screen.findByLabelText(/user id/i)
    await waitFor(() => expect(screen.queryByText(/try a demo account/i)).not.toBeInTheDocument())
  })

  it('signs out through the server and returns to the sign-in form', async () => {
    const calls = backend({ me: 200 })
    render(<App />)
    fireEvent.click(await screen.findByRole('button', { name: /sign out/i }))
    await screen.findByLabelText(/^password/i)
    const logout = calls.find((c) => c.url === '/api/auth/logout')
    expect(logout.method).toBe('POST')
    expect(logout.headers['X-Clarity-Client']).toBe('web')
  })

  it('returns to sign-in with a message when any data call reports the session ended', async () => {
    backend({ me: 200 })
    render(<App />)
    await screen.findByRole('heading', { name: 'Finance Q&A' })
    vi.stubGlobal('fetch', vi.fn(async () => res({ detail: 'Please sign in.' }, 401)))
    await act(async () => { await apiFetch('/api/portfolio/learner-204') })
    expect(await screen.findByText(/your session ended/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/^password/i)).toBeInTheDocument()
  })

  it('hides knowledge uploads from non-administrators and shows them to administrators', async () => {
    backend({ me: 200 })
    render(<App />)
    await screen.findByText(/limited to administrators/i)
    expect(screen.queryByText(/add json knowledge/i)).not.toBeInTheDocument()
    cleanup()
    vi.stubGlobal('fetch', vi.fn(async (url) => (url === '/api/auth/me' ? res({ ...USER, is_admin: true }) : res({ openai_ready: true, knowledge_chunks: 5 }))))
    render(<App />)
    expect(await screen.findByText(/add json knowledge/i)).toBeInTheDocument()
  })
})

describe('apiFetch', () => {
  it('adds the app header and cookies, and does not announce a 401 from the auth endpoints themselves', async () => {
    const seen = []
    vi.stubGlobal('fetch', vi.fn(async (url, options) => { seen.push(options); return res({}, 401) }))
    const listener = vi.fn()
    window.addEventListener('clarity:unauthorized', listener)
    await apiFetch('/api/auth/login', { method: 'POST', headers: { 'Content-Type': 'application/json' } })
    expect(seen[0].headers).toEqual({ 'X-Clarity-Client': 'web', 'Content-Type': 'application/json' })
    expect(seen[0].credentials).toBe('same-origin')
    expect(listener).not.toHaveBeenCalled()
    await apiFetch('/api/portfolio/x')
    expect(listener).toHaveBeenCalledTimes(1)
    window.removeEventListener('clarity:unauthorized', listener)
  })
})
