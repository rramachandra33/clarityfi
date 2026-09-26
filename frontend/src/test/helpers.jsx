import { fireEvent, render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import App from '../App'

export const USER = { user_id: 'learner-204', synthetic: false, is_admin: false }
const res = (data, status = 200) => ({ ok: status < 400, status, json: async () => data, headers: { get: () => null } })

// A fetch stub that answers the sign-in endpoints and hands everything else to `inner` (default: the backend is offline).
export function stubBackend(inner) {
  const mock = vi.fn(async (url, options) => {
    if (url === '/api/auth/me') return res({ detail: 'Please sign in.', code: 'not_authenticated' }, 401)
    if (url === '/api/auth/config') return res({ min_password_length: 8, demo_login: false, demo_users: [], demo_password: null })
    if (url === '/api/auth/login' || url === '/api/auth/register') return res(USER, url.endsWith('register') ? 201 : 200)
    if (inner) return inner(url, options)
    throw new Error('offline')
  })
  vi.stubGlobal('fetch', mock)
  return mock
}

export async function signIn() {
  render(<App />)
  fireEvent.change(await screen.findByLabelText(/user id/i), { target: { value: 'learner-204' } })
  fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: 'a-decent-password' } })
  fireEvent.click(screen.getByRole('button', { name: /^enter clarityfi/i }))
  await screen.findByRole('heading', { name: 'Finance Q&A' })
}

