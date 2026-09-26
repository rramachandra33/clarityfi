// Every request goes through here so it carries the sign-in cookie and the header the server requires on
// state-changing requests (a cross-site form post cannot set it). A 401 on any data call means the session ended.
export const CLIENT_HEADERS = { 'X-Clarity-Client': 'web' }
export const UNAUTHORIZED_EVENT = 'clarity:unauthorized'

export async function apiFetch(url, options = {}) {
  const response = await fetch(url, { credentials: 'same-origin', ...options, headers: { ...CLIENT_HEADERS, ...(options.headers || {}) } })
  if (response.status === 401 && !String(url).startsWith('/api/auth/') && typeof window !== 'undefined') {
    window.dispatchEvent(new Event(UNAUTHORIZED_EVENT))
  }
  return response
}
