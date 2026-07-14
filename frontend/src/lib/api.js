import { supabase } from './supabase'

/**
 * Attach the signed-in user's Supabase access token to API requests.
 * The backend verifies this token on every expensive endpoint, so
 * unauthenticated callers can't spend model credits.
 *
 * When Supabase isn't configured (local-only mode), this degrades to a
 * plain fetch and the backend runs with auth disabled — nothing breaks.
 */
export async function authHeaders() {
  if (!supabase) return {}
  try {
    const { data } = await supabase.auth.getSession()
    const token = data?.session?.access_token
    return token ? { Authorization: `Bearer ${token}` } : {}
  } catch {
    return {}
  }
}

export async function apiFetch(url, opts = {}) {
  const headers = { ...(opts.headers || {}), ...(await authHeaders()) }
  return fetch(url, { ...opts, headers })
}
