import { createClient } from '@supabase/supabase-js'

const url = import.meta.env.VITE_SUPABASE_URL
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

// Only create a client if both env vars are configured. When unset, the app
// runs in local-only mode (history in localStorage, no auth).
export const supabase = url && anonKey ? createClient(url, anonKey) : null
export const supabaseEnabled = !!supabase

/**
 * Why is Supabase unavailable? Used by the auth UI so a misconfigured build
 * shows a clear message instead of a button that silently does nothing.
 */
export const supabaseConfigError = supabase
  ? null
  : !url && !anonKey
    ? 'Sign-in is not configured: VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY were not set when the app was built.'
    : !url
      ? 'Sign-in is misconfigured: VITE_SUPABASE_URL is missing from the build.'
      : 'Sign-in is misconfigured: VITE_SUPABASE_ANON_KEY is missing from the build.'

/**
 * Translate raw Supabase auth errors into messages that say what to actually
 * do about them. Falls back to the original message.
 */
export function friendlyAuthError(err) {
  const msg = (err?.message || '').toLowerCase()

  if (msg.includes('failed to fetch') || msg.includes('networkerror') || msg.includes('load failed')) {
    return (
      "Couldn't reach the auth server. Check your internet connection — and if " +
      'this is a free Supabase project, it may be PAUSED after inactivity: open ' +
      'the Supabase dashboard and click "Restore project".'
    )
  }
  if (msg.includes('email not confirmed')) {
    return (
      'This email has not been confirmed yet. Check your inbox (and spam) for the ' +
      'confirmation link. If the link points to localhost, the Site URL in ' +
      'Supabase → Authentication → URL Configuration is wrong.'
    )
  }
  if (msg.includes('invalid login credentials')) {
    return 'Wrong email or password (or the account does not exist — use "Create account" first).'
  }
  if (msg.includes('provider is not enabled') || msg.includes('unsupported provider')) {
    return 'Google sign-in is not enabled for this project. Enable the Google provider in Supabase → Authentication → Providers, or use email/password.'
  }
  if (msg.includes('rate limit') || msg.includes('too many requests')) {
    return 'Too many attempts — wait a minute and try again.'
  }
  if (msg.includes('redirect') || msg.includes('not allowed')) {
    return (
      'This site is not in the allowed redirect list. Add ' +
      `${window.location.origin} to Supabase → Authentication → URL Configuration → Redirect URLs.`
    )
  }
  return err?.message || 'Something went wrong.'
}