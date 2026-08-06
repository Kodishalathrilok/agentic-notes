import { useState, useEffect, useRef } from 'react'
import { supabase, supabaseConfigError, friendlyAuthError, providerEnabled } from '../lib/supabase'

/**
 * Glassmorphism sign in / sign up card, embedded in the landing page.
 * - Sliding pill tab switcher (Sign in <-> Create account)
 * - Word-by-word heading animation on tab change
 * - Google OAuth + email/password (same Supabase logic as AuthModal)
 * - `pulse` briefly highlights the card after the hero "Sign in" scroll
 */

function StaggerText({ text, className = '' }) {
  // Re-animates whenever `text` changes (keyed remount from parent).
  const words = text.split(' ')
  return (
    <span className={className}>
      {words.map((w, i) => (
        <span key={i} className="inline-block overflow-hidden pb-[0.1em] -mb-[0.1em] align-bottom">
          <span
            className="inline-block animate-slide-up will-change-transform"
            style={{ animationDelay: `${i * 70}ms`, animationFillMode: 'backwards' }}
          >
            {w}
            {i < words.length - 1 ? '\u00A0' : ''}
          </span>
        </span>
      ))}
    </span>
  )
}

export default function AuthCard({ onAuthed, pulse = false }) {
  const [mode, setMode] = useState('signin') // 'signin' | 'signup'
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [notice, setNotice] = useState(null)
  const emailRef = useRef(null)

  // When the hero button scrolls us into view, focus the email field once
  // the smooth scroll has (roughly) finished.
  useEffect(() => {
    if (!pulse) return
    const t = setTimeout(() => emailRef.current?.focus({ preventScroll: true }), 650)
    return () => clearTimeout(t)
  }, [pulse])

  const switchMode = (m) => {
    if (m === mode) return
    setMode(m)
    setError(null)
    setNotice(null)
  }

  const google = async () => {
    if (!supabase) {
      setError(supabaseConfigError)
      return
    }
    setError(null)
    // Check before navigating — a disabled provider would otherwise dump the
    // user on GoTrue's raw JSON error, where nothing here can catch it.
    if (!(await providerEnabled('google'))) {
      setError(friendlyAuthError({ message: 'Unsupported provider: provider is not enabled' }))
      return
    }
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: { redirectTo: window.location.origin },
    })
    if (error) setError(friendlyAuthError(error))
  }

  const submit = async (e) => {
    e.preventDefault()
    if (!supabase) {
      setError(supabaseConfigError)
      return
    }
    setBusy(true)
    setError(null)
    setNotice(null)
    try {
      if (mode === 'signup') {
        const { data, error } = await supabase.auth.signUp({ email, password })
        if (error) throw error
        if (data.session) {
          onAuthed && onAuthed()
        } else {
          setNotice('Account created — check your email to confirm, then sign in.')
          setMode('signin')
        }
      } else {
        const { error } = await supabase.auth.signInWithPassword({ email, password })
        if (error) throw error
        onAuthed && onAuthed()
      }
    } catch (err) {
      setError(friendlyAuthError(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div
      className={`relative mx-auto w-full max-w-md rounded-[28px] border border-white/60 bg-white/45 p-7 shadow-lift backdrop-blur-2xl transition-shadow duration-700 sm:p-8 ${
        pulse ? 'ring-2 ring-brand-500/50 shadow-[0_0_60px_-12px_rgba(0,0,0,0.25)]' : ''
      }`}
    >
      {/* soft internal glow */}
      <div className="pointer-events-none absolute -top-10 right-0 h-40 w-40 rounded-full bg-brand-500/15 blur-3xl" aria-hidden />
      <div className="pointer-events-none absolute -bottom-8 -left-6 h-36 w-36 rounded-full bg-rose-300/25 blur-3xl" aria-hidden />

      {/* Tab switcher with sliding pill */}
      <div className="relative mb-6 grid grid-cols-2 rounded-full bg-espresso-900/[0.06] p-1 text-sm font-semibold">
        <span
          className="absolute inset-y-1 w-[calc(50%-4px)] rounded-full bg-white shadow-soft transition-transform duration-300 ease-[cubic-bezier(.2,.6,.2,1)]"
          style={{ transform: mode === 'signin' ? 'translateX(4px)' : 'translateX(calc(100% + 4px))' }}
          aria-hidden
        />
        <button
          type="button"
          onClick={() => switchMode('signin')}
          className={`relative z-10 rounded-full py-2 transition-colors ${mode === 'signin' ? 'text-espresso-900' : 'text-espresso-500 hover:text-espresso-700'}`}
        >
          Sign in
        </button>
        <button
          type="button"
          onClick={() => switchMode('signup')}
          className={`relative z-10 rounded-full py-2 transition-colors ${mode === 'signup' ? 'text-espresso-900' : 'text-espresso-500 hover:text-espresso-700'}`}
        >
          Create account
        </button>
      </div>

      <h3 key={mode} className="mb-1 font-display text-2xl font-bold text-espresso-900">
        <StaggerText text={mode === 'signin' ? 'Welcome back.' : 'Join the study lab.'} />
      </h3>

      {!supabase && (
        <div className="mb-4 rounded-lg bg-amber-50/90 px-3 py-2 text-xs text-amber-700">
          ⚠️ {supabaseConfigError}
        </div>
      )}
      <p key={mode + '-sub'} className="mb-5 animate-slide-up text-sm text-espresso-500" style={{ animationDelay: '160ms', animationFillMode: 'backwards' }}>
        {mode === 'signin'
          ? 'Sign in to generate notes, quizzes and flashcards.'
          : 'Access is limited to approved accounts.'}
      </p>

      <button
        onClick={google}
        type="button"
        className="mb-3 flex w-full items-center justify-center gap-2.5 rounded-xl border border-white/70 bg-white/70 px-4 py-2.5 text-sm font-semibold text-espresso-700 shadow-soft transition-all hover:-translate-y-0.5 hover:bg-white"
      >
        <svg className="h-4 w-4" viewBox="0 0 24 24" aria-hidden="true">
          <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.27-4.74 3.27-8.1z" />
          <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84A11 11 0 0 0 12 23z" />
          <path fill="#FBBC05" d="M5.84 14.1a6.6 6.6 0 0 1 0-4.2V7.06H2.18a11 11 0 0 0 0 9.88l3.66-2.84z" />
          <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84C6.71 7.31 9.14 5.38 12 5.38z" />
        </svg>
        Continue with Google
      </button>

      <div className="mb-3 flex items-center gap-3 text-xs text-espresso-400">
        <span className="h-px flex-1 bg-espresso-900/10" />
        or with email
        <span className="h-px flex-1 bg-espresso-900/10" />
      </div>

      <form onSubmit={submit} className="space-y-3">
        <input
          ref={emailRef}
          type="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Email"
          className="field bg-white/70 backdrop-blur"
        />
        <input
          type="password"
          required
          minLength={6}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password (min 6 chars)"
          className="field bg-white/70 backdrop-blur"
        />

        {error && (
          <div className="animate-slide-up rounded-lg bg-red-50/90 px-3 py-2 text-xs text-red-600">{error}</div>
        )}
        {notice && (
          <div className="animate-slide-up rounded-lg bg-green-50/90 px-3 py-2 text-xs text-green-700">{notice}</div>
        )}

        <button type="submit" disabled={busy} className="btn-primary w-full py-2.5">
          {busy ? '…' : mode === 'signup' ? 'Create account' : 'Sign in'}
        </button>
      </form>

      <p className="mt-4 text-center text-[11px] leading-relaxed text-espresso-400">
        Only authorized accounts can use the app — new sign-ups may need the owner's approval.
      </p>
    </div>
  )
}