import { useState, useEffect, useRef } from 'react'
import Icon from './Icons'
import TextEffect from './TextEffect'
import MenuOverlay from './MenuOverlay'
import AuthCard from './AuthCard'

const GITHUB_URL = 'https://github.com/Kodishalathrilok/agentic-notes'

const PIPELINE = [
  { name: 'Plan', desc: 'Outlines the topic & checklist' },
  { name: 'Write', desc: 'Drafts notes, streamed live' },
  { name: 'Critique', desc: 'Checks faithfulness vs source' },
  { name: 'Revise', desc: 'Fixes & re-checks (loop)' },
  { name: 'Quiz', desc: 'MCQs + verified answer key' },
  { name: 'Flashcards', desc: 'Spaced-repetition cards' },
]

const FEATURES = [
  { icon: Icon.Refresh, title: 'Self-correcting agents', body: 'A Plan → Write → Critique → Revise loop where agents review and rewrite each other until the notes pass a quality bar.' },
  { icon: Icon.Check, title: 'Faithfulness-checked', body: 'A grounded critique compares every claim against the source and removes anything fabricated — fighting hallucination by design.' },
  { icon: Icon.Zap, title: 'Real-time streaming', body: 'Notes type out token-by-token over SSE while a live pipeline view shows each agent working.' },
  { icon: Icon.Book, title: 'Multimodal input', body: 'Text, PDF, a photo of a page (vision OCR), a web link, a YouTube transcript, or audio — all become study notes.' },
  { icon: Icon.Star, title: 'Spaced repetition', body: 'Flashcards with Again / Good / Easy scheduling, a due-cards filter, and Anki-compatible CSV export.' },
  { icon: Icon.Sparkles, title: 'Ask your notes', body: 'Select any text to explain or rewrite it inline, grounded in the notes you just generated.' },
]

const QUALITY = [
  ['Grounded critique', 'Judges notes against the original source for faithfulness & coverage.'],
  ['Iterative revision', 'Re-critiques after each revision and keeps the best-scoring version.'],
  ['Automated eval suite', 'An LLM-judge harness measures faithfulness, coverage & quiz accuracy.'],
  ['Multi-provider router', 'Groq → Gemini → Ollama with automatic failover on rate limits.'],
  ['RAG + citations', 'Chunked retrieval (neural or TF-IDF) with clickable source citations.'],
  ['Answer verification', 'Generated quiz keys are re-checked against the notes before display.'],
]

const STACK = ['FastAPI', 'SSE Streaming', 'React', 'Vite', 'Tailwind', 'Groq · Llama', 'Gemini', 'Supabase', 'Docker']

const SOURCES = ['lectures', 'PDFs', 'YouTube', 'photos', 'audio']

// ----- small animation helpers -----
function Reveal({ children, delay = 0, className = '' }) {
  const ref = useRef(null)
  const [shown, setShown] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setShown(true)
          obs.disconnect()
        }
      },
      { threshold: 0.12 }
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [])
  return (
    <div
      ref={ref}
      style={{ transitionDelay: `${delay}ms` }}
 className={`transition-all duration-[900ms] ease-[cubic-bezier(.2,.6,.2,1)] ${
        shown
          ? 'translate-y-0 scale-100 opacity-100 blur-0'
          : 'translate-y-12 scale-[0.97] opacity-0 blur-sm'
      } ${className}`}
    >
      {children}
    </div>
  )
}

// Webild-style masked word-by-word title reveal, triggered on scroll
function AnimatedTitle({ children, className = '', as: Tag = 'h2' }) {
  const ref = useRef(null)
  const [shown, setShown] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setShown(true)
          obs.disconnect()
        }
      },
      { threshold: 0.35 }
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [])
  const words = String(children).split(' ')
  return (
    <Tag ref={ref} className={className}>
      {words.map((w, i) => (
        <span key={i} className="inline-block overflow-hidden pb-[0.12em] -mb-[0.12em] align-bottom">
          <span
            className="inline-block will-change-transform"
            style={{
              transform: shown ? 'translateY(0)' : 'translateY(110%)',
              opacity: shown ? 1 : 0,
              filter: shown ? 'blur(0)' : 'blur(4px)',
              transition: 'transform .7s cubic-bezier(.2,.6,.2,1), opacity .7s ease, filter .7s ease',
              transitionDelay: `${i * 80}ms`,
            }}
          >
            {w}
            {i < words.length - 1 ? '\u00A0' : ''}
          </span>
        </span>
      ))}
    </Tag>
  )
}

// 3D tilt + cursor glow card (interactive hover)
function TiltCard({ children, className = '', max = 7 }) {
  const ref = useRef(null)
  const onMove = (e) => {
    const el = ref.current
    if (!el) return
    const r = el.getBoundingClientRect()
    const px = (e.clientX - r.left) / r.width - 0.5
    const py = (e.clientY - r.top) / r.height - 0.5
    el.style.setProperty('--rx', `${(-py * max).toFixed(2)}deg`)
    el.style.setProperty('--ry', `${(px * max).toFixed(2)}deg`)
    el.style.setProperty('--gx', `${e.clientX - r.left}px`)
    el.style.setProperty('--gy', `${e.clientY - r.top}px`)
  }
  const onLeave = () => {
    const el = ref.current
    if (!el) return
    el.style.setProperty('--rx', '0deg')
    el.style.setProperty('--ry', '0deg')
  }
  return (
    <div ref={ref} onMouseMove={onMove} onMouseLeave={onLeave} className={`tilt-card ${className}`}>
      {children}
    </div>
  )
}

function RotatingWord({ words }) {
  const [i, setI] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setI((x) => (x + 1) % words.length), 2000)
    return () => clearInterval(t)
  }, [words.length])
  return (
    <span
      key={i}
 className="inline-block animate-slide-up bg-gradient-to-r from-espresso-700 to-espresso-900 bg-clip-text font-bold text-transparent"
    >
      {words[i]}
    </span>
  )
}

function Nav({ onLaunch, onEval, user, supabaseEnabled, onSignIn, onSignOut, onMenu }) {
  return (
    <header className="fixed inset-x-0 top-0 z-40 px-3 pt-3 sm:px-4 sm:pt-4">
      <div className="glass-pill flex w-full items-center justify-between rounded-full py-2 pl-3 pr-2 sm:pl-4">
        <div className="flex items-center gap-2.5">
          <span className="flex h-9 w-9 items-center justify-center rounded-full bg-gradient-to-br from-espresso-700 to-espresso-900 text-white shadow-lift">
            <Icon.Book className="h-5 w-5" />
          </span>
          <span className="text-base font-extrabold tracking-tight text-espresso-900">Agentic Notes</span>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={onLaunch} className="btn-primary rounded-full">
            Launch App
            <span aria-hidden>→</span>
          </button>
          <button
            onClick={onMenu}
            aria-label="Open menu"
            className="flex h-11 w-11 items-center justify-center rounded-full bg-espresso-900 text-cream shadow-soft transition-transform hover:scale-105"
          >
            <Icon.Menu className="h-5 w-5" />
          </button>
        </div>
      </div>
    </header>
  )
}

export default function Landing({ onLaunch, onEval, user, supabaseEnabled, onSignIn, onSignOut, authRequired = false }) {
  const [menuOpen, setMenuOpen] = useState(false)
  const heroRef = useRef(null)

  // ----- auth gate: launching requires a signed-in user --------------------
  const [authPulse, setAuthPulse] = useState(false)
  const scrollToAuth = () => {
    document.getElementById('auth-section')?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    setAuthPulse(true)
    setTimeout(() => setAuthPulse(false), 2200)
  }
  // Every "launch" entry point funnels through this: signed out -> glide down
  // to the sign-in card instead of opening the app.
  const launch = () => (authRequired ? scrollToAuth() : onLaunch())

  // Auto-cycle the active pipeline step in "How it works"
  const [activeStep, setActiveStep] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setActiveStep((s) => (s + 1) % PIPELINE.length), 1700)
    return () => clearInterval(t)
  }, [])

  // Liquid hover reveal — lerped cursor position + gentle wobble drives the
  // mask on the colored art layer (see .hero-art-color in index.css)
  useEffect(() => {
    const el = heroRef.current
    if (!el) return
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    let raf
    const pos = { x: el.offsetWidth / 2, y: el.offsetHeight * 0.4 }
    const tgt = { x: pos.x, y: pos.y }
    let r = 0
    let tr = 0
    const onMove = (e) => {
      const rect = el.getBoundingClientRect()
      tgt.x = e.clientX - rect.left
      tgt.y = e.clientY - rect.top
      tr = 300
      // instant vars for the existing spotlight glow
      el.style.setProperty('--mx', `${tgt.x}px`)
      el.style.setProperty('--my', `${tgt.y}px`)
    }
    const onLeave = () => {
      tr = 0
    }
    const tick = (t) => {
      // trailing lerp = the "liquid" flow; wobble keeps the blob organic
      pos.x += (tgt.x - pos.x) * 0.095
      pos.y += (tgt.y - pos.y) * 0.095
      r += (tr - r) * 0.055
      const w = t * 0.0016
      const wob = reduced ? 1 : 1 + 0.05 * Math.sin(w * 2.1)
      el.style.setProperty('--hx', pos.x.toFixed(1))
      el.style.setProperty('--hy', pos.y.toFixed(1))
      el.style.setProperty('--hr', (r * wob).toFixed(1))
      el.style.setProperty('--ox', (reduced ? 0 : Math.cos(w * 0.9) * r * 0.28).toFixed(1))
      el.style.setProperty('--oy', (reduced ? 0 : Math.sin(w * 1.3) * r * 0.24).toFixed(1))
      raf = requestAnimationFrame(tick)
    }
    el.addEventListener('mousemove', onMove)
    el.addEventListener('mouseleave', onLeave)
    raf = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(raf)
      el.removeEventListener('mousemove', onMove)
      el.removeEventListener('mouseleave', onLeave)
    }
  }, [])
  return (
    <div className="min-h-screen overflow-x-hidden bg-latte-100 text-espresso-800">
      <Nav
        onLaunch={launch}
        onEval={onEval}
        user={user}
        supabaseEnabled={supabaseEnabled}
        onSignIn={onSignIn}
        onSignOut={onSignOut}
        onMenu={() => setMenuOpen(true)}
      />

      <MenuOverlay
        open={menuOpen}
        onClose={() => setMenuOpen(false)}
        items={[
          { label: 'Launch App', onClick: launch },
          { label: 'Eval', onClick: onEval },
          { label: 'GitHub', href: GITHUB_URL },
          ...(supabaseEnabled
            ? [user ? { label: 'Sign out', onClick: onSignOut } : { label: 'Sign in', onClick: scrollToAuth }]
            : []),
        ]}
      />

      {/* Hero — sketch art base, colored art revealed under the cursor */}
      <section ref={heroRef} className="isolate relative flex min-h-screen items-center overflow-hidden">
        {/* Layer 1: hand-drawn sketch */}
        <div className="hero-art hero-art-sketch" aria-hidden />
        {/* Layer 2: colored version, liquid-masked around the cursor */}
        <div className="hero-art hero-art-color" aria-hidden />
        <div className="spotlight pointer-events-none absolute inset-0 -z-[5]" />
        <div className="pointer-events-none absolute inset-0 -z-10">
          <div className="animate-float-blob absolute left-[15%] top-[-8%] h-[360px] w-[360px] rounded-full bg-brand-500/25 blur-3xl dark:bg-brand-600/20" />
          <div
 className="animate-float-blob absolute right-[10%] top-[10%] h-[320px] w-[320px] rounded-full bg-rose-300/40 blur-3xl"
            style={{ animationDelay: '-6s' }}
          />
        </div>

        <div className="mx-auto flex w-full max-w-5xl flex-col items-center px-4 py-28 text-center sm:py-32">
          <h1 className="mx-auto font-display text-6xl font-bold leading-[1.05] tracking-tight text-espresso-900 sm:text-7xl md:text-8xl">
            <TextEffect per="word" preset="slide" as="span" className="block sm:whitespace-nowrap" stagger={0.08} startDelay={0.05}>
              Generate notes
            </TextEffect>
            <TextEffect
              per="word"
              preset="slide"
              as="span"
 className="block sm:whitespace-nowrap pb-2"
              unitClassName="bg-gradient-to-r from-brand-600 via-rose-500 to-brand-700 bg-clip-text text-transparent pb-[0.2em] -mb-[0.2em]"
              stagger={0.08}
              startDelay={0.29}
            >
              from anything.
            </TextEffect>
          </h1>

          <p
 className="mt-6 animate-slide-up text-lg text-espresso-600 sm:text-xl"
            style={{ animationDelay: '300ms' }}
          >
            Turn <RotatingWord words={SOURCES} /> into exam-ready notes.
          </p>

          <div
 className="mt-9 flex animate-slide-up flex-wrap items-center justify-center gap-3"
            style={{ animationDelay: '420ms' }}
          >
            <button onClick={launch} className="btn-primary px-6 py-3 text-base">
              <Icon.Sparkles className="h-5 w-5" />
              Launch the app
            </button>
            {supabaseEnabled && (
              user ? (
                <button onClick={onSignOut} className="btn-ghost px-6 py-3 text-base">
                  Sign out
                </button>
              ) : (
                <button onClick={scrollToAuth} className="btn-ghost group px-6 py-3 text-base">
                  Sign in
                  <span aria-hidden className="inline-block transition-transform duration-300 group-hover:translate-y-0.5">↓</span>
                </button>
              )
            )}
          </div>
          <p
 className="mt-4 animate-slide-up text-xs text-slate-400"
            style={{ animationDelay: '520ms' }}
          >
            Powered by Llama on Groq, Gemini & local Ollama — with automatic failover
          </p>
        </div>
      </section>

      {/* How it works */}
      <section className="mx-auto max-w-6xl px-5 py-24">
        <div className="mb-10 text-center">
          <AnimatedTitle className="display-title">How it works</AnimatedTitle>
          <Reveal delay={250}>
            <p className="mt-2 text-espresso-500">
              Six specialized agents, with a self-correcting critique loop at the core.
            </p>
          </Reveal>
        </div>

        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-6">
          {PIPELINE.map((step, i) => (
            <Reveal key={step.name} delay={i * 90}>
              <TiltCard
 className={`group relative h-full cursor-default rounded-2xl glass-card p-5 pb-7 text-center transition-all duration-500 ${
                  i === activeStep ? '-translate-y-1.5 shadow-lift ring-2 ring-brand-500/35' : 'hover:-translate-y-1'
                }`}
              >
                <div
 className={`mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-2xl text-sm font-bold transition-all duration-500 ${
                    i === activeStep
                      ? 'scale-110 bg-gradient-to-br from-brand-500 to-rose-400 text-white shadow-lift'
                      : 'bg-espresso-900/[0.06] text-espresso-700 group-hover:bg-espresso-900/10'
                  }`}
                >
                  {i + 1}
                </div>
                <div className="text-sm font-bold text-espresso-900">{step.name}</div>
                <div className="mt-1 text-xs leading-relaxed text-espresso-500">{step.desc}</div>
                {(step.name === 'Critique' || step.name === 'Revise') && (
                  <span
 className="absolute right-2.5 top-2.5 text-espresso-400/70 transition-colors duration-500"
                    title="Part of the self-correcting loop"
                    aria-hidden
                  >
                    <Icon.Refresh className={`h-3.5 w-3.5 ${i === activeStep ? 'animate-spin text-brand-600' : ''}`} />
                  </span>
                )}
                {/* progress underline */}
                <span
 className={`absolute inset-x-5 bottom-3 h-0.5 origin-left rounded-full bg-gradient-to-r from-brand-500 to-rose-400 transition-transform duration-500 ${
                    i === activeStep ? 'scale-x-100' : 'scale-x-0 group-hover:scale-x-100'
                  }`}
                  aria-hidden
                />
              </TiltCard>
            </Reveal>
          ))}
        </div>

        <Reveal delay={200} className="glass-card mx-auto mt-8 max-w-md rounded-full px-5 py-3 text-center text-sm text-espresso-700">
          <Icon.Refresh className="mr-1.5 inline h-4 w-4 animate-[spin_4s_linear_infinite] text-brand-600" />
          Critique → Revise repeats until the notes are faithful and complete.
        </Reveal>
      </section>

      {/* Features */}
      <section className="bg-espresso-900/[0.04] py-24">
        <div className="mx-auto max-w-6xl px-4">
          <AnimatedTitle className="display-title mb-10 text-center">Everything you need to study</AnimatedTitle>
          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {FEATURES.map((f, i) => (
              <Reveal key={f.title} delay={(i % 3) * 90}>
                <TiltCard className="group glass-card h-full rounded-3xl p-6">
                  <span className="mb-4 flex h-11 w-11 items-center justify-center rounded-xl bg-brand-500/15 text-brand-700 transition-all duration-300 group-hover:scale-110 group-hover:bg-brand-500/25 group-hover:rotate-3">
                    <f.icon className="h-5 w-5" />
                  </span>
                  <h3 className="mb-1.5 text-base font-bold text-espresso-900">{f.title}</h3>
                  <p className="text-sm leading-relaxed text-espresso-600">{f.body}</p>
                </TiltCard>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* Engineered for quality */}
      <section className="mx-auto max-w-6xl px-5 py-24">
        <div className="mb-10 text-center">
          <AnimatedTitle className="display-title">Engineered for quality</AnimatedTitle>
          <Reveal delay={250}>
            <p className="mt-2 text-espresso-500">
              The parts that separate a real system from an API wrapper.
            </p>
          </Reveal>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {QUALITY.map(([title, body], i) => (
            <Reveal key={title} delay={(i % 3) * 90}>
              <TiltCard className="group glass-card h-full rounded-3xl p-6">
                <div className="mb-1.5 flex items-center gap-2">
                  <Icon.Check className="h-4 w-4 text-green-500 transition-transform duration-300 group-hover:scale-125" />
                  <h3 className="text-sm font-bold text-espresso-900">{title}</h3>
                </div>
                <p className="text-sm leading-relaxed text-espresso-600">{body}</p>
              </TiltCard>
            </Reveal>
          ))}
        </div>

        <div className="mt-12 flex flex-wrap items-center justify-center gap-2">
          {STACK.map((t, i) => (
            <Reveal key={t} delay={i * 60}>
              <span className="chip glass-card border border-white/60 text-espresso-600 transition-transform duration-200 hover:scale-110">
                {t}
              </span>
            </Reveal>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="px-4 pb-20">
        <Reveal className="mx-auto max-w-4xl">
          <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-espresso-800 to-espresso-900 px-6 py-14 text-center text-white shadow-lift">
            <div className="animate-float-blob pointer-events-none absolute -right-10 -top-10 h-48 w-48 rounded-full bg-white/10 blur-2xl" />
            <AnimatedTitle className="font-display text-4xl font-bold leading-tight text-white sm:text-6xl">
              Ready to study smarter?
            </AnimatedTitle>
            <p className="mx-auto mt-2 max-w-xl text-latte-200">
              Turn any source into structured notes, a quiz, and flashcards in under a minute.
            </p>
            <button
              onClick={launch}
 className="mt-7 inline-flex items-center gap-2 rounded-xl bg-white px-6 py-3 text-base font-bold text-brand-700 shadow-soft transition-transform hover:scale-[1.03] active:scale-100"
            >
              <Icon.Sparkles className="h-5 w-5" />
              Launch the app
            </button>
          </div>
        </Reveal>
      </section>


      {/* Sign in — glassmorphism card, scrolled to from the hero */}
      {supabaseEnabled && (
        <section id="auth-section" className="relative overflow-hidden px-4 pb-24 pt-4">
          <div className="pointer-events-none absolute inset-0 -z-10">
            <div className="animate-float-blob absolute left-[12%] top-[15%] h-[300px] w-[300px] rounded-full bg-brand-500/20 blur-3xl" />
            <div className="animate-float-blob absolute right-[10%] bottom-[5%] h-[280px] w-[280px] rounded-full bg-rose-300/30 blur-3xl" style={{ animationDelay: '-4s' }} />
          </div>
          <div className="mx-auto max-w-4xl">
            <div className="mb-8 text-center">
              <AnimatedTitle className="display-title">Your notes are waiting.</AnimatedTitle>
              <Reveal delay={220}>
                <p className="mt-2 text-espresso-500">
                  Sign in to unlock the app — access is limited to authorized accounts.
                </p>
              </Reveal>
            </div>
            <Reveal delay={120}>
              {user ? (
                <div className="mx-auto w-full max-w-md rounded-[28px] border border-white/60 bg-white/45 p-8 text-center shadow-lift backdrop-blur-2xl">
                  <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-green-100 text-green-600">
                    <Icon.Check className="h-6 w-6" />
                  </div>
                  <p className="font-semibold text-espresso-900">You're signed in</p>
                  <p className="mb-5 mt-0.5 truncate text-sm text-espresso-500">{user.email}</p>
                  <button onClick={onLaunch} className="btn-primary w-full py-2.5">
                    <Icon.Sparkles className="h-5 w-5" />
                    Launch the app
                  </button>
                  <button onClick={onSignOut} className="mt-3 text-xs font-semibold text-espresso-400 hover:text-espresso-700">
                    Sign out
                  </button>
                </div>
              ) : (
                <AuthCard pulse={authPulse} onAuthed={onLaunch} />
              )}
            </Reveal>
          </div>
        </section>
      )}

      {/* Footer */}
      <footer className="border-t border-espresso-900/10 py-10">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-3 px-4 text-sm text-espresso-500 sm:flex-row">
          <span>Agentic AI Notes Generator</span>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="hover:text-brand-600">
            github.com/Kodishalathrilok/agentic-notes
          </a>
        </div>
      </footer>
    </div>
  )
}