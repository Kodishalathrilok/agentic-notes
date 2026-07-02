import { useState, useEffect, useRef } from 'react'
import Icon from './Icons'
import TextEffect from './TextEffect'

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
      className={`transition-all duration-700 ease-out ${
        shown ? 'translate-y-0 opacity-100' : 'translate-y-6 opacity-0'
      } ${className}`}
    >
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
      className="inline-block animate-slide-up bg-gradient-to-r from-brand-500 to-brand-700 bg-clip-text font-bold text-transparent"
    >
      {words[i]}
    </span>
  )
}

function Nav({ onLaunch, onEval, darkMode, setDarkMode, user, supabaseEnabled, onSignIn, onSignOut }) {
  const link =
    'hidden rounded-xl px-3 py-2 text-sm font-semibold text-slate-600 transition-colors hover:bg-slate-100 hover:text-slate-900 sm:inline-flex dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white'
  return (
    <header className="sticky top-0 z-30 border-b border-slate-200/70 bg-white/70 backdrop-blur-xl dark:border-slate-800 dark:bg-slate-900/70">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <div className="flex items-center gap-2.5">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 text-white shadow-lift">
            <Icon.Book className="h-5 w-5" />
          </span>
          <span className="text-base font-extrabold tracking-tight">Agentic Notes</span>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={onEval} className={link}>
            Eval
          </button>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer" className={link}>
            GitHub
          </a>
          {supabaseEnabled &&
            (user ? (
              <button onClick={onSignOut} className={link.replace('hidden ', '').replace('sm:inline-flex', 'inline-flex')}>
                Sign out
              </button>
            ) : (
              <button onClick={onSignIn} className={link.replace('hidden ', '').replace('sm:inline-flex', 'inline-flex')}>
                Sign in
              </button>
            ))}
          <button
            onClick={() => setDarkMode((d) => !d)}
            className="flex h-9 w-9 items-center justify-center rounded-xl border border-slate-200 text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-900 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-white"
            aria-label="Toggle dark mode"
          >
            {darkMode ? <Icon.Sun className="h-4 w-4" /> : <Icon.Moon className="h-4 w-4" />}
          </button>
          <button onClick={onLaunch} className="btn-primary">
            Launch App
            <span aria-hidden>→</span>
          </button>
        </div>
      </div>
    </header>
  )
}

export default function Landing({ onLaunch, onEval, darkMode, setDarkMode, user, supabaseEnabled, onSignIn, onSignOut }) {
  return (
    <div className="min-h-screen overflow-x-hidden bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <Nav
        onLaunch={onLaunch}
        onEval={onEval}
        darkMode={darkMode}
        setDarkMode={setDarkMode}
        user={user}
        supabaseEnabled={supabaseEnabled}
        onSignIn={onSignIn}
        onSignOut={onSignOut}
      />

      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="pointer-events-none absolute inset-0 -z-10">
          <div className="animate-float-blob absolute left-[15%] top-[-8%] h-[360px] w-[360px] rounded-full bg-brand-400/25 blur-3xl dark:bg-brand-600/20" />
          <div
            className="animate-float-blob absolute right-[10%] top-[10%] h-[320px] w-[320px] rounded-full bg-fuchsia-400/20 blur-3xl dark:bg-fuchsia-600/15"
            style={{ animationDelay: '-6s' }}
          />
          <div
            className="absolute inset-0 opacity-[0.04] dark:opacity-[0.07]"
            style={{
              backgroundImage:
                'linear-gradient(to right, currentColor 1px, transparent 1px), linear-gradient(to bottom, currentColor 1px, transparent 1px)',
              backgroundSize: '44px 44px',
            }}
          />
        </div>

        <div className="mx-auto max-w-4xl px-4 py-24 text-center sm:py-32">
          <div
            className="mb-6 inline-flex animate-slide-up items-center gap-2 rounded-full border border-brand-200 bg-brand-50 px-3 py-1 text-xs font-semibold text-brand-700 dark:border-brand-800/60 dark:bg-brand-900/30 dark:text-brand-300"
            style={{ animationDelay: '0ms' }}
          >
            <Icon.Sparkles className="h-3.5 w-3.5" />
            Multi-agent · self-correcting · faithfulness-checked
          </div>

          <h1 className="text-4xl font-extrabold leading-[1.05] tracking-tight sm:text-6xl">
            <TextEffect per="word" preset="slide" as="span" className="block" stagger={0.08} startDelay={0.05}>
              Study notes that
            </TextEffect>
            <TextEffect
              per="word"
              preset="slide"
              as="span"
              className="block"
              unitClassName="bg-gradient-to-r from-brand-500 via-fuchsia-500 to-brand-700 bg-clip-text text-transparent"
              stagger={0.08}
              startDelay={0.29}
            >
              fact-check themselves
            </TextEffect>
          </h1>

          <p
            className="mx-auto mt-6 max-w-2xl animate-slide-up text-lg text-slate-600 dark:text-slate-300"
            style={{ animationDelay: '300ms' }}
          >
            A pipeline of AI agents plans, writes, critiques, and revises your notes — checking every
            claim against the source — then builds a quiz and spaced-repetition flashcards.
          </p>

          <p
            className="mt-3 animate-slide-up text-base text-slate-500 dark:text-slate-400"
            style={{ animationDelay: '380ms' }}
          >
            Turn <RotatingWord words={SOURCES} /> into exam-ready notes.
          </p>

          <div
            className="mt-9 flex animate-slide-up flex-wrap items-center justify-center gap-3"
            style={{ animationDelay: '460ms' }}
          >
            <button onClick={onLaunch} className="btn-primary px-6 py-3 text-base">
              <Icon.Sparkles className="h-5 w-5" />
              Launch the app
            </button>
            <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="btn-ghost px-6 py-3 text-base">
              View source on GitHub
            </a>
          </div>
          <p
            className="mt-4 animate-slide-up text-xs text-slate-400"
            style={{ animationDelay: '540ms' }}
          >
            Powered by Llama on Groq, Gemini & local Ollama — with automatic failover
          </p>
        </div>
      </section>

      {/* How it works */}
      <section className="mx-auto max-w-6xl px-4 py-16">
        <Reveal className="mb-10 text-center">
          <h2 className="text-3xl font-extrabold tracking-tight">How it works</h2>
          <p className="mt-2 text-slate-500 dark:text-slate-400">
            Six specialized agents, with a self-correcting critique loop at the core.
          </p>
        </Reveal>

        <div className="flex flex-wrap items-stretch justify-center gap-3">
          {PIPELINE.map((step, i) => (
            <Reveal key={step.name} delay={i * 90} className="flex items-center gap-3">
              <div
                className={`w-40 rounded-2xl border p-4 text-center transition-shadow hover:shadow-lift ${
                  step.name === 'Critique' || step.name === 'Revise'
                    ? 'border-brand-300 bg-brand-50/60 dark:border-brand-700/60 dark:bg-brand-900/20'
                    : 'border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900/50'
                }`}
              >
                <div className="mx-auto mb-2 flex h-8 w-8 items-center justify-center rounded-full bg-brand-600 text-sm font-bold text-white">
                  {i + 1}
                </div>
                <div className="text-sm font-bold">{step.name}</div>
                <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">{step.desc}</div>
              </div>
              {i < PIPELINE.length - 1 && (
                <span className="hidden text-slate-300 dark:text-slate-600 lg:inline" aria-hidden>
                  →
                </span>
              )}
            </Reveal>
          ))}
        </div>

        <Reveal delay={200} className="mx-auto mt-6 max-w-md rounded-xl border border-brand-200 bg-brand-50/60 px-4 py-3 text-center text-sm text-brand-700 dark:border-brand-800/60 dark:bg-brand-900/20 dark:text-brand-300">
          <Icon.Refresh className="mr-1.5 inline h-4 w-4" />
          Critique → Revise repeats until the notes are faithful and complete.
        </Reveal>
      </section>

      {/* Features */}
      <section className="bg-white py-16 dark:bg-slate-900/40">
        <div className="mx-auto max-w-6xl px-4">
          <Reveal className="mb-10 text-center">
            <h2 className="text-3xl font-extrabold tracking-tight">Everything you need to study</h2>
          </Reveal>
          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {FEATURES.map((f, i) => (
              <Reveal key={f.title} delay={(i % 3) * 90}>
                <div className="card h-full p-6 transition-shadow duration-200 hover:shadow-lift">
                  <span className="mb-4 flex h-11 w-11 items-center justify-center rounded-xl bg-brand-50 text-brand-600 dark:bg-brand-900/30 dark:text-brand-300">
                    <f.icon className="h-5 w-5" />
                  </span>
                  <h3 className="mb-1.5 text-base font-bold">{f.title}</h3>
                  <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-400">{f.body}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* Engineered for quality */}
      <section className="mx-auto max-w-6xl px-4 py-16">
        <Reveal className="mb-10 text-center">
          <h2 className="text-3xl font-extrabold tracking-tight">Engineered for quality</h2>
          <p className="mt-2 text-slate-500 dark:text-slate-400">
            The parts that separate a real system from an API wrapper.
          </p>
        </Reveal>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {QUALITY.map(([title, body], i) => (
            <Reveal key={title} delay={(i % 3) * 90}>
              <div className="h-full rounded-2xl border border-slate-200 p-5 dark:border-slate-800">
                <div className="mb-1.5 flex items-center gap-2">
                  <Icon.Check className="h-4 w-4 text-green-500" />
                  <h3 className="text-sm font-bold">{title}</h3>
                </div>
                <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-400">{body}</p>
              </div>
            </Reveal>
          ))}
        </div>

        <Reveal className="mt-12 flex flex-wrap items-center justify-center gap-2">
          {STACK.map((t) => (
            <span
              key={t}
              className="chip border border-slate-200 bg-white text-slate-600 dark:border-slate-700 dark:bg-slate-900/50 dark:text-slate-300"
            >
              {t}
            </span>
          ))}
        </Reveal>
      </section>

      {/* CTA */}
      <section className="px-4 pb-20">
        <Reveal className="mx-auto max-w-4xl">
          <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-brand-600 to-brand-800 px-6 py-14 text-center text-white shadow-lift">
            <div className="animate-float-blob pointer-events-none absolute -right-10 -top-10 h-48 w-48 rounded-full bg-white/10 blur-2xl" />
            <h2 className="text-3xl font-extrabold tracking-tight">Ready to study smarter?</h2>
            <p className="mx-auto mt-2 max-w-xl text-brand-100">
              Turn any source into structured notes, a quiz, and flashcards in under a minute.
            </p>
            <button
              onClick={onLaunch}
              className="mt-7 inline-flex items-center gap-2 rounded-xl bg-white px-6 py-3 text-base font-bold text-brand-700 shadow-soft transition-transform hover:scale-[1.03] active:scale-100"
            >
              <Icon.Sparkles className="h-5 w-5" />
              Launch the app
            </button>
          </div>
        </Reveal>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-200 py-8 dark:border-slate-800">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-3 px-4 text-sm text-slate-500 sm:flex-row">
          <span>Agentic AI Notes Generator</span>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="hover:text-brand-600 dark:hover:text-brand-400">
            github.com/Kodishalathrilok/agentic-notes
          </a>
        </div>
      </footer>
    </div>
  )
}
