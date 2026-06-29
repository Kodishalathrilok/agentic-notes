import Icon from './Icons'

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
  {
    icon: Icon.Refresh,
    title: 'Self-correcting agents',
    body: 'A Plan → Write → Critique → Revise loop where agents review and rewrite each other until the notes pass a quality bar.',
  },
  {
    icon: Icon.Check,
    title: 'Faithfulness-checked',
    body: 'A grounded critique compares every claim against the source and removes anything fabricated — fighting hallucination by design.',
  },
  {
    icon: Icon.Zap,
    title: 'Real-time streaming',
    body: 'Notes type out token-by-token over Server-Sent Events while a live pipeline view shows each agent working.',
  },
  {
    icon: Icon.Book,
    title: 'Multimodal input',
    body: 'Paste text, upload a PDF, drop a web link or YouTube URL (transcript), or record audio — all become study notes.',
  },
  {
    icon: Icon.Star,
    title: 'Spaced repetition',
    body: 'Flashcards with Again / Good / Easy scheduling, a due-cards filter, and Anki-compatible CSV export.',
  },
  {
    icon: Icon.Sparkles,
    title: 'Ask your notes',
    body: 'A built-in tutor chat answers follow-up questions grounded in the notes you just generated.',
  },
]

const QUALITY = [
  ['Grounded critique', 'Judges notes against the original source for faithfulness & coverage.'],
  ['Iterative revision', 'Re-critiques after each revision and keeps the best-scoring version.'],
  ['Automated eval suite', 'An LLM-judge harness measures faithfulness, coverage & quiz accuracy.'],
  ['Provider fallback', 'Groq (cloud) with automatic Ollama (local) fallback and rate-limit backoff.'],
  ['Structured outputs', 'JSON-mode agents with conservative fallbacks for reliable parsing.'],
  ['Answer verification', 'Generated quiz keys are re-checked against the notes before display.'],
]

const STACK = [
  'FastAPI',
  'SSE Streaming',
  'React',
  'Vite',
  'Tailwind CSS',
  'Groq · Llama 3.3',
  'Whisper',
  'reportlab',
  'Docker',
]

function Nav({ onLaunch, onEval, darkMode, setDarkMode }) {
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
          <button
            onClick={onEval}
            className="hidden rounded-xl px-3 py-2 text-sm font-semibold text-slate-600 transition-colors hover:bg-slate-100 hover:text-slate-900 sm:inline-flex dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white"
          >
            Eval
          </button>
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer"
            className="hidden rounded-xl px-3 py-2 text-sm font-semibold text-slate-600 transition-colors hover:bg-slate-100 hover:text-slate-900 sm:inline-flex dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white"
          >
            GitHub
          </a>
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

export default function Landing({ onLaunch, onEval, darkMode, setDarkMode }) {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <Nav onLaunch={onLaunch} onEval={onEval} darkMode={darkMode} setDarkMode={setDarkMode} />

      {/* Hero */}
      <section className="relative overflow-hidden">
        {/* soft gradient + grid backdrop */}
        <div className="pointer-events-none absolute inset-0 -z-10">
          <div className="absolute left-1/2 top-[-10%] h-[420px] w-[820px] -translate-x-1/2 rounded-full bg-brand-400/20 blur-3xl dark:bg-brand-600/20" />
          <div
            className="absolute inset-0 opacity-[0.04] dark:opacity-[0.07]"
            style={{
              backgroundImage:
                'linear-gradient(to right, currentColor 1px, transparent 1px), linear-gradient(to bottom, currentColor 1px, transparent 1px)',
              backgroundSize: '40px 40px',
            }}
          />
        </div>

        <div className="mx-auto max-w-4xl px-4 py-20 text-center sm:py-28">
          <div className="mb-5 inline-flex animate-fade-in items-center gap-2 rounded-full border border-brand-200 bg-brand-50 px-3 py-1 text-xs font-semibold text-brand-700 dark:border-brand-800/60 dark:bg-brand-900/30 dark:text-brand-300">
            <Icon.Sparkles className="h-3.5 w-3.5" />
            Multi-agent · self-correcting · faithfulness-checked
          </div>
          <h1 className="animate-slide-up text-4xl font-extrabold leading-tight tracking-tight sm:text-6xl">
            Study notes that
            <span className="bg-gradient-to-r from-brand-500 to-brand-700 bg-clip-text text-transparent">
              {' '}fact-check themselves
            </span>
          </h1>
          <p className="mx-auto mt-5 max-w-2xl animate-slide-up text-lg text-slate-600 dark:text-slate-300">
            A pipeline of AI agents plans, writes, critiques, and revises your notes — checking every
            claim against the source — then builds a quiz and spaced-repetition flashcards. From text,
            PDFs, web links, YouTube, or audio.
          </p>
          <div className="mt-8 flex animate-slide-up flex-wrap items-center justify-center gap-3">
            <button onClick={onLaunch} className="btn-primary px-6 py-3 text-base">
              <Icon.Sparkles className="h-5 w-5" />
              Launch the app
            </button>
            <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="btn-ghost px-6 py-3 text-base">
              View source on GitHub
            </a>
          </div>
          <p className="mt-4 text-xs text-slate-400">Powered by Llama 3.3 on Groq · runs locally with Ollama too</p>
        </div>
      </section>

      {/* How it works — pipeline */}
      <section className="mx-auto max-w-6xl px-4 py-16">
        <div className="mb-10 text-center">
          <h2 className="text-3xl font-extrabold tracking-tight">How it works</h2>
          <p className="mt-2 text-slate-500 dark:text-slate-400">
            Six specialized agents, with a self-correcting critique loop at the core.
          </p>
        </div>

        <div className="flex flex-wrap items-stretch justify-center gap-3">
          {PIPELINE.map((step, i) => (
            <div key={step.name} className="flex items-center gap-3">
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
            </div>
          ))}
        </div>

        <div className="mx-auto mt-6 max-w-md rounded-xl border border-brand-200 bg-brand-50/60 px-4 py-3 text-center text-sm text-brand-700 dark:border-brand-800/60 dark:bg-brand-900/20 dark:text-brand-300">
          <Icon.Refresh className="mr-1.5 inline h-4 w-4" />
          Critique → Revise repeats until the notes are faithful and complete.
        </div>
      </section>

      {/* Features */}
      <section className="bg-white py-16 dark:bg-slate-900/40">
        <div className="mx-auto max-w-6xl px-4">
          <div className="mb-10 text-center">
            <h2 className="text-3xl font-extrabold tracking-tight">Everything you need to study</h2>
          </div>
          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {FEATURES.map((f) => (
              <div key={f.title} className="card p-6 transition-shadow duration-200 hover:shadow-lift">
                <span className="mb-4 flex h-11 w-11 items-center justify-center rounded-xl bg-brand-50 text-brand-600 dark:bg-brand-900/30 dark:text-brand-300">
                  <f.icon className="h-5 w-5" />
                </span>
                <h3 className="mb-1.5 text-base font-bold">{f.title}</h3>
                <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-400">{f.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Engineered for quality */}
      <section className="mx-auto max-w-6xl px-4 py-16">
        <div className="mb-10 text-center">
          <h2 className="text-3xl font-extrabold tracking-tight">Engineered for quality</h2>
          <p className="mt-2 text-slate-500 dark:text-slate-400">
            The parts that separate a real system from an API wrapper.
          </p>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {QUALITY.map(([title, body]) => (
            <div key={title} className="rounded-2xl border border-slate-200 p-5 dark:border-slate-800">
              <div className="mb-1.5 flex items-center gap-2">
                <Icon.Check className="h-4 w-4 text-green-500" />
                <h3 className="text-sm font-bold">{title}</h3>
              </div>
              <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-400">{body}</p>
            </div>
          ))}
        </div>

        {/* Tech stack */}
        <div className="mt-12 flex flex-wrap items-center justify-center gap-2">
          {STACK.map((t) => (
            <span
              key={t}
              className="chip border border-slate-200 bg-white text-slate-600 dark:border-slate-700 dark:bg-slate-900/50 dark:text-slate-300"
            >
              {t}
            </span>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="px-4 pb-20">
        <div className="mx-auto max-w-4xl overflow-hidden rounded-3xl bg-gradient-to-br from-brand-600 to-brand-800 px-6 py-14 text-center text-white shadow-lift">
          <h2 className="text-3xl font-extrabold tracking-tight">Ready to study smarter?</h2>
          <p className="mx-auto mt-2 max-w-xl text-brand-100">
            Turn any source into structured notes, a quiz, and flashcards in under a minute.
          </p>
          <button
            onClick={onLaunch}
            className="mt-7 inline-flex items-center gap-2 rounded-xl bg-white px-6 py-3 text-base font-bold text-brand-700 shadow-soft transition-transform hover:scale-[1.02] active:scale-100"
          >
            <Icon.Sparkles className="h-5 w-5" />
            Launch the app
          </button>
        </div>
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
