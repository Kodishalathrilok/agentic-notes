import { useEffect, useState } from 'react'
import Icon from './Icons'

const METRICS = [
  { key: 'faithfulness', label: 'Faithfulness' },
  { key: 'coverage', label: 'Coverage' },
  { key: 'clarity', label: 'Clarity' },
]

function Bar({ value, color }) {
  const pct = Math.max(0, Math.min(100, (value / 10) * 100))
  return (
    <div className="h-2.5 w-full rounded-full bg-slate-100 dark:bg-slate-800">
      <div className={`h-2.5 rounded-full ${color}`} style={{ width: `${pct}%` }} />
    </div>
  )
}

function MetricGroup({ label, summary, variants }) {
  return (
    <div className="card p-5">
      <h3 className="mb-3 text-sm font-bold">{label}</h3>
      <div className="space-y-3">
        {variants.map((v) => {
          const m = summary[v]?.[label.toLowerCase()]
          if (!m) return null
          return (
            <div key={v}>
              <div className="mb-1 flex justify-between text-xs">
                <span className="capitalize text-slate-500 dark:text-slate-400">{v}</span>
                <span className="font-semibold">
                  {m.mean.toFixed(2)}
                  <span className="ml-1 font-normal text-slate-400">±{m.std.toFixed(2)}</span>
                </span>
              </div>
              <Bar value={m.mean} color={v === 'full' ? 'bg-brand-600' : 'bg-slate-400'} />
            </div>
          )
        })}
      </div>
    </div>
  )
}

function Stat({ label, value, sub }) {
  return (
    <div className="card p-5 text-center">
      <div className="text-3xl font-extrabold text-brand-600 dark:text-brand-400">{value}</div>
      <div className="mt-1 text-sm font-semibold">{label}</div>
      {sub && <div className="mt-0.5 text-xs text-slate-400">{sub}</div>}
    </div>
  )
}

export default function EvalDashboard({ onBack, darkMode, setDarkMode }) {
  const [state, setState] = useState({ loading: true })

  useEffect(() => {
    fetch('/api/eval-report')
      .then((r) => r.json())
      .then((d) => setState({ loading: false, ...d }))
      .catch(() => setState({ loading: false, available: false }))
  }, [])

  const report = state.report
  const summary = report?.summary || {}
  const variants = Object.keys(summary)
  const hasBoth = variants.includes('baseline') && variants.includes('full')
  const lift = hasBoth
    ? summary.full.faithfulness.mean - summary.baseline.faithfulness.mean
    : null
  const hallucReduction = hasBoth
    ? summary.baseline.avg_hallucinations - summary.full.avg_hallucinations
    : null

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <header className="sticky top-0 z-30 border-b border-slate-200/70 bg-white/70 backdrop-blur-xl dark:border-slate-800 dark:bg-slate-900/70">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3">
          <button onClick={onBack} className="flex items-center gap-2.5">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 text-white shadow-lift">
              <Icon.Book className="h-5 w-5" />
            </span>
            <span className="text-base font-extrabold tracking-tight">Agentic Notes</span>
          </button>
          <div className="flex items-center gap-2">
            <button onClick={onBack} className="btn-ghost">
              ← Back to app
            </button>
            <button
              onClick={() => setDarkMode((d) => !d)}
              className="flex h-9 w-9 items-center justify-center rounded-xl border border-slate-200 text-slate-500 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-slate-800"
              aria-label="Toggle dark mode"
            >
              {darkMode ? <Icon.Sun className="h-4 w-4" /> : <Icon.Moon className="h-4 w-4" />}
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-10">
        <div className="mb-8">
          <h1 className="text-3xl font-extrabold tracking-tight">Pipeline evaluation</h1>
          <p className="mt-2 max-w-2xl text-slate-500 dark:text-slate-400">
            An independent LLM-judge scores generated notes for faithfulness, coverage, clarity, and
            quiz-answer accuracy — comparing the basic pipeline against the full self-correcting one.
          </p>
        </div>

        {state.loading && <p className="text-slate-400">Loading report…</p>}

        {!state.loading && !state.available && (
          <div className="card p-8 text-center">
            <h2 className="text-lg font-bold">No eval report yet</h2>
            <p className="mx-auto mt-2 max-w-md text-sm text-slate-500 dark:text-slate-400">
              Generate one by running the eval harness — it scores the pipeline against fixed test
              inputs and writes <code className="font-mono">eval/report.json</code>.
            </p>
            <pre className="mx-auto mt-4 w-fit rounded-lg bg-slate-900 px-4 py-3 text-left text-xs text-slate-100 dark:bg-black">
              <code>cd backend{'\n'}python -m eval.run_eval</code>
            </pre>
          </div>
        )}

        {!state.loading && state.available && report && (
          <div className="space-y-8">
            {/* meta */}
            <div className="flex flex-wrap gap-2 text-xs text-slate-400">
              <span className="chip border border-slate-200 dark:border-slate-700">model: {report.model}</span>
              <span className="chip border border-slate-200 dark:border-slate-700">runs/fixture: {report.runs}</span>
              <span className="chip border border-slate-200 dark:border-slate-700">{report.elapsed_s}s</span>
              {report.timestamp && (
                <span className="chip border border-slate-200 dark:border-slate-700">
                  {new Date(report.timestamp).toLocaleString()}
                </span>
              )}
            </div>

            {/* headline lift */}
            {hasBoth && (
              <div className="grid gap-4 sm:grid-cols-2">
                <Stat
                  label="Faithfulness lift"
                  value={`${lift >= 0 ? '+' : ''}${lift.toFixed(2)}`}
                  sub="full vs baseline (out of 10)"
                />
                <Stat
                  label="Hallucination reduction"
                  value={`${hallucReduction >= 0 ? '−' : '+'}${Math.abs(hallucReduction).toFixed(2)}`}
                  sub="fewer unsupported claims per run"
                />
              </div>
            )}

            {/* metric bars */}
            <div className="grid gap-4 md:grid-cols-3">
              {METRICS.map((m) => (
                <MetricGroup key={m.key} label={m.label} summary={summary} variants={variants} />
              ))}
            </div>

            {/* secondary stats per variant */}
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {variants.map((v) => (
                <div key={v} className="card p-5">
                  <div className="mb-2 text-xs font-bold uppercase tracking-widest text-slate-400 capitalize">
                    {v}
                  </div>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-slate-500 dark:text-slate-400">Quiz accuracy</span>
                      <span className="font-semibold">
                        {summary[v].quiz_accuracy
                          ? `${Math.round(summary[v].quiz_accuracy.mean * 100)}%`
                          : 'n/a'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500 dark:text-slate-400">Avg hallucinations</span>
                      <span className="font-semibold">{summary[v].avg_hallucinations.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500 dark:text-slate-400">Avg revisions</span>
                      <span className="font-semibold">{summary[v].avg_revisions.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <p className="text-center text-xs text-slate-400">
              A CI workflow re-runs this eval and fails the build if faithfulness drops below threshold.
            </p>
          </div>
        )}
      </main>
    </div>
  )
}
