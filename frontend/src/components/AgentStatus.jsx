function StatusIcon({ status }) {
  if (status === 'active') {
    return (
      <svg className="h-4 w-4 animate-spin text-white" viewBox="0 0 24 24" fill="none">
        <circle className="opacity-30" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path className="opacity-90" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.4 0 0 5.4 0 12h4z" />
      </svg>
    )
  }
  if (status === 'done') {
    return (
      <svg className="h-4 w-4 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={3}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
      </svg>
    )
  }
  if (status === 'error') {
    return (
      <svg className="h-4 w-4 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={3}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
      </svg>
    )
  }
  return <span className="block h-2 w-2 rounded-full bg-espresso-900/20" />
}

const NODE = {
  pending: 'bg-white/70 border border-espresso-900/15',
  active: 'bg-gradient-to-br from-brand-500 to-rose-400 shadow-lift ring-4 ring-brand-500/20 scale-110',
  done: 'bg-green-500 shadow-soft',
  error: 'bg-red-500 shadow-soft',
}

const TEXT = {
  pending: 'text-espresso-400',
  active: 'text-espresso-900',
  done: 'text-espresso-700',
  error: 'text-red-600',
}

export default function AgentStatus({ steps, critique }) {
  const doneCount = steps.filter((s) => s.status === 'done').length
  const activeIdx = steps.findIndex((s) => s.status === 'active')
  const progress = Math.min(
    100,
    ((doneCount + (activeIdx >= 0 ? 0.5 : 0)) / steps.length) * 100
  )

  return (
    <div className="glass-card relative overflow-hidden rounded-3xl p-6 transition-shadow duration-300 hover:shadow-lift">
      <div className="mb-1 flex items-center justify-between">
        <h2 className="text-xs font-bold uppercase tracking-[0.2em] text-espresso-500">
          Agent Pipeline
        </h2>
        <span className="text-xs font-semibold tabular-nums text-espresso-400">
          {doneCount}/{steps.length}
        </span>
      </div>

      {/* overall progress bar */}
      <div className="mb-5 h-1 w-full overflow-hidden rounded-full bg-espresso-900/[0.06]">
        <div
          className="h-full rounded-full bg-gradient-to-r from-brand-500 to-rose-400 transition-all duration-700 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* timeline */}
      <ol className="relative space-y-1">
        {/* connecting line */}
        <span className="absolute bottom-4 left-[15px] top-4 w-px bg-espresso-900/10" aria-hidden />

        {steps.map((s) => {
          const isCritique = s.step === 'critique'
          return (
            <li
              key={s.step}
              className={`relative flex items-center gap-3 rounded-2xl px-1.5 py-2 transition-all duration-300 ${
                s.status === 'active' ? 'bg-brand-500/[0.07]' : ''
              }`}
            >
              <span
                className={`relative z-10 flex h-7 w-7 shrink-0 items-center justify-center rounded-full transition-all duration-300 ${NODE[s.status]}`}
              >
                <StatusIcon status={s.status} />
              </span>

              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className={`text-sm font-semibold capitalize transition-colors duration-300 ${TEXT[s.status]}`}>
                    {s.step}
                  </span>
                  {isCritique && critique && typeof critique.score === 'number' && (
                    <span className="animate-slide-up rounded-full bg-gradient-to-r from-brand-500 to-rose-400 px-2 py-0.5 text-[11px] font-bold text-white shadow-soft">
                      {critique.score}/10
                    </span>
                  )}
                </div>
                {s.message && (
                  <p className="truncate text-xs text-espresso-400">{s.message}</p>
                )}
              </div>

              {s.status === 'active' && (
                <span className="flex gap-1" aria-hidden>
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-brand-500 [animation-delay:0ms]" />
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-brand-500 [animation-delay:120ms]" />
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-rose-400 [animation-delay:240ms]" />
                </span>
              )}
            </li>
          )
        })}
      </ol>
    </div>
  )
}