function StatusIcon({ status }) {
  if (status === 'active') {
    return (
      <svg className="h-5 w-5 animate-spin text-brand-500" viewBox="0 0 24 24" fill="none">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.4 0 0 5.4 0 12h4z" />
      </svg>
    )
  }
  if (status === 'done') {
    return (
      <svg className="h-5 w-5 text-green-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
      </svg>
    )
  }
  if (status === 'error') {
    return (
      <svg className="h-5 w-5 text-red-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
      </svg>
    )
  }
  // pending — clock
  return (
    <svg className="h-5 w-5 text-slate-300 dark:text-slate-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
      <circle cx="12" cy="12" r="9" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 7v5l3 2" />
    </svg>
  )
}

const COLORS = {
  pending: 'text-slate-400 dark:text-slate-500',
  active: 'text-brand-600 dark:text-brand-400',
  done: 'text-green-600 dark:text-green-400',
  error: 'text-red-600 dark:text-red-400',
}

export default function AgentStatus({ steps, critique }) {
  return (
    <div className="card p-5">
      <h2 className="mb-4 text-xs font-bold uppercase tracking-widest text-slate-400 dark:text-slate-500">
        Agent Pipeline
      </h2>
      <ol className="space-y-2">
        {steps.map((s) => {
          const isCritique = s.step === 'critique'
          return (
            <li
              key={s.step}
              className={`flex items-start gap-3 rounded-xl px-3 py-2 transition-all duration-300 ${
                s.status === 'active'
                  ? 'bg-brand-50 dark:bg-brand-900/20'
                  : 'hover:bg-slate-50 dark:hover:bg-slate-800/40'
              }`}
            >
              <div className="mt-0.5 shrink-0">
                <StatusIcon status={s.status} />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className={`text-sm font-semibold capitalize ${COLORS[s.status]}`}>
                    {s.step}
                  </span>
                  {isCritique && critique && typeof critique.score === 'number' && (
                    <span className="rounded-full bg-brand-100 px-2 py-0.5 text-xs font-semibold text-brand-700 dark:bg-brand-900/40 dark:text-brand-300">
                      {critique.score}/10
                    </span>
                  )}
                </div>
                {s.message && (
                  <p className="truncate text-xs text-slate-500 dark:text-slate-400">{s.message}</p>
                )}
              </div>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
