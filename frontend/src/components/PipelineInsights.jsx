import { useState } from 'react'

function Chips({ items, color }) {
  if (!items || items.length === 0) return null
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((it, i) => (
        <span
          key={i}
 className={`rounded-full px-2 py-0.5 text-xs ${color}`}
        >
          {it}
        </span>
      ))}
    </div>
  )
}

export default function PipelineInsights({ plan, critique }) {
  const [open, setOpen] = useState(false)

  if (!plan && !critique) return null

  return (
    <div className="card overflow-hidden">
      <button
        onClick={() => setOpen((o) => !o)}
 className="flex w-full cursor-pointer items-center justify-between px-5 py-3.5 text-left transition-colors hover:bg-slate-50 dark:hover:bg-slate-800/40"
      >
        <span className="text-xs font-bold uppercase tracking-[0.2em] text-espresso-500">
          Pipeline Insights
        </span>
        <span className={`text-slate-400 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}>
          <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="m6 9 6 6 6-6" />
          </svg>
        </span>
      </button>

      {open && (
        <div className="space-y-4 border-t border-slate-100 px-4 py-3 dark:border-slate-700">
          {plan && (
            <div>
              <div className="mb-1 flex items-center gap-2 text-xs font-semibold text-brand-600 dark:text-brand-600">
                PLAN
                {plan.difficulty && (
                  <span className="rounded-full bg-slate-100 px-2 py-0.5 capitalize text-slate-500 dark:bg-slate-700 dark:text-slate-300">
                    {plan.difficulty}
                  </span>
                )}
              </div>
              {plan.outline?.length > 0 && (
                <ul className="mb-2 ml-4 list-disc text-sm text-slate-600 dark:text-slate-300">
                  {plan.outline.map((o, i) => (
                    <li key={i}>{o}</li>
                  ))}
                </ul>
              )}
              <Chips items={plan.checklist} color="bg-brand-50 text-brand-700 dark:bg-brand-900/30 dark:text-brand-700" />
            </div>
          )}

          {critique && (
            <div>
              <div className="mb-1 flex items-center gap-2 text-xs font-semibold text-slate-500 dark:text-slate-400">
                CRITIQUE
                {typeof critique.score === 'number' && (
                  <span className="rounded-full bg-brand-100 px-2 py-0.5 font-bold text-brand-700 dark:bg-brand-900/40 dark:text-brand-700">
                    {critique.score}/10
                  </span>
                )}
              </div>
              {critique.strengths?.length > 0 && (
                <div className="mb-2">
                  <p className="mb-1 text-xs text-green-600 dark:text-green-400">Strengths</p>
                  <Chips items={critique.strengths} color="bg-green-50 text-green-700 dark:bg-green-900/30 dark:text-green-300" />
                </div>
              )}
              {critique.unsupported_claims?.length > 0 && (
                <div className="mb-2">
                  <p className="mb-1 text-xs text-red-600 dark:text-red-400">
                    ⚠ Unsupported claims caught (removed in revision)
                  </p>
                  <Chips
                    items={critique.unsupported_claims}
                    color="bg-red-50 text-red-700 dark:bg-red-900/30 dark:text-red-300"
                  />
                </div>
              )}
              {critique.issues?.length > 0 && (
                <div className="mb-2">
                  <p className="mb-1 text-xs text-amber-600 dark:text-amber-400">Issues fixed</p>
                  <Chips items={critique.issues} color="bg-amber-50 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300" />
                </div>
              )}
              {critique.missing_topics?.length > 0 && (
                <div>
                  <p className="mb-1 text-xs text-red-500 dark:text-red-400">Was missing</p>
                  <Chips items={critique.missing_topics} color="bg-red-50 text-red-700 dark:bg-red-900/30 dark:text-red-300" />
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
