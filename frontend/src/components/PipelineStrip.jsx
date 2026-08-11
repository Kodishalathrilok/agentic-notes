import Icon from './Icons'

// Horizontal AI pipeline indicator: o -> o -> o -> o -> o -> o
// Idle on the home screen; lights up step by step while generating.
// status per step: 'pending' | 'active' | 'done' (matches App's agentSteps)

const LABELS = {
  plan: 'Plan',
  write: 'Write',
  critique: 'Critique',
  revise: 'Revise',
  quiz: 'Quiz',
  flashcards: 'Cards',
}

export default function PipelineStrip({ steps = [], compact = false }) {
  const items = steps.length
    ? steps
    : Object.keys(LABELS).map((step) => ({ step, status: 'pending' }))

  return (
    <div
      className={`glass-pill mx-auto flex w-fit max-w-full items-center overflow-x-auto rounded-full ${
        compact ? 'gap-1 px-3 py-1.5' : 'gap-1.5 px-5 py-2.5'
      }`}
      title="The agent pipeline: plan → write → critique → revise → quiz → flashcards"
    >
      {items.map((s, i) => {
        const done = s.status === 'done'
        const active = s.status === 'active' || s.status === 'running'
        return (
          <div key={s.step} className="flex shrink-0 items-center">
            <div className="flex flex-col items-center gap-0.5">
              <span
                className={`flex items-center justify-center rounded-full border-2 transition-all duration-300 ${
                  compact ? 'h-3.5 w-3.5' : 'h-5 w-5'
                } ${
                  done
                    ? 'border-espresso-900 bg-espresso-900'
                    : active
                      ? 'animate-pulse border-espresso-900 bg-white/80'
                      : 'border-espresso-900/25 bg-white/50'
                }`}
              >
                {done && <Icon.Check className={compact ? 'h-2 w-2 text-white' : 'h-3 w-3 text-white'} />}
              </span>
              {!compact && (
                <span
                  className={`text-[10px] font-semibold leading-none ${
                    done || active ? 'text-espresso-900' : 'text-espresso-400'
                  }`}
                >
                  {LABELS[s.step] || s.step}
                </span>
              )}
            </div>
            {i < items.length - 1 && (
              <span
                className={`mx-1 mb-0.5 self-start ${compact ? 'mt-1' : 'mt-1.5'} text-espresso-300`}
                aria-hidden="true"
              >
                <svg width={compact ? 12 : 16} height="8" viewBox="0 0 16 8" fill="none">
                  <path d="M0 4h13M10 1l4 3-4 3" stroke="currentColor" strokeWidth="1.5" />
                </svg>
              </span>
            )}
          </div>
        )
      })}
    </div>
  )
}