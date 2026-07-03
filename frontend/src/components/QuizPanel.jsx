import { useMemo, useState } from 'react'
import Icon from './Icons'

// Parse the plain-text quiz format into structured question objects.
function parseQuiz(raw) {
  if (!raw) return []
  const questions = []
  let current = null

  const push = () => {
    if (current && current.question) questions.push(current)
  }

  raw.split('\n').forEach((lineRaw) => {
    const line = lineRaw.trim()
    if (!line) return

    const qMatch = line.match(/^Q?\s*\d+[).:]\s*(.+)$/i)
    const optMatch = line.match(/^([A-D])[).:]\s*(.+)$/)
    const ansMatch = line.match(/^Answer\s*:?\s*([A-D])/i)
    const expMatch = line.match(/^Explanation\s*:?\s*(.+)$/i)

    if (qMatch && !optMatch) {
      push()
      current = { question: qMatch[1].trim(), options: {}, answer: '', explanation: '' }
    } else if (optMatch && current) {
      current.options[optMatch[1].toUpperCase()] = optMatch[2].trim()
    } else if (ansMatch && current) {
      current.answer = ansMatch[1].toUpperCase()
    } else if (expMatch && current) {
      current.explanation = expMatch[1].trim()
    } else if (current && !current.answer && Object.keys(current.options).length === 0) {
      // continuation of question text
      current.question += ' ' + line
    }
  })
  push()
  return questions
}

export default function QuizPanel({ quiz, onRegenerate, regenerating }) {
  const questions = useMemo(() => parseQuiz(quiz), [quiz])
  const [picked, setPicked] = useState({}) // { index: 'A' }
  const [resetKey, setResetKey] = useState(0)

  if (!quiz || questions.length === 0) {
    return (
      <div className="flex h-48 flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-slate-300 text-sm text-slate-400 dark:border-slate-600">
        No quiz yet. Generate notes to create a quiz.
        {onRegenerate && (
          <button
            onClick={onRegenerate}
            disabled={regenerating}
 className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-100 disabled:opacity-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-700/70"
          >
            <Icon.Refresh className="h-3.5 w-3.5" />
            {regenerating ? 'Generating…' : 'Generate quiz from notes'}
          </button>
        )}
      </div>
    )
  }

  const answeredCount = Object.keys(picked).length
  const score = questions.reduce(
    (acc, q, i) => acc + (picked[i] && picked[i] === q.answer ? 1 : 0),
    0
  )
  const allAnswered = answeredCount === questions.length

  const choose = (qi, opt) => {
    if (picked[qi]) return // lock after first answer
    setPicked((p) => ({ ...p, [qi]: opt }))
  }

  const reset = () => {
    setPicked({})
    setResetKey((k) => k + 1)
  }

  return (
    <div key={resetKey} className="space-y-4">
      <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-4 py-3 dark:border-slate-700 dark:bg-slate-800">
        <div className="text-sm font-medium text-slate-600 dark:text-slate-300">
          {allAnswered ? (
            <span className="text-brand-600 dark:text-brand-600">
              Final score: {score} / {questions.length}
            </span>
          ) : (
            <span>
              Answered {answeredCount} / {questions.length} · Score {score}
            </span>
          )}
        </div>
        <div className="flex gap-2">
          {onRegenerate && (
            <button
              onClick={onRegenerate}
              disabled={regenerating}
 className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-100 disabled:opacity-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-700/70"
            >
              <Icon.Refresh className="h-3.5 w-3.5" />
              {regenerating ? 'Generating…' : 'New questions'}
            </button>
          )}
          <button
            onClick={reset}
 className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-700"
          >
            Reset Quiz
          </button>
        </div>
      </div>

      {questions.map((q, qi) => {
        const chosen = picked[qi]
        return (
          <div
            key={qi}
 className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-800"
          >
            <p className="mb-3 text-sm font-semibold text-slate-800 dark:text-slate-100">
              {qi + 1}. {q.question}
            </p>
            <div className="space-y-2">
              {['A', 'B', 'C', 'D'].map((opt) => {
                if (!q.options[opt]) return null
                const isChosen = chosen === opt
                const isCorrect = opt === q.answer
                let cls =
                  'border-slate-300 text-slate-700 hover:bg-slate-50 dark:border-slate-600 dark:text-slate-200 dark:hover:bg-slate-700'
                if (chosen) {
                  if (isCorrect)
                    cls = 'border-green-500 bg-green-50 text-green-700 dark:bg-green-900/20 dark:text-green-300'
                  else if (isChosen)
                    cls = 'border-red-500 bg-red-50 text-red-700 dark:bg-red-900/20 dark:text-red-300'
                  else cls = 'border-slate-200 text-slate-400 dark:border-slate-700'
                }
                return (
                  <button
                    key={opt}
                    onClick={() => choose(qi, opt)}
                    disabled={!!chosen}
 className={`flex w-full items-start gap-2 rounded-lg border px-3 py-2 text-left text-sm transition ${cls}`}
                  >
                    <span className="font-semibold">{opt})</span>
                    <span>{q.options[opt]}</span>
                  </button>
                )
              })}
            </div>

            {chosen && (
              <div className="mt-3 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-600 dark:bg-espresso-900/40 dark:text-slate-300">
                {chosen === q.answer ? (
                  <span className="font-medium text-green-600 dark:text-green-400">Correct! </span>
                ) : (
                  <span className="font-medium text-red-600 dark:text-red-400">
                    Correct answer: {q.answer}.{' '}
                  </span>
                )}
                {q.explanation}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
