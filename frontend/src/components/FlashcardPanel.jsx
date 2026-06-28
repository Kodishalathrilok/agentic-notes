import { useMemo, useState, useEffect } from 'react'
import Icon from './Icons'

// Parse "CARD n / Front: / Back:" format into { front, back } objects.
function parseFlashcards(raw) {
  if (!raw) return []
  const cards = []
  let current = null
  let mode = null // 'front' | 'back'

  const push = () => {
    if (current && (current.front || current.back)) cards.push(current)
  }

  raw.split('\n').forEach((lineRaw) => {
    const line = lineRaw.trim()
    if (!line) return

    if (/^CARD\s*\d+/i.test(line)) {
      push()
      current = { front: '', back: '' }
      mode = null
      return
    }
    const frontMatch = line.match(/^Front\s*:?\s*(.*)$/i)
    const backMatch = line.match(/^Back\s*:?\s*(.*)$/i)

    if (frontMatch) {
      if (!current) current = { front: '', back: '' }
      current.front = frontMatch[1].trim()
      mode = 'front'
    } else if (backMatch) {
      if (!current) current = { front: '', back: '' }
      current.back = backMatch[1].trim()
      mode = 'back'
    } else if (current && mode) {
      current[mode] += (current[mode] ? ' ' : '') + line
    }
  })
  push()
  return cards
}

function shuffleArray(arr) {
  const a = [...arr]
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

// ---- Spaced repetition (lightweight, localStorage-backed) -----------------
const SRS_KEY = 'agentic-notes-srs'

function loadSRS() {
  try {
    return JSON.parse(localStorage.getItem(SRS_KEY) || '{}')
  } catch {
    return {}
  }
}
function saveSRS(data) {
  localStorage.setItem(SRS_KEY, JSON.stringify(data))
}
function cardKey(card) {
  return (card.front || '').slice(0, 80)
}
function isDue(card, srs) {
  const s = srs[cardKey(card)]
  if (!s) return true // new card
  return new Date(s.due).getTime() <= Date.now()
}
// Returns the new schedule for a rating.
function schedule(prev, level) {
  const now = Date.now()
  const day = 24 * 60 * 60 * 1000
  let interval = prev?.interval || 0
  if (level === 'again') interval = 0 // ~minutes, treat as due again now
  else if (level === 'good') interval = interval ? interval * 2 : 1
  else if (level === 'easy') interval = interval ? interval * 3 : 3
  const dueMs = now + (level === 'again' ? 60 * 1000 : interval * day)
  return { interval, due: new Date(dueMs).toISOString(), level }
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

export default function FlashcardPanel({ flashcards, onRegenerate, regenerating }) {
  const parsed = useMemo(() => parseFlashcards(flashcards), [flashcards])
  const [order, setOrder] = useState(parsed)
  const [index, setIndex] = useState(0)
  const [flipped, setFlipped] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [srs, setSrs] = useState(loadSRS)
  const [dueOnly, setDueOnly] = useState(false)

  useEffect(() => {
    setOrder(parsed)
    setIndex(0)
    setFlipped(false)
    setDueOnly(false)
  }, [parsed])

  const dueCount = parsed.filter((c) => isDue(c, srs)).length

  const go = (delta) => {
    setFlipped(false)
    setIndex((i) => (i + delta + order.length) % order.length)
  }

  // Keyboard shortcuts: ← / → navigate, Space flips.
  useEffect(() => {
    const onKey = (e) => {
      const tag = (e.target.tagName || '').toLowerCase()
      if (tag === 'input' || tag === 'textarea') return
      if (e.key === 'ArrowRight') go(1)
      else if (e.key === 'ArrowLeft') go(-1)
      else if (e.key === ' ') {
        e.preventDefault()
        setFlipped((f) => !f)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [order.length])

  const rate = (level) => {
    const card = order[index]
    if (!card) return
    const next = { ...srs, [cardKey(card)]: schedule(srs[cardKey(card)], level) }
    setSrs(next)
    saveSRS(next)
    go(1)
  }

  const toggleDueOnly = () => {
    if (!dueOnly) {
      const due = parsed.filter((c) => isDue(c, srs))
      setOrder(due.length ? due : parsed)
      setDueOnly(true)
    } else {
      setOrder(parsed)
      setDueOnly(false)
    }
    setIndex(0)
    setFlipped(false)
  }

  const exportCsv = async () => {
    setExporting(true)
    try {
      const res = await fetch('/api/export/flashcards-csv', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ flashcards }),
      })
      if (!res.ok) throw new Error('Export failed')
      const blob = await res.blob()
      triggerDownload(blob, 'flashcards.csv')
    } catch {
      /* ignore */
    } finally {
      setExporting(false)
    }
  }

  if (!flashcards || order.length === 0) {
    return (
      <div className="flex h-48 flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-slate-300 text-sm text-slate-400 dark:border-slate-600">
        No flashcards yet. Generate notes to create flashcards.
        {onRegenerate && (
          <button
            onClick={onRegenerate}
            disabled={regenerating}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-100 disabled:opacity-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-700/70"
          >
            <Icon.Refresh className="h-3.5 w-3.5" />
            {regenerating ? 'Generating…' : 'Generate flashcards from notes'}
          </button>
        )}
      </div>
    )
  }

  const card = order[index]

  const fcBtn =
    'inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 transition-colors cursor-pointer hover:bg-slate-100 hover:text-slate-900 disabled:opacity-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-700/70 dark:hover:text-white'

  const shuffle = () => {
    setOrder(shuffleArray(order))
    setIndex(0)
    setFlipped(false)
  }

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="flex w-full flex-wrap items-center justify-between gap-2">
        <span className="text-sm text-slate-500 dark:text-slate-400">
          Card {index + 1} of {order.length}
        </span>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={toggleDueOnly}
            className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-semibold transition-colors ${
              dueOnly
                ? 'border-brand-500 bg-brand-50 text-brand-600 dark:bg-brand-900/30 dark:text-brand-300'
                : 'border-slate-200 text-slate-600 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-700/70'
            }`}
          >
            <Icon.Star className={`h-3.5 w-3.5 ${dueOnly ? 'fill-current' : ''}`} />
            {dueOnly ? 'Due only' : `Due (${dueCount})`}
          </button>
          {onRegenerate && (
            <button onClick={onRegenerate} disabled={regenerating} className={fcBtn}>
              <Icon.Refresh className="h-3.5 w-3.5" />
              {regenerating ? 'Generating…' : 'New cards'}
            </button>
          )}
          <button onClick={exportCsv} disabled={exporting} className={fcBtn}>
            <Icon.Download className="h-3.5 w-3.5" />
            {exporting ? 'Exporting…' : 'Anki CSV'}
          </button>
          <button onClick={shuffle} className={fcBtn}>
            <Icon.Shuffle className="h-3.5 w-3.5" />
            Shuffle
          </button>
        </div>
      </div>

      <div
        className="flip-card h-56 w-full max-w-md cursor-pointer"
        onClick={() => setFlipped((f) => !f)}
      >
        <div className={`flip-card-inner ${flipped ? 'flipped' : ''}`}>
          <div className="flip-face rounded-2xl border border-slate-200 bg-white p-6 shadow-md dark:border-slate-700 dark:bg-slate-800">
            <div className="text-center">
              <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-brand-500">
                Front
              </div>
              <div className="text-lg font-medium text-slate-800 dark:text-slate-100">
                {card.front}
              </div>
            </div>
          </div>
          <div className="flip-face flip-face-back rounded-2xl border border-brand-200 bg-brand-50 p-6 shadow-md dark:border-brand-800 dark:bg-brand-900/30">
            <div className="text-center">
              <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-brand-500">
                Back
              </div>
              <div className="text-base text-slate-800 dark:text-slate-100">{card.back}</div>
            </div>
          </div>
        </div>
      </div>

      <p className="text-xs text-slate-400">
        Click or press <kbd className="rounded bg-slate-200 px-1 dark:bg-slate-700">Space</kbd> to flip ·{' '}
        <kbd className="rounded bg-slate-200 px-1 dark:bg-slate-700">←</kbd>
        <kbd className="rounded bg-slate-200 px-1 dark:bg-slate-700">→</kbd> to navigate
      </p>

      {/* Spaced-repetition rating (rate then auto-advance) */}
      {flipped && (
        <div className="flex items-center gap-2">
          <button
            onClick={() => rate('again')}
            className="rounded-lg bg-red-100 px-4 py-1.5 text-sm font-medium text-red-700 hover:bg-red-200 dark:bg-red-900/30 dark:text-red-300"
          >
            Again
          </button>
          <button
            onClick={() => rate('good')}
            className="rounded-lg bg-brand-100 px-4 py-1.5 text-sm font-medium text-brand-700 hover:bg-brand-200 dark:bg-brand-900/30 dark:text-brand-300"
          >
            Good
          </button>
          <button
            onClick={() => rate('easy')}
            className="rounded-lg bg-green-100 px-4 py-1.5 text-sm font-medium text-green-700 hover:bg-green-200 dark:bg-green-900/30 dark:text-green-300"
          >
            Easy
          </button>
        </div>
      )}

      <div className="flex items-center gap-3">
        <button
          onClick={() => go(-1)}
          className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-700"
        >
          ← Previous
        </button>
        <button
          onClick={() => go(1)}
          className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-700"
        >
          Next →
        </button>
      </div>
    </div>
  )
}
