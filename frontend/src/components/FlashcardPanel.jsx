import { useMemo, useState, useEffect } from 'react'
import { apiFetch } from '../lib/api'
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

const RATE_STYLES = {
  again: 'bg-red-500/15 text-red-300 hover:bg-red-500/25',
  good: 'bg-brand-500/20 text-brand-700 hover:bg-brand-500/30',
  easy: 'bg-green-500/15 text-green-300 hover:bg-green-500/25',
}

export default function FlashcardPanel({ flashcards, onRegenerate, regenerating }) {
  const parsed = useMemo(() => parseFlashcards(flashcards), [flashcards])
  const [order, setOrder] = useState(parsed)
  const [flippedIds, setFlippedIds] = useState(() => new Set())
  const [exporting, setExporting] = useState(false)
  const [srs, setSrs] = useState(loadSRS)
  const [dueOnly, setDueOnly] = useState(false)

  useEffect(() => {
    setOrder(parsed)
    setFlippedIds(new Set())
    setDueOnly(false)
  }, [parsed])

  const dueCount = parsed.filter((c) => isDue(c, srs)).length

  const toggleFlip = (i) => {
    setFlippedIds((prev) => {
      const next = new Set(prev)
      if (next.has(i)) next.delete(i)
      else next.add(i)
      return next
    })
  }

  const rate = (card, level, i) => {
    const next = { ...srs, [cardKey(card)]: schedule(srs[cardKey(card)], level) }
    setSrs(next)
    saveSRS(next)
    // unflip the rated card
    setFlippedIds((prev) => {
      const n = new Set(prev)
      n.delete(i)
      return n
    })
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
    setFlippedIds(new Set())
  }

  const shuffle = () => {
    setOrder(shuffleArray(order))
    setFlippedIds(new Set())
  }

  const exportCsv = async () => {
    setExporting(true)
    try {
      const res = await apiFetch('/api/export/flashcards-csv', {
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
      <div className="card flex h-56 flex-col items-center justify-center gap-3 text-sm text-espresso-500">
        {regenerating ? (
          <>
            <span className="h-6 w-6 animate-spin rounded-full border-2 border-espresso-400 border-t-transparent" />
            Building spaced-repetition flashcards…
          </>
        ) : (
          <>
            Flashcards are generated on demand from your notes.
            {onRegenerate && (
              <button onClick={onRegenerate} className="btn-primary px-5 py-2 text-sm">
                <Icon.Sparkles className="h-4 w-4" />
                Generate flashcards
              </button>
            )}
          </>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-5">
      {/* Controls — pill group */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <span className="text-sm text-espresso-500">
          {order.length} cards · {dueCount} due
        </span>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={toggleDueOnly}
 className={`pill ${dueOnly ? 'border-brand-500/50 bg-brand-500/15 text-brand-700' : ''}`}
          >
            <Icon.Star className={`h-3.5 w-3.5 ${dueOnly ? 'fill-current' : ''}`} />
            {dueOnly ? 'Due only' : `Due (${dueCount})`}
          </button>
          {onRegenerate && (
            <button onClick={onRegenerate} disabled={regenerating} className="pill">
              <Icon.Refresh className="h-3.5 w-3.5" />
              {regenerating ? 'Generating…' : 'New cards'}
            </button>
          )}
          <button onClick={exportCsv} disabled={exporting} className="pill">
            <Icon.Download className="h-3.5 w-3.5" />
            {exporting ? 'Exporting…' : 'Anki CSV'}
          </button>
          <button onClick={shuffle} className="pill">
            <Icon.Shuffle className="h-3.5 w-3.5" />
            Shuffle
          </button>
        </div>
      </div>

      {/* Card grid — mobile-first: 1 col → 2 (sm) → 3 (xl) */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-3">
        {order.map((card, i) => {
          const flipped = flippedIds.has(i)
          return (
            <div
              key={`${cardKey(card)}-${i}`}
              onClick={() => toggleFlip(i)}
 className="flip-card h-56 cursor-pointer transition-transform duration-300 hover:-translate-y-1.5"
            >
              <div className={`flip-card-inner ${flipped ? 'flipped' : ''}`}>
                {/* Front */}
                <div className="flip-face rounded-3xl border border-espresso-900/10 bg-white p-6 shadow-card transition-shadow hover:shadow-lift">
                  <div className="text-center">
                    <div className="mb-3 text-[10px] font-bold uppercase tracking-[0.25em] text-brand-600">
                      Front
                    </div>
                    <div className="text-lg font-semibold leading-snug text-espresso-900">{card.front}</div>
                    <div className="mt-4 text-[11px] text-espresso-500">tap to flip</div>
                  </div>
                </div>
                {/* Back */}
                <div className="flip-face flip-face-back flex-col rounded-3xl border border-espresso-900/30 bg-gradient-to-br from-espresso-700 to-espresso-900 p-6 shadow-card">
                  <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
                    <div>
                      <div className="mb-2 text-[10px] font-bold uppercase tracking-[0.25em] text-brand-300">
                        Back
                      </div>
                      <div className="text-sm leading-relaxed text-latte-100">{card.back}</div>
                    </div>
                    <div className="flex gap-2" onClick={(e) => e.stopPropagation()}>
                      {['again', 'good', 'easy'].map((level) => (
                        <button
                          key={level}
                          onClick={() => rate(card, level, i)}
 className={`rounded-full px-4 py-1.5 text-xs font-semibold capitalize transition-colors ${RATE_STYLES[level]}`}
                        >
                          {level}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}