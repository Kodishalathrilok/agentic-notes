import { useMemo, useState } from 'react'
import Icon from './Icons'

// Strip Markdown/formatting markers so previews read as clean plain text.
function cleanPreview(text) {
  if (!text) return ''
  return text
    .replace(/\*\*(.+?)\*\*/g, '$1') // **bold** -> bold
    .replace(/[*_`#>]/g, '') // stray markdown chars
    .replace(/^\s*[•\-]\s*/gm, '') // leading bullets
    .replace(/^\s*\d+[.)]\s*/gm, '') // leading numbering
    .replace(/\s+/g, ' ') // collapse whitespace/newlines
    .trim()
}

// Derive a readable title from the first heading/line of the notes.
function defaultTitle(session) {
  const src = session.notes || session.notes_preview || ''
  if (!src) return ''
  const firstLine = src
    .split('\n')
    .map((l) => l.trim())
    .find((l) => l)
  const cleaned = cleanPreview(firstLine || '')
  if (!cleaned) return ''
  const words = cleaned.split(' ').slice(0, 7).join(' ')
  return words.length < cleaned.length ? words + '…' : words
}

function formatDate(ts) {
  try {
    return new Date(ts).toLocaleString(undefined, {
      dateStyle: 'medium',
      timeStyle: 'short',
    })
  } catch {
    return ts
  }
}

function SessionCard({ session, onLoad, onDelete, onUpdate }) {
  const [editingTitle, setEditingTitle] = useState(false)
  const [title, setTitle] = useState(session.title || '')
  const [tagInput, setTagInput] = useState('')

  const saveTitle = () => {
    onUpdate(session.id, { title: title.trim() })
    setEditingTitle(false)
  }

  const addTag = (e) => {
    e.preventDefault()
    const t = tagInput.trim()
    if (!t) return
    const tags = Array.from(new Set([...(session.tags || []), t]))
    onUpdate(session.id, { tags })
    setTagInput('')
  }

  const removeTag = (t) => {
    onUpdate(session.id, { tags: (session.tags || []).filter((x) => x !== t) })
  }

  return (
    <div className="card p-4 transition-shadow duration-200 hover:shadow-lift">
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="text-xs text-slate-400">{formatDate(session.date)}</span>
        <span className="chip bg-brand-50 capitalize text-brand-700 ring-1 ring-brand-100 dark:bg-brand-900/30 dark:text-brand-300 dark:ring-brand-800/50">
          {session.mode}
        </span>
      </div>

      {/* Title */}
      {editingTitle ? (
        <div className="mb-2 flex gap-2">
          <input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && saveTitle()}
            autoFocus
            placeholder="Session title…"
            className="flex-1 rounded-lg border border-slate-300 px-2 py-1 text-sm dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
          />
          <button onClick={saveTitle} className="text-xs font-medium text-brand-600">
            Save
          </button>
        </div>
      ) : (
        <button
          onClick={() => setEditingTitle(true)}
          className="group mb-1 inline-flex items-center gap-1.5 text-left text-sm font-bold text-slate-800 hover:text-brand-600 dark:text-slate-100 dark:hover:text-brand-400"
          title="Click to rename"
        >
          {session.title || defaultTitle(session) || 'Untitled session'}
          <Icon.Pencil className="h-3 w-3 opacity-0 transition-opacity group-hover:opacity-60" />
        </button>
      )}

      <p className="mb-2 line-clamp-2 text-sm text-slate-600 dark:text-slate-300">
        {cleanPreview(session.notes || session.notes_preview)}
      </p>

      {/* Tags */}
      <div className="mb-3 flex flex-wrap items-center gap-1.5">
        {(session.tags || []).map((t) => (
          <span
            key={t}
            className="flex items-center gap-1 rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600 dark:bg-slate-700 dark:text-slate-300"
          >
            {t}
            <button onClick={() => removeTag(t)} className="text-slate-400 hover:text-red-500">
              ×
            </button>
          </span>
        ))}
        <form onSubmit={addTag} className="inline">
          <input
            value={tagInput}
            onChange={(e) => setTagInput(e.target.value)}
            placeholder="+ tag"
            className="w-16 rounded-full border border-dashed border-slate-300 px-2 py-0.5 text-xs outline-none focus:w-24 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
          />
        </form>
      </div>

      <div className="flex gap-2">
        <button
          onClick={() => onLoad(session)}
          className="rounded-lg bg-brand-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-brand-700"
        >
          Load
        </button>
        <button
          onClick={() => onDelete(session.id)}
          className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-medium text-red-600 hover:bg-red-50 dark:border-slate-600 dark:hover:bg-red-900/20"
        >
          Delete
        </button>
      </div>
    </div>
  )
}

export default function HistoryPanel({ history, onLoad, onDelete, onUpdate }) {
  const [query, setQuery] = useState('')

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return history
    return history.filter((s) => {
      const hay = [
        s.title || '',
        s.mode || '',
        s.notes_preview || '',
        ...(s.tags || []),
      ]
        .join(' ')
        .toLowerCase()
      return hay.includes(q)
    })
  }, [history, query])

  if (!history || history.length === 0) {
    return (
      <div className="flex h-48 items-center justify-center rounded-xl border border-dashed border-slate-300 px-4 text-center text-sm text-slate-400 dark:border-slate-600">
        No past sessions yet. Generate some notes to see them here.
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="relative">
        <Icon.Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search history (title, mode, tag, text)…"
          className="field pl-9"
        />
      </div>

      {filtered.length === 0 ? (
        <div className="py-8 text-center text-sm text-slate-400">No sessions match “{query}”.</div>
      ) : (
        filtered.map((s) => (
          <SessionCard
            key={s.id}
            session={s}
            onLoad={onLoad}
            onDelete={onDelete}
            onUpdate={onUpdate}
          />
        ))
      )}
    </div>
  )
}
