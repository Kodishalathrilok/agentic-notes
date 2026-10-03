import { useState, useEffect, useRef } from 'react'
import { apiFetch } from '../lib/api'
import Icon from './Icons'
import InlineAssistant from './InlineAssistant'

// A run of citations, rendered as page chips.
//
// The marker the model emits is a PASSAGE index, which meant the notes showed
// a bare superscript number that read exactly like a page reference — the one
// question a reader actually has ("where in my document is this?") answered
// with an internal retrieval id. Where the passage carries a page, show that
// instead; passage numbers only survive for sources without pages (pasted
// text, URLs, transcripts) where there is nothing better to show.
// A chunk cut against page spans sits on exactly one page. A chunk from the
// fallback path (spans missing or stale) can cover several, and it reports all
// of them — "p. 20-22" is coarse but true, where naming just the first was the
// bug that put every citation 1-3 pages early.
function pageLabel(pages) {
  if (!pages || !pages.length) return null
  if (pages.length === 1) return `p. ${pages[0]}`
  return `p. ${pages[0]}-${pages[pages.length - 1]}`
}

function CiteChips({ ids, cite, keyPrefix }) {
  const seen = new Set()
  const items = []
  for (const n of ids) {
    const pages = cite.pagesOf(n)
    const label = pageLabel(pages)
    const page = pages && pages.length ? pages[0] : null
    const key = label ? `p${label}` : `s${n}`
    if (seen.has(key)) continue // the same page cited twice in one sentence
    seen.add(key)
    items.push({ n, page, label: label || `[${n}]` })
  }

  const shown = items.slice(0, 2)
  const extra = items.length - shown.length

  return (
    <span className="ml-1 inline-flex items-center gap-1 align-baseline">
      {shown.map((it) => (
        <button
          key={`${keyPrefix}-c${it.n}`}
          onClick={() => cite.onCite(it.n)}
          title={it.page ? `Go to page ${it.page}` : `Show passage ${it.n}`}
          className="inline-flex items-center rounded-md bg-neutral-100 px-1.5 py-0.5 text-[11px] font-medium leading-none text-neutral-500 transition-colors hover:bg-neutral-200 hover:text-neutral-900"
        >
          {it.label}
        </button>
      ))}
      {extra > 0 && (
        <button
          onClick={() => cite.onCite(items[shown.length].n)}
          title={`${extra} more source${extra === 1 ? '' : 's'}`}
          className="inline-flex items-center rounded-md bg-neutral-100 px-1.5 py-0.5 text-[11px] font-medium leading-none text-neutral-400 transition-colors hover:bg-neutral-200 hover:text-neutral-900"
        >
          +{extra}
        </button>
      )}
    </span>
  )
}

// Render inline **bold** spans and [n] citation runs within a line.
// Consecutive markers ("[2][5][7]") are captured as ONE token so they can be
// collapsed into a single chip group rather than three floating numbers.
function renderInline(text, keyPrefix, cite) {
  const parts = text.split(/(\*\*.+?\*\*|\[\d+\](?:\s*\[\d+\])*)/g)
  return parts.map((part, i) => {
    const bold = part.match(/^\*\*(.+?)\*\*$/)
    if (bold) {
      return (
        <strong key={`${keyPrefix}-${i}`} className="font-semibold text-neutral-900">
          {bold[1]}
        </strong>
      )
    }
    if (cite && /^\[\d+\]/.test(part)) {
      const ids = [...part.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1]))
      return <CiteChips key={`${keyPrefix}-${i}`} ids={ids} cite={cite} keyPrefix={`${keyPrefix}-${i}`} />
    }
    return <span key={`${keyPrefix}-${i}`}>{part}</span>
  })
}

function renderNotes(notes, cite) {

  const lines = notes.split('\n')
  const out = []

  // Handle fenced ``` code blocks first.
  let inCode = false
  let codeBuffer = []
  let codeKey = 0
  let codeStart = 0

  // data-line / data-line-start+end tag each element with the SOURCE line(s)
  // it renders, so an inline-edit selection can be mapped back to the raw
  // markdown (see lib/lineAnchors.js). No visual effect.
  lines.forEach((line, idx) => {
    const trimmed = line.trim()

    if (trimmed.startsWith('```')) {
      if (inCode) {
        out.push(
          <pre
            key={`code-${codeKey++}`}
            data-line-start={codeStart}
            data-line-end={idx}
 className="my-2 overflow-x-auto rounded-lg bg-espresso-900 p-3 text-xs text-slate-100 "
          >
            <code>{codeBuffer.join('\n')}</code>
          </pre>
        )
        codeBuffer = []
        inCode = false
      } else {
        inCode = true
        codeStart = idx
      }
      return
    }
    if (inCode) {
      codeBuffer.push(line)
      return
    }

    if (!trimmed) {
      out.push(<div key={idx} data-line={idx} className="h-2" />)
      return
    }

    const headerMatch = trimmed.match(/^\*\*(.+?):?\*\*$/)
    if (headerMatch) {
      out.push(
        <h3 key={idx} data-line={idx} className="mb-1.5 mt-6 text-[17px] font-bold leading-snug text-neutral-900 first:mt-0">
          {headerMatch[1]}
        </h3>
      )
      return
    }

    if (trimmed.startsWith('•')) {
      out.push(
        <div key={idx} data-line={idx} className="flex gap-2.5 py-1 text-[14px] leading-[24px] text-neutral-700">
          <span className="mt-[0.6rem] h-[5px] w-[5px] shrink-0 rounded-full bg-neutral-300" />
          <span>{renderInline(trimmed.slice(1).trim(), idx, cite)}</span>
        </div>
      )
      return
    }

    if (trimmed.startsWith('-')) {
      out.push(
        <div key={idx} data-line={idx} className="ml-5 flex gap-2.5 py-1 text-[14px] leading-[24px] text-neutral-600">
          <span className="mt-[0.6rem] h-[5px] w-[5px] shrink-0 rounded-full bg-neutral-200" />
          <span>{renderInline(trimmed.slice(1).trim(), idx, cite)}</span>
        </div>
      )
      return
    }

    if (/^\d+[.)]/.test(trimmed)) {
      out.push(
        <div key={idx} data-line={idx} className="py-1 text-[14px] leading-[24px] text-neutral-700">
          {renderInline(trimmed, idx, cite)}
        </div>
      )
      return
    }

    out.push(
      <p key={idx} data-line={idx} className="py-1 text-[14px] leading-[24px] text-neutral-700">
        {renderInline(trimmed, idx, cite)}
      </p>
    )
  })

  // Flush an unterminated code block.
  if (codeBuffer.length) {
    out.push(
      <pre
        key={`code-${codeKey++}`}
        data-line-start={codeStart}
        data-line-end={lines.length - 1}
 className="my-2 overflow-x-auto rounded-lg bg-espresso-900 p-3 text-xs text-slate-100 "
      >
        <code>{codeBuffer.join('\n')}</code>
      </pre>
    )
  }

  return out
}

// Simple line-based LCS diff -> [{ type: 'same'|'add'|'del', text }]
function diffLines(before, after) {
  const a = before.split('\n')
  const b = after.split('\n')
  const m = a.length
  const n = b.length
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0))
  for (let i = m - 1; i >= 0; i--) {
    for (let j = n - 1; j >= 0; j--) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1])
    }
  }
  const result = []
  let i = 0
  let j = 0
  while (i < m && j < n) {
    if (a[i] === b[j]) {
      result.push({ type: 'same', text: a[i] })
      i++
      j++
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      result.push({ type: 'del', text: a[i] })
      i++
    } else {
      result.push({ type: 'add', text: b[j] })
      j++
    }
  }
  while (i < m) result.push({ type: 'del', text: a[i++] })
  while (j < n) result.push({ type: 'add', text: b[j++] })
  return result
}

function MenuItem({ onClick, label, disabled = false }) {
  return (
    <button
      role="menuitem"
      onClick={onClick}
      disabled={disabled}
      className="block w-full px-4 py-2 text-left text-sm text-neutral-700 transition-colors hover:bg-neutral-50 hover:text-neutral-900 disabled:pointer-events-none disabled:opacity-40"
    >
      {label}
    </button>
  )
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

export default function NotesOutput({
  notes,
  setNotes,
  quiz,
  flashcards,
  notesBefore,
  unverified = false,
  onRewrite,
  rewriting,
  sources = [],
  onEditSelection,
  onCitePage,
  streaming = false,
}) {
  const [copied, setCopied] = useState(false)
  const [busy, setBusy] = useState(null)
  const [editing, setEditing] = useState(false)
  const [showDiff, setShowDiff] = useState(false)
  const [speaking, setSpeaking] = useState(false)
  const [showSources, setShowSources] = useState(false)
  const [highlight, setHighlight] = useState(null)
  const [menuOpen, setMenuOpen] = useState(false)
  const containerRef = useRef(null)
  const menuRef = useRef(null)
  const menuBtnRef = useRef(null)

  // Which page a citation points at, for the p. N chips.
  // Pages come from the retrieved chunk's metadata, never from the model.
  const pagesOf = (n) => {
    const src = sources.find((s) => s.id === n)
    if (!src) return null
    if (src.pages && src.pages.length) return src.pages
    return src.page ? [src.page] : null
  }
  const pageOf = (n) => pagesOf(n)?.[0] ?? null

  // Dismiss the overflow menu on an outside click or Escape.
  useEffect(() => {
    if (!menuOpen) return
    const onDown = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) setMenuOpen(false)
    }
    const onKey = (e) => {
      if (e.key !== 'Escape') return
      setMenuOpen(false)
      menuBtnRef.current?.focus()
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [menuOpen])

  // Citation [n] clicked.
  //
  // When the source is a PDF the passage carries the page it came from, so the
  // useful thing is to move the document itself to that page — the reader gets
  // the claim in its original context instead of a text excerpt. Only sources
  // without a page (pasted text, URLs, transcripts) fall back to unfolding the
  // Sources list, which is a lot of panel to throw open when the answer is
  // already sitting in the left pane.
  const handleCite = (n) => {
    const cited = sources.find((s) => s.id === n)
    if (cited?.page && onCitePage) {
      onCitePage(cited.page)
      setHighlight(n)
      setTimeout(() => setHighlight(null), 2500)
      return
    }
    setShowSources(true)
    setHighlight(n)
    setTimeout(() => {
      document.getElementById(`src-${n}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }, 50)
    setTimeout(() => setHighlight(null), 2500)
  }

  // Render LaTeX math with KaTeX whenever the notes view updates.
  useEffect(() => {
    if (editing || showDiff) return
    const el = containerRef.current
    if (el && window.renderMathInElement) {
      try {
        window.renderMathInElement(el, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\(', right: '\\)', display: false },
            { left: '\\[', right: '\\]', display: true },
          ],
          throwOnError: false,
        })
      } catch {
        /* ignore */
      }
    }
  }, [notes, editing, showDiff])

  // Stop speech if the component unmounts.
  useEffect(() => () => window.speechSynthesis?.cancel(), [])

  const toggleSpeak = () => {
    const synth = window.speechSynthesis
    if (!synth) return
    if (speaking) {
      synth.cancel()
      setSpeaking(false)
      return
    }
    const plain = notes.replace(/\*\*/g, '').replace(/[•#`]/g, '')
    const utter = new SpeechSynthesisUtterance(plain)
    utter.onend = () => setSpeaking(false)
    utter.onerror = () => setSpeaking(false)
    synth.cancel()
    synth.speak(utter)
    setSpeaking(true)
  }

  if (!notes && !editing) {
    return (
      <div className="flex h-56 flex-col items-center justify-center gap-3 rounded-3xl border border-dashed border-neutral-300 text-sm text-neutral-500">
        {streaming ? (
          <>
            <span className="relative flex h-2.5 w-2.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-brand-500 opacity-75" />
              <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-brand-500" />
            </span>
            Agents are reading your source…
          </>
        ) : (
          'Your generated notes will appear here.'
        )}
      </div>
    )
  }

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(notes)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* ignore */
    }
  }

  const exportFile = async (kind) => {
    setBusy(kind)
    try {
      const res = await apiFetch(`/api/export/${kind}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes, quiz, flashcards }),
      })
      if (!res.ok) throw new Error('Export failed')
      const blob = await res.blob()
      const names = { pdf: 'notes.pdf', markdown: 'notes.md', docx: 'notes.docx' }
      triggerDownload(blob, names[kind] || 'notes.txt')
    } catch {
      /* ignore */
    } finally {
      setBusy(null)
    }
  }

  // Not the global `.pill` — that styles text with espresso-*, and every
  // espresso shade in tailwind.config.js is #000000, so "muted" renders solid
  // black. Muted text has to come from neutral-*.
  const btn =
    'inline-flex items-center gap-1.5 rounded-full border border-neutral-200 bg-white px-3.5 py-1.5 ' +
    'text-xs font-medium text-neutral-600 transition-colors hover:border-neutral-300 ' +
    'hover:bg-neutral-50 hover:text-neutral-900 disabled:pointer-events-none disabled:opacity-40'

  const diff = showDiff && notesBefore ? diffLines(notesBefore, notes) : null

  return (
    <div>
      {/* Streaming indicator */}
      {streaming && (
        <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-brand-500/30 bg-brand-500/10 px-4 py-1.5 text-xs font-semibold text-brand-700">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-brand-500 opacity-75" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-brand-500" />
          </span>
          Agents writing…
        </div>
      )}

      {/* Action row.
          Nine same-shaped pills wrapped to two rows and pushed the notes past
          the middle of the screen, with nothing to say which of them mattered.
          Three live controls plus an overflow: the ones you reach for while
          reading stay out, the ones you use once at the end fold away. */}
      <div className="mb-4 flex items-center justify-end gap-1.5">
        {/* Rewrites and inline edits are sent the notes only, never the
            sources, so their output skips citation validation and grounding.
            Say so, rather than let edited text pass for checked text. */}
        {unverified && (
          <span
            role="status"
            title="Rewrites and inline edits are not re-checked against the source. Citations the notes did not already have are removed, but the edited text has not been through citation validation or grounding."
            className="mr-auto rounded-full border border-amber-300 bg-amber-50 px-2.5 py-1 text-[11px] font-semibold text-amber-800"
          >
            Edited — not re-verified
          </span>
        )}
        <button onClick={copy} className={btn}>
          {copied ? <Icon.Check className="h-3.5 w-3.5 text-green-600" /> : <Icon.Copy className="h-3.5 w-3.5" />}
          {copied ? 'Copied' : 'Copy'}
        </button>
        <button onClick={() => setEditing((e) => !e)} className={btn}>
          <Icon.Pencil className="h-3.5 w-3.5" />
          {editing ? 'Done' : 'Edit'}
        </button>
        <button onClick={toggleSpeak} className={btn}>
          {speaking ? <Icon.Stop className="h-3.5 w-3.5" /> : <Icon.Volume className="h-3.5 w-3.5" />}
          {speaking ? 'Stop' : 'Listen'}
        </button>

        <div className="relative" ref={menuRef}>
          <button
            ref={menuBtnRef}
            onClick={() => setMenuOpen((o) => !o)}
            aria-label="More actions"
            aria-haspopup="menu"
            aria-expanded={menuOpen}
            className="flex h-8 w-8 items-center justify-center rounded-full text-neutral-400 transition-colors hover:bg-neutral-100 hover:text-neutral-900"
          >
            <Icon.More className="h-4 w-4" />
          </button>

          {menuOpen && (
            <div
              role="menu"
              aria-label="More actions"
              className="absolute right-0 z-30 mt-1 w-52 overflow-hidden rounded-2xl border border-neutral-200 bg-white py-1 shadow-[0_12px_32px_rgba(0,0,0,0.12)]"
            >
              {onRewrite && (
                <>
                  <MenuItem
                    onClick={() => { onRewrite('shorter'); setMenuOpen(false) }}
                    disabled={!!rewriting}
                    label={rewriting === 'shorter' ? 'Shortening…' : 'Make shorter'}
                  />
                  <MenuItem
                    onClick={() => { onRewrite('longer'); setMenuOpen(false) }}
                    disabled={!!rewriting}
                    label={rewriting === 'longer' ? 'Expanding…' : 'Make longer'}
                  />
                  <div className="my-1 border-t border-neutral-100" />
                </>
              )}
              <MenuItem
                onClick={() => { exportFile('pdf'); setMenuOpen(false) }}
                disabled={busy === 'pdf'}
                label={busy === 'pdf' ? 'Exporting…' : 'Export as PDF'}
              />
              <MenuItem
                onClick={() => { exportFile('markdown'); setMenuOpen(false) }}
                disabled={busy === 'markdown'}
                label={busy === 'markdown' ? 'Exporting…' : 'Export as Markdown'}
              />
              <MenuItem
                onClick={() => { exportFile('docx'); setMenuOpen(false) }}
                disabled={busy === 'docx'}
                label={busy === 'docx' ? 'Exporting…' : 'Export as DOCX'}
              />
              {(notesBefore || sources.length > 0) && <div className="my-1 border-t border-neutral-100" />}
              {notesBefore && (
                <MenuItem
                  onClick={() => { setShowDiff((d) => !d); setMenuOpen(false) }}
                  label={showDiff ? 'Hide changes' : 'Show changes'}
                />
              )}
              {sources.length > 0 && (
                <MenuItem
                  onClick={() => { setShowSources((s) => !s); setMenuOpen(false) }}
                  label={showSources ? 'Hide sources' : `Sources (${sources.length})`}
                />
              )}
            </div>
          )}
        </div>
      </div>

      {/* Content */}
      {editing ? (
        <textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          rows={18}
 className="field resize-y rounded-3xl p-5 font-mono text-sm"
        />
      ) : diff ? (
        <div className="rounded-3xl border border-neutral-200 bg-white p-5 font-mono text-xs leading-relaxed">
          {diff.map((d, i) => (
            <div
              key={i}
 className={
                d.type === 'add'
                  ? 'bg-green-50 text-green-700 dark:bg-green-900/20 dark:text-green-300'
                  : d.type === 'del'
                    ? 'bg-red-50 text-red-600 line-through dark:bg-red-900/20 dark:text-red-300'
                    : 'text-slate-600 dark:text-slate-300'
              }
            >
              <span className="mr-2 select-none opacity-50">
                {d.type === 'add' ? '+' : d.type === 'del' ? '−' : ' '}
              </span>
              {d.text || ' '}
            </div>
          ))}
        </div>
      ) : (
        <>
          {/* Reading surface.
              No max-height and no overflow of its own: the output column above
              already scrolls, and having both meant two scrollbars side by side
              with the wheel captured by whichever the pointer happened to be
              over. One scroll region, and the notes fill the column. */}
          <div
            ref={containerRef}
            className="rounded-3xl border border-neutral-200 bg-white px-6 py-7 sm:px-8"
          >
            <div className="mx-auto max-w-3xl">
              {renderNotes(notes, sources.length ? { onCite: handleCite, pageOf, pagesOf } : null)}
              {streaming && <span className="stream-caret" aria-hidden />}
            </div>
          </div>
          <InlineAssistant containerRef={containerRef} notes={notes} onEditSelection={onEditSelection} />
        </>
      )}

      {/* Citations / sources. The toggle lives in the overflow menu now — a
          second copy here was the ninth pill competing with the notes. */}
      {sources.length > 0 && showSources && !editing && !diff && (
        <div className="mt-3 space-y-2">
          {sources.map((s) => (
            <div
              id={`src-${s.id}`}
              key={s.id}
              className={`rounded-xl border p-3 text-xs leading-relaxed transition-colors ${
                highlight === s.id ? 'border-brand-400 bg-brand-50' : 'border-neutral-200 bg-white'
              }`}
            >
              {/* Only PDFs carry pages; pasted text and transcripts don't. */}
              {s.page ? (
                <span className="mr-2 rounded bg-neutral-100 px-1.5 py-0.5 text-[11px] font-medium text-neutral-500">
                  p. {s.page}
                </span>
              ) : (
                <span className="mr-2 rounded bg-neutral-100 px-1.5 py-0.5 text-[11px] font-medium text-neutral-500">
                  [{s.id}]
                </span>
              )}
              <span className="text-neutral-600">{s.text}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
