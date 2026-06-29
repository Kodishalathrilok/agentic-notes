import { useState, useEffect, useRef } from 'react'
import Icon from './Icons'

// Render inline **bold** spans and [n] citation pills within a line.
function renderInline(text, keyPrefix, onCite) {
  const parts = text.split(/(\*\*.+?\*\*|\[\d+\])/g)
  return parts.map((part, i) => {
    const bold = part.match(/^\*\*(.+?)\*\*$/)
    if (bold) {
      return (
        <strong key={`${keyPrefix}-${i}`} className="font-semibold text-slate-900 dark:text-white">
          {bold[1]}
        </strong>
      )
    }
    const cite = part.match(/^\[(\d+)\]$/)
    if (cite && onCite) {
      const n = Number(cite[1])
      return (
        <button
          key={`${keyPrefix}-${i}`}
          onClick={() => onCite(n)}
          className="mx-0.5 inline-flex translate-y-[-1px] items-center rounded-md bg-brand-100 px-1.5 text-[10px] font-bold text-brand-700 align-super hover:bg-brand-200 dark:bg-brand-900/40 dark:text-brand-300"
          title={`Jump to source ${n}`}
        >
          {n}
        </button>
      )
    }
    return <span key={`${keyPrefix}-${i}`}>{part}</span>
  })
}

function renderNotes(notes, onCite) {
  const lines = notes.split('\n')
  const out = []

  // Handle fenced ``` code blocks first.
  let inCode = false
  let codeBuffer = []
  let codeKey = 0

  lines.forEach((line, idx) => {
    const trimmed = line.trim()

    if (trimmed.startsWith('```')) {
      if (inCode) {
        out.push(
          <pre
            key={`code-${codeKey++}`}
            className="my-2 overflow-x-auto rounded-lg bg-slate-900 p-3 text-xs text-slate-100 dark:bg-black"
          >
            <code>{codeBuffer.join('\n')}</code>
          </pre>
        )
        codeBuffer = []
        inCode = false
      } else {
        inCode = true
      }
      return
    }
    if (inCode) {
      codeBuffer.push(line)
      return
    }

    if (!trimmed) {
      out.push(<div key={idx} className="h-2" />)
      return
    }

    const headerMatch = trimmed.match(/^\*\*(.+?):?\*\*$/)
    if (headerMatch) {
      out.push(
        <h3 key={idx} className="mt-4 mb-1 text-base font-bold text-brand-700 dark:text-brand-400">
          {headerMatch[1]}
        </h3>
      )
      return
    }

    if (trimmed.startsWith('•')) {
      out.push(
        <div key={idx} className="flex gap-2 py-0.5 text-sm leading-relaxed text-slate-700 dark:text-slate-200">
          <span className="mt-1.5 h-1.5 w-1.5 rotate-45 shrink-0 bg-brand-500" />
          <span>{renderInline(trimmed.slice(1).trim(), idx, onCite)}</span>
        </div>
      )
      return
    }

    if (trimmed.startsWith('-')) {
      out.push(
        <div key={idx} className="ml-6 flex gap-2 py-0.5 text-sm leading-relaxed text-slate-600 dark:text-slate-300">
          <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-slate-400" />
          <span>{renderInline(trimmed.slice(1).trim(), idx, onCite)}</span>
        </div>
      )
      return
    }

    if (/^\d+[.)]/.test(trimmed)) {
      out.push(
        <div key={idx} className="py-0.5 text-sm leading-relaxed text-slate-700 dark:text-slate-200">
          {renderInline(trimmed, idx, onCite)}
        </div>
      )
      return
    }

    out.push(
      <p key={idx} className="py-0.5 text-sm leading-relaxed text-slate-700 dark:text-slate-200">
        {renderInline(trimmed, idx, onCite)}
      </p>
    )
  })

  // Flush an unterminated code block.
  if (codeBuffer.length) {
    out.push(
      <pre
        key={`code-${codeKey++}`}
        className="my-2 overflow-x-auto rounded-lg bg-slate-900 p-3 text-xs text-slate-100 dark:bg-black"
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
  onRewrite,
  rewriting,
  sources = [],
}) {
  const [copied, setCopied] = useState(false)
  const [busy, setBusy] = useState(null)
  const [editing, setEditing] = useState(false)
  const [showDiff, setShowDiff] = useState(false)
  const [speaking, setSpeaking] = useState(false)
  const [showSources, setShowSources] = useState(false)
  const [highlight, setHighlight] = useState(null)
  const containerRef = useRef(null)

  // Citation [n] clicked -> open the Sources panel and scroll to passage n.
  const handleCite = (n) => {
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
      <div className="flex h-48 items-center justify-center rounded-xl border border-dashed border-slate-300 text-sm text-slate-400 dark:border-slate-600">
        Your generated notes will appear here.
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
      const res = await fetch(`/api/export/${kind}`, {
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

  const btn =
    'inline-flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 transition-colors cursor-pointer hover:bg-slate-100 hover:text-slate-900 disabled:opacity-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-700/70 dark:hover:text-white'

  const diff = showDiff && notesBefore ? diffLines(notesBefore, notes) : null

  return (
    <div>
      {/* Action buttons */}
      <div className="mb-3 flex flex-wrap gap-2">
        <button onClick={copy} className={btn}>
          {copied ? <Icon.Check className="h-3.5 w-3.5 text-green-500" /> : <Icon.Copy className="h-3.5 w-3.5" />}
          {copied ? 'Copied' : 'Copy'}
        </button>
        <button onClick={() => exportFile('pdf')} disabled={busy === 'pdf'} className={btn}>
          {busy === 'pdf' ? 'Exporting…' : 'Export PDF'}
        </button>
        <button onClick={() => exportFile('markdown')} disabled={busy === 'markdown'} className={btn}>
          {busy === 'markdown' ? 'Exporting…' : 'Export MD'}
        </button>
        <button onClick={() => exportFile('docx')} disabled={busy === 'docx'} className={btn}>
          {busy === 'docx' ? 'Exporting…' : 'Export DOCX'}
        </button>
        <button onClick={toggleSpeak} className={btn}>
          {speaking ? <Icon.Stop className="h-3.5 w-3.5" /> : <Icon.Volume className="h-3.5 w-3.5" />}
          {speaking ? 'Stop' : 'Read aloud'}
        </button>
        <button onClick={() => setEditing((e) => !e)} className={btn}>
          {editing ? 'Done editing' : 'Edit'}
        </button>
        {onRewrite && (
          <>
            <button onClick={() => onRewrite('shorter')} disabled={!!rewriting} className={btn}>
              {rewriting === 'shorter' ? 'Shortening…' : 'Make shorter'}
            </button>
            <button onClick={() => onRewrite('longer')} disabled={!!rewriting} className={btn}>
              {rewriting === 'longer' ? 'Expanding…' : 'Make longer'}
            </button>
          </>
        )}
        {notesBefore && (
          <button onClick={() => setShowDiff((d) => !d)} className={btn}>
            {showDiff ? 'Hide changes' : 'Show changes'}
          </button>
        )}
      </div>

      {/* Content */}
      {editing ? (
        <textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          rows={18}
          className="w-full resize-y rounded-xl border border-slate-300 bg-white p-4 font-mono text-sm text-slate-800 outline-none focus:border-brand-500 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
        />
      ) : diff ? (
        <div className="scroll-area max-h-[60vh] overflow-y-auto rounded-xl border border-slate-200 bg-white p-5 font-mono text-xs leading-relaxed dark:border-slate-700 dark:bg-slate-800">
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
        <div
          ref={containerRef}
          className="scroll-area max-h-[60vh] overflow-y-auto rounded-xl border border-slate-200 bg-white p-5 dark:border-slate-700 dark:bg-slate-800"
        >
          {renderNotes(notes, sources.length ? handleCite : null)}
        </div>
      )}

      {/* Citations / sources */}
      {sources.length > 0 && !editing && !diff && (
        <div className="mt-3">
          <button onClick={() => setShowSources((s) => !s)} className={btn}>
            {showSources ? 'Hide sources' : `Sources (${sources.length})`}
          </button>
          {showSources && (
            <div className="scroll-area mt-2 max-h-[40vh] space-y-2 overflow-y-auto">
              {sources.map((s) => (
                <div
                  id={`src-${s.id}`}
                  key={s.id}
                  className={`rounded-lg border p-3 text-xs leading-relaxed transition-colors ${
                    highlight === s.id
                      ? 'border-brand-400 bg-brand-50 dark:border-brand-600 dark:bg-brand-900/20'
                      : 'border-slate-200 dark:border-slate-700'
                  }`}
                >
                  <span className="mr-2 font-bold text-brand-600 dark:text-brand-400">[{s.id}]</span>
                  <span className="text-slate-600 dark:text-slate-300">{s.text}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
