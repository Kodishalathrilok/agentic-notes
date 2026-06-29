import { useState, useEffect, useRef } from 'react'
import Icon from './Icons'

// --- tiny markdown renderer for the answer (bold + bullets) ---
function renderInline(text, key) {
  return text.split(/(\*\*.+?\*\*)/g).map((p, i) => {
    const m = p.match(/^\*\*(.+?)\*\*$/)
    return m ? (
      <strong key={`${key}-${i}`} className="font-semibold">
        {m[1]}
      </strong>
    ) : (
      <span key={`${key}-${i}`}>{p}</span>
    )
  })
}
function renderMd(text) {
  return text.split('\n').map((line, idx) => {
    const t = line.trim()
    if (!t) return <div key={idx} className="h-1.5" />
    const b = t.match(/^[*\-+]\s+(.*)$/)
    if (b) {
      return (
        <div key={idx} className="flex gap-1.5">
          <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-current opacity-50" />
          <span>{renderInline(b[1], idx)}</span>
        </div>
      )
    }
    return <p key={idx}>{renderInline(t, idx)}</p>
  })
}

const POP_W = 320

export default function InlineAssistant({ containerRef, notes, onEditSelection }) {
  const [sel, setSel] = useState(null) // { text, rect:{top,bottom,left,right} }
  const [open, setOpen] = useState(false)
  const [instruction, setInstruction] = useState('')
  const [answer, setAnswer] = useState('')
  const [loading, setLoading] = useState(false)
  const popRef = useRef(null)
  const answerRef = useRef(null)

  // Detect a selection inside the notes container.
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const onUp = () => {
      const s = window.getSelection()
      const text = s && s.toString().trim()
      if (text && text.length > 1 && s.anchorNode && el.contains(s.anchorNode)) {
        const r = s.getRangeAt(0).getBoundingClientRect()
        setSel({ text, rect: { top: r.top, bottom: r.bottom, left: r.left, right: r.right } })
      } else if (!open) {
        setSel(null)
      }
    }
    el.addEventListener('mouseup', onUp)
    return () => el.removeEventListener('mouseup', onUp)
  }, [containerRef, open])

  // Close on outside click / Escape.
  useEffect(() => {
    if (!open) return
    const onDown = (e) => popRef.current && !popRef.current.contains(e.target) && close()
    const onKey = (e) => e.key === 'Escape' && close()
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  // Render LaTeX in the streamed answer.
  useEffect(() => {
    if (answerRef.current && window.renderMathInElement) {
      try {
        window.renderMathInElement(answerRef.current, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\(', right: '\\)', display: false },
          ],
          throwOnError: false,
        })
      } catch {
        /* ignore */
      }
    }
  }, [answer])

  const close = () => {
    setOpen(false)
    setAnswer('')
    setInstruction('')
    setLoading(false)
    setSel(null)
  }

  const explain = async () => {
    if (!sel) return
    setLoading(true)
    setAnswer('')
    const question = instruction
      ? `${instruction} (about this excerpt: "${sel.text}")`
      : `Explain this part of the notes simply: "${sel.text}"`
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes, question, history: [] }),
      })
      if (!res.ok) throw new Error()
      const reader = res.body.getReader()
      const dec = new TextDecoder()
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        setAnswer((a) => a + dec.decode(value, { stream: true }))
      }
    } catch {
      setAnswer('Sorry — something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  const rewrite = async () => {
    if (!sel || !onEditSelection) return
    setLoading(true)
    try {
      await onEditSelection(sel.text, instruction)
      close()
    } catch {
      setLoading(false)
    }
  }

  if (!sel) return null

  const r = sel.rect
  const midY = (r.top + r.bottom) / 2

  // Prefer opening to the RIGHT of the selection; fall back to the left; clamp.
  const fitsRight = r.right + 8 + POP_W <= window.innerWidth - 8
  let left = fitsRight ? r.right + 8 : r.left - POP_W - 8
  left = Math.min(Math.max(8, left), window.innerWidth - POP_W - 8)
  const top = Math.max(8, Math.min(r.top, window.innerHeight - 80))
  const maxHeight = window.innerHeight - top - 16

  return (
    <>
      {!open && (
        <button
          style={{
            position: 'fixed',
            top: midY,
            left: Math.min(r.right + 8, window.innerWidth - 92),
            transform: 'translateY(-50%)',
            zIndex: 50,
          }}
          onMouseDown={(e) => e.preventDefault()} // keep selection
          onClick={() => setOpen(true)}
          className="flex items-center gap-1 rounded-lg bg-slate-900 px-2.5 py-1 text-xs font-semibold text-white shadow-lift dark:bg-white dark:text-slate-900"
        >
          <Icon.Sparkles className="h-3.5 w-3.5" /> Ask AI
        </button>
      )}

      {open && (
        <div
          ref={popRef}
          style={{ position: 'fixed', top, left, width: POP_W, maxHeight, zIndex: 50 }}
          className="card flex flex-col p-3 shadow-lift"
        >
          <div className="mb-2 line-clamp-2 shrink-0 rounded-md bg-slate-50 px-2 py-1 text-xs italic text-slate-500 dark:bg-slate-900/50 dark:text-slate-400">
            “{sel.text}”
          </div>
          <input
            autoFocus
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && explain()}
            placeholder="Ask about it, or say how to change it…"
            className="field mb-2 shrink-0 text-sm"
          />
          <div className="flex shrink-0 gap-2">
            <button onClick={explain} disabled={loading} className="btn-ghost flex-1 py-1.5 text-xs">
              {loading && !answer ? '…' : 'Explain'}
            </button>
            {onEditSelection && (
              <button onClick={rewrite} disabled={loading} className="btn-primary flex-1 py-1.5 text-xs">
                Rewrite
              </button>
            )}
          </div>
          {answer && (
            <div
              ref={answerRef}
              className="scroll-area mt-2 min-h-0 flex-1 space-y-0.5 overflow-y-auto rounded-lg bg-slate-50 p-2 text-xs leading-relaxed text-slate-700 dark:bg-slate-900/50 dark:text-slate-200"
            >
              {renderMd(answer)}
            </div>
          )}
        </div>
      )}
    </>
  )
}
