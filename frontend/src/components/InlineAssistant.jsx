import { useState, useEffect, useRef } from 'react'
import Icon from './Icons'

/**
 * Inline AI assistant for the notes view. Select text in the notes container and
 * a small "Ask AI" popover appears: Explain (streams an answer) or Rewrite
 * (edits that passage in place via onEditSelection).
 */
export default function InlineAssistant({ containerRef, notes, onEditSelection }) {
  const [sel, setSel] = useState(null) // { text, top, left }
  const [open, setOpen] = useState(false)
  const [instruction, setInstruction] = useState('')
  const [answer, setAnswer] = useState('')
  const [loading, setLoading] = useState(false)
  const popRef = useRef(null)

  // Detect a text selection inside the notes container.
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const onUp = () => {
      const s = window.getSelection()
      const text = s && s.toString().trim()
      if (text && text.length > 1 && s.anchorNode && el.contains(s.anchorNode)) {
        const rect = s.getRangeAt(0).getBoundingClientRect()
        setSel({ text, top: rect.top, left: rect.left + rect.width / 2 })
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
    const onDown = (e) => {
      if (popRef.current && !popRef.current.contains(e.target)) close()
    }
    const onKey = (e) => e.key === 'Escape' && close()
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

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

  return (
    <>
      {!open && (
        <button
          style={{
            position: 'fixed',
            top: Math.max(8, sel.top - 8),
            left: sel.left,
            transform: 'translate(-50%, -100%)',
            zIndex: 50,
          }}
          onMouseDown={(e) => e.preventDefault()} // keep the text selection
          onClick={() => setOpen(true)}
          className="flex items-center gap-1 rounded-lg bg-slate-900 px-2.5 py-1 text-xs font-semibold text-white shadow-lift dark:bg-white dark:text-slate-900"
        >
          <Icon.Sparkles className="h-3.5 w-3.5" /> Ask AI
        </button>
      )}

      {open && (
        <div
          ref={popRef}
          style={{
            position: 'fixed',
            top: Math.min(window.innerHeight - 360, sel.top + 14),
            left: Math.min(window.innerWidth - 336, Math.max(12, sel.left - 160)),
            width: 324,
            zIndex: 50,
          }}
          className="card p-3 shadow-lift"
        >
          <div className="mb-2 line-clamp-2 rounded-md bg-slate-50 px-2 py-1 text-xs italic text-slate-500 dark:bg-slate-900/50 dark:text-slate-400">
            “{sel.text}”
          </div>
          <input
            autoFocus
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && explain()}
            placeholder="Ask about it, or say how to change it…"
            className="field mb-2 text-sm"
          />
          <div className="flex gap-2">
            <button onClick={explain} disabled={loading} className="btn-ghost flex-1 py-1.5 text-xs">
              {loading ? '…' : 'Explain'}
            </button>
            {onEditSelection && (
              <button onClick={rewrite} disabled={loading} className="btn-primary flex-1 py-1.5 text-xs">
                Rewrite
              </button>
            )}
          </div>
          {answer && (
            <div className="scroll-area mt-2 max-h-48 overflow-y-auto whitespace-pre-wrap rounded-lg bg-slate-50 p-2 text-xs leading-relaxed text-slate-700 dark:bg-slate-900/50 dark:text-slate-200">
              {answer}
            </div>
          )}
        </div>
      )}
    </>
  )
}
