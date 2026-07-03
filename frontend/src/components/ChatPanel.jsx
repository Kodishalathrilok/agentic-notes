import { useState, useRef, useEffect } from 'react'
import Icon from './Icons'

// Inline **bold** rendering.
function renderInline(text, key) {
  return text.split(/(\*\*.+?\*\*)/g).map((part, i) => {
    const m = part.match(/^\*\*(.+?)\*\*$/)
    return m ? (
      <strong key={`${key}-${i}`} className="font-semibold">
        {m[1]}
      </strong>
    ) : (
      <span key={`${key}-${i}`}>{part}</span>
    )
  })
}

// Lightweight markdown for chat replies: bold, bullets, numbered, paragraphs.
function renderMarkdown(text) {
  return text.split('\n').map((line, idx) => {
    const t = line.trim()
    if (!t) return <div key={idx} className="h-1.5" />
    const bullet = t.match(/^[*\-+]\s+(.*)$/)
    if (bullet) {
      return (
        <div key={idx} className="flex gap-2">
          <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-current opacity-50" />
          <span>{renderInline(bullet[1], idx)}</span>
        </div>
      )
    }
    if (/^\d+[.)]\s+/.test(t)) {
      return <div key={idx}>{renderInline(t, idx)}</div>
    }
    return <p key={idx}>{renderInline(t, idx)}</p>
  })
}

export default function ChatPanel({ notes, model }) {
  const [messages, setMessages] = useState([]) // { role: 'user'|'assistant', content }
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const scrollRef = useRef(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  if (!notes) {
    return (
      <div className="flex h-48 items-center justify-center rounded-xl border border-dashed border-slate-300 px-4 text-center text-sm text-slate-400 dark:border-slate-600">
        Generate notes first, then ask questions about them here.
      </div>
    )
  }

  const send = async () => {
    const question = input.trim()
    if (!question || busy) return
    setInput('')

    const history = messages.map((m) => ({ role: m.role, content: m.content }))
    setMessages((m) => [...m, { role: 'user', content: question }, { role: 'assistant', content: '' }])
    setBusy(true)

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes, question, history, model }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Chat failed (${res.status})`)
      }
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value, { stream: true })
        setMessages((m) => {
          const copy = [...m]
          copy[copy.length - 1] = {
            role: 'assistant',
            content: copy[copy.length - 1].content + chunk,
          }
          return copy
        })
      }
    } catch (err) {
      setMessages((m) => {
        const copy = [...m]
        copy[copy.length - 1] = { role: 'assistant', content: `⚠️ ${err.message}` }
        return copy
      })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card flex h-[60vh] flex-col">
      <div ref={scrollRef} className="scroll-area flex-1 space-y-3 overflow-y-auto p-4">
        {messages.length === 0 && (
          <div className="space-y-2 text-sm text-slate-400">
            <p>Ask anything about your notes. For example:</p>
            <div className="flex flex-wrap gap-2">
              {['Explain this more simply', 'Give me a real-world example', 'What should I memorise?'].map(
                (s) => (
                  <button
                    key={s}
                    onClick={() => setInput(s)}
 className="rounded-full border border-slate-300 px-3 py-1 text-xs hover:bg-slate-100 dark:border-slate-600 dark:hover:bg-slate-700"
                  >
                    {s}
                  </button>
                )
              )}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
 className={`max-w-[85%] space-y-0.5 rounded-2xl px-4 py-2 text-sm leading-relaxed ${
                m.role === 'user'
                  ? 'bg-brand-600 text-white'
                  : 'bg-slate-100 text-slate-800 dark:bg-slate-700 dark:text-slate-100'
              }`}
            >
              {m.role === 'assistant' ? (
                m.content ? (
                  renderMarkdown(m.content)
                ) : (
                  <span>{busy ? '…' : ''}</span>
                )
              ) : (
                <span className="whitespace-pre-wrap">{m.content}</span>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="flex gap-2 border-t border-slate-100 p-3 dark:border-slate-700">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && send()}
          placeholder="Ask about your notes…"
          disabled={busy}
 className="field flex-1"
        />
        <button onClick={send} disabled={busy || !input.trim()} className="btn-primary shrink-0">
          <Icon.Send className="h-4 w-4" />
          {busy ? '…' : 'Send'}
        </button>
      </div>
    </div>
  )
}
