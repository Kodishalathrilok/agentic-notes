import { useState, useRef, useEffect } from 'react'
import { apiFetch } from '../lib/api'
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
      <div className="flex h-40 items-center justify-center rounded-2xl border border-dashed border-espresso-900/15 px-4 text-center text-sm text-espresso-400">
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
      const res = await apiFetch('/api/chat', {
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
    <div className="flex h-[420px] flex-col overflow-hidden rounded-2xl border border-espresso-900/[0.06] bg-white shadow-neu-inset">
      <div ref={scrollRef} className="scroll-area flex-1 space-y-3 overflow-y-auto p-3">
        {messages.length === 0 && (
          <div className="space-y-2 text-sm text-espresso-400">
            <p>Ask anything about your notes. For example:</p>
            <div className="flex flex-wrap gap-2">
              {['Explain this more simply', 'Give me a real-world example', 'What should I memorise?'].map(
                (s) => (
                  <button
                    key={s}
                    onClick={() => setInput(s)}
 className="rounded-full border border-espresso-900/15 bg-white px-3 py-1 text-xs text-espresso-700 shadow-neu-sm transition-colors hover:bg-espresso-900/5"
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
                  ? 'bg-espresso-900 text-cream'
                  : 'bg-espresso-900/5 text-espresso-800'
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

      <div className="flex gap-2 border-t border-espresso-900/[0.06] bg-white/70 p-2.5">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && send()}
          placeholder="Ask anything…"
          disabled={busy}
 className="field flex-1"
        />
        <button onClick={send} disabled={busy || !input.trim()} className="btn-primary shrink-0 px-4">
          <Icon.Send className="h-4 w-4" />
          {busy ? '…' : 'Send'}
        </button>
      </div>
    </div>
  )
}