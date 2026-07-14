import { useState, useRef } from 'react'
import { apiFetch } from '../lib/api'
import Icon from './Icons'

const MAX_CHARS = 300000

// One smart input: paste text, paste/type a link, drop or attach ANY file
// (PDF / photo / audio / plain text), paste a screenshot, or record audio.
// The type is detected automatically — no tabs, no mode switching.

const isUrl = (s) => /^https?:\/\/\S+$/i.test((s || '').trim())

export default function InputPanel({ inputText, setInputText, isStreaming }) {
  const [dragging, setDragging] = useState(false)
  const [busy, setBusy] = useState(null) // string label while extracting
  const [info, setInfo] = useState(null) // { label } success chip
  const [error, setError] = useState(null)
  const [recording, setRecording] = useState(false)
  const fileInputRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])

  const done = (label) => {
    setInfo({ label })
    setError(null)
    setBusy(null)
  }
  const fail = (message) => {
    setError(message)
    setInfo(null)
    setBusy(null)
  }

  // ----- shared extraction call --------------------------------------------
  const extractFile = async (endpoint, file, busyLabel, describe) => {
    setBusy(busyLabel)
    setError(null)
    setInfo(null)
    try {
      const form = new FormData()
      form.append('file', file, file.name || 'upload')
      const res = await apiFetch(endpoint, { method: 'POST', body: form })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Extraction failed (${res.status})`)
      }
      const data = await res.json()
      setInputText((data.text || '').slice(0, MAX_CHARS))
      done(describe(data))
    } catch (err) {
      fail(err.message)
    }
  }

  // ----- auto-detect any file ----------------------------------------------
  const handleFile = (file) => {
    if (!file || isStreaming) return
    const name = file.name || ''
    const type = file.type || ''

    if (type === 'application/pdf' || /\.pdf$/i.test(name)) {
      return extractFile('/api/extract-pdf', file, `Extracting ${name}…`, (d) => `${name} — ${d.pages} page${d.pages === 1 ? '' : 's'} extracted`)
    }
    if (type.startsWith('image/')) {
      return extractFile('/api/extract-image', file, `Reading text from ${name || 'image'}…`, (d) => `${name || 'Pasted image'} — ${(d.chars || 0).toLocaleString()} chars extracted`)
    }
    if (type.startsWith('audio/') || type.startsWith('video/') || /\.(mp3|wav|m4a|ogg|webm)$/i.test(name)) {
      return extractFile('/api/transcribe', file, `Transcribing ${name || 'audio'}…`, () => `${name || 'Recording'} — transcribed`)
    }
    if (type.startsWith('text/') || /\.(txt|md|csv|json)$/i.test(name)) {
      setBusy(`Reading ${name}…`)
      file.text().then((t) => {
        setInputText(t.slice(0, MAX_CHARS))
        done(`${name} — ${Math.min(t.length, MAX_CHARS).toLocaleString()} chars loaded`)
      }).catch(() => fail(`Could not read ${name}.`))
      return
    }
    fail(`Unsupported file type: ${name || type || 'unknown'}. Try a PDF, image, audio, or text file.`)
  }

  // ----- URL fetch ----------------------------------------------------------
  const fetchUrl = async (raw) => {
    const target = (raw || inputText).trim()
    if (!isUrl(target)) return
    setBusy('Fetching page…')
    setError(null)
    setInfo(null)
    try {
      const res = await apiFetch('/api/extract-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: target }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Fetch failed (${res.status})`)
      }
      const data = await res.json()
      setInputText((data.text || '').slice(0, MAX_CHARS))
      done(`${data.title || target} — ${(data.chars || 0).toLocaleString()} chars extracted`)
    } catch (err) {
      fail(err.message)
    }
  }

  // ----- paste: files (screenshots) come through the clipboard --------------
  const onPaste = (e) => {
    const file = Array.from(e.clipboardData?.files || [])[0]
    if (file) {
      e.preventDefault()
      handleFile(file)
    }
    // Pasted URLs are left in the textarea; the Fetch chip appears for them.
  }

  // ----- drag & drop over the whole card ------------------------------------
  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files?.[0])
  }

  // ----- audio recording -----------------------------------------------------
  const startRecording = async () => {
    setError(null)
    setInfo(null)
    try {
      const streamObj = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(streamObj)
      chunksRef.current = []
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }
      recorder.onstop = () => {
        streamObj.getTracks().forEach((t) => t.stop())
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        handleFile(new File([blob], 'recording.webm', { type: 'audio/webm' }))
      }
      mediaRecorderRef.current = recorder
      recorder.start()
      setRecording(true)
    } catch {
      fail('Microphone access denied or unavailable.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop()
      setRecording(false)
    }
  }

  const showFetchChip = isUrl(inputText) && !busy

  // ----- render --------------------------------------------------------------
  const nearLimit = inputText.length > MAX_CHARS * 0.9
  const iconBtn =
    'flex h-8 w-8 items-center justify-center rounded-full text-espresso-500 transition-colors duration-150 ' +
    'hover:bg-espresso-900/5 hover:text-espresso-900 disabled:pointer-events-none disabled:opacity-40'

  return (
    <div
      className="glass-card group/panel relative overflow-hidden rounded-3xl p-6 transition-shadow duration-300 hover:shadow-lift sm:p-7"
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget)) setDragging(false)
      }}
      onDrop={onDrop}
    >
      {/* soft corner glow */}
      <div className="pointer-events-none absolute -right-16 -top-16 h-40 w-40 rounded-full bg-brand-500/10 blur-2xl transition-opacity duration-500 group-hover/panel:opacity-100" />

      <div className="mb-3 flex items-baseline justify-between gap-3">
        <h2 className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.2em] text-espresso-500">
          <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-gradient-to-br from-brand-500 to-rose-400 text-white shadow-soft">
            <Icon.Book className="h-3.5 w-3.5" />
          </span>
          Source
        </h2>
        <p className="hidden text-xs text-espresso-400 sm:block">
          Text, link, PDF, image, or audio — drop a file anywhere on this card
        </p>
      </div>

      {/* One composer: textarea + toolbar in a single container */}
      <div
        className={`relative rounded-2xl border transition-all duration-200 ${dragging
            ? 'border-2 border-dashed border-brand-500 bg-brand-500/5'
            : 'border-espresso-900/15 bg-white/60 focus-within:border-espresso-900/40 focus-within:bg-white/80 focus-within:shadow-[0_0_0_3px_rgb(0_0_0_/_0.06)]'
          }`}
      >
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value.slice(0, MAX_CHARS))}
          onPaste={onPaste}
          disabled={isStreaming || !!busy}
          rows={9}
          placeholder={'Paste your material here — lecture notes, an article, or a link.'}
          className="w-full resize-none rounded-t-2xl bg-transparent px-4 pb-2 pt-4 leading-relaxed text-espresso-800 placeholder:text-espresso-400/70 focus:outline-none"
        />

        {/* toolbar inside the composer */}
        <div className="flex items-center gap-1 border-t border-espresso-900/[0.07] px-2 py-1.5">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isStreaming || !!busy}
            className={iconBtn}
            title="Attach a PDF, image, audio, or text file"
            aria-label="Attach file"
          >
            <Icon.Paperclip className="h-4 w-4" />
          </button>

          {!recording ? (
            <button
              onClick={startRecording}
              disabled={isStreaming || !!busy}
              className={iconBtn}
              title="Record audio and transcribe it"
              aria-label="Record audio"
            >
              <Icon.Mic className="h-4 w-4" />
            </button>
          ) : (
            <button
              onClick={stopRecording}
              className="flex items-center gap-1.5 rounded-full bg-red-500 px-3 py-1 text-xs font-semibold text-white shadow-soft hover:bg-red-600"
            >
              <span className="h-2 w-2 animate-ping rounded-full bg-white" />
              Stop
            </button>
          )}

          {showFetchChip && (
            <button
              onClick={() => fetchUrl()}
              className="ml-1 flex items-center gap-1.5 rounded-full bg-gradient-to-r from-espresso-800 to-espresso-900 px-3 py-1 text-xs font-semibold text-cream shadow-soft transition-transform hover:scale-[1.03]"
              title="Extract the readable text from this link"
            >
              <Icon.Link className="h-3.5 w-3.5" />
              Fetch this link
            </button>
          )}

          {/* status inline, one line */}
          <div className="min-w-0 flex-1 px-2">
            {busy && (
              <span className="flex items-center gap-2 truncate text-xs text-espresso-500">
                <span className="h-3 w-3 shrink-0 animate-spin rounded-full border-2 border-brand-500 border-t-transparent" />
                {busy}
              </span>
            )}
            {info && !busy && (
              <span className="block truncate text-xs font-medium text-green-700">✓ {info.label}</span>
            )}
            {error && !busy && <span className="block truncate text-xs text-red-600">{error}</span>}
          </div>

          {inputText.length > 0 && (
            <span
              className={`shrink-0 text-[11px] tabular-nums ${nearLimit ? 'font-semibold text-red-500' : 'text-espresso-400'
                }`}
              title={`${inputText.length.toLocaleString()} of ${MAX_CHARS.toLocaleString()} characters`}
            >
              {inputText.length.toLocaleString()}
              {nearLimit && ` / ${MAX_CHARS.toLocaleString()}`}
            </span>
          )}
        </div>

        {dragging && (
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-2 rounded-2xl bg-latte-100/85 backdrop-blur-sm">
            <span className="flex h-12 w-12 animate-bounce items-center justify-center rounded-2xl bg-gradient-to-br from-brand-500 to-rose-400 text-white shadow-lift">
              <Icon.Paperclip className="h-6 w-6" />
            </span>
            <p className="text-sm font-semibold text-espresso-700">Drop it — I'll figure out what it is ☕</p>
          </div>
        )}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="application/pdf,image/*,audio/*,.txt,.md,.csv,.json,.mp3,.wav,.m4a,.ogg,.webm"
        className="hidden"
        onChange={(e) => {
          handleFile(e.target.files?.[0])
          e.target.value = ''
        }}
      />
    </div>
  )
}