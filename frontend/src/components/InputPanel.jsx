import { useState, useRef, useEffect } from 'react'
import { apiFetch } from '../lib/api'
import Icon from './Icons'

const MAX_CHARS = 300000

// Source panel, two states.
//
// Empty state  -> a centred stack: four action cards (Upload / Link / Paste /
//                 Record) over one "Learn anything" bar. The toolbar inside the
//                 bar stays collapsed until you type, so the resting state is
//                 quiet. Drop a file anywhere on the panel.
// Loaded state -> a read-only source viewer with title, size, and Edit /
//                 Replace. Pass `fill` to make it own the available height
//                 (the workspace's left pane).

const isUrl = (s) => /^https?:\/\/\S+$/i.test((s || '').trim())
const looksLikeUrl = (s) => /^(https?:\/\/)?[\w-]+(\.[\w-]+)+(\/\S*)?$/i.test((s || '').trim())

const CARD =
  'group relative flex flex-row items-center gap-2.5 rounded-3xl border border-neutral-200 bg-white p-4 ' +
  'text-left shadow-[0_4px_10px_rgba(0,0,0,0.04)] transition-colors duration-200 hover:border-neutral-300 ' +
  'hover:bg-neutral-50 disabled:pointer-events-none disabled:opacity-40 ' +
  'sm:h-[110px] sm:flex-col sm:items-start sm:justify-center sm:gap-y-2'

function SourceCard({ icon: Glyph, title, sub, badge, onClick, disabled }) {
  return (
    <button type="button" onClick={onClick} disabled={disabled} className={CARD}>
      {badge && (
        <span className="absolute right-2 top-2 hidden rounded-md border border-[#3CB371]/50 bg-[#3CB371]/10 px-2 py-0.5 text-xs font-medium text-[#3CB371] sm:block">
          {badge}
        </span>
      )}
      <Glyph className="h-5 w-5 shrink-0 text-neutral-500 transition-colors group-hover:text-neutral-900" />
      <span className="flex min-w-0 flex-col">
        <span className="text-sm font-medium text-neutral-700 transition-colors group-hover:text-neutral-900 sm:text-base">
          {title}
        </span>
        <span className="line-clamp-1 text-xs text-neutral-400 transition-colors group-hover:text-neutral-500 sm:text-sm">
          {sub}
        </span>
      </span>
    </button>
  )
}

export default function InputPanel({ inputText, setInputText, setPageSpans = () => {}, isStreaming, fill = false }) {
  const [dragging, setDragging] = useState(false)
  const [busy, setBusy] = useState(null)      // string label while extracting
  const [info, setInfo] = useState(null)      // { label } -> doubles as the source title
  const [error, setError] = useState(null)
  const [recording, setRecording] = useState(false)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')      // "Learn anything" bar value
  // The original document, shown as-is. The text the model reads is extracted
  // in the background and deliberately never rendered.
  const [preview, setPreview] = useState(null) // { kind: 'pdf', url, name }
  const [showText, setShowText] = useState(false)
  const fileInputRef = useRef(null)
  const barRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])
  const previewUrlRef = useRef(null)

  const hasSource = !!inputText.trim() || !!preview

  // If the source is cleared elsewhere (e.g. loading a history session), exit edit mode.
  useEffect(() => {
    if (!hasSource) setEditing(false)
  }, [hasSource])

  // Object URLs are leaked memory until revoked, so drop the last one on unmount.
  useEffect(() => () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current)
  }, [])

  const showPreview = (file, name) => {
    const url = URL.createObjectURL(file)
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current)
    previewUrlRef.current = url
    setPreview({ kind: 'pdf', url, name })
    setShowText(false)
    setEditing(false)
  }

  // Auto-grow the bar with its content, up to a cap.
  useEffect(() => {
    const el = barRef.current
    if (!el || hasSource) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }, [draft, hasSource])

  const done = (label) => { setInfo({ label }); setError(null); setBusy(null) }
  const fail = (message) => { setError(message); setInfo(null); setBusy(null) }

  // ----- shared extraction call --------------------------------------------
  const extractFile = async (endpoint, file, busyLabel, describe) => {
    setBusy(busyLabel); setError(null); setInfo(null)
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
      // Only /api/extract-pdf returns these; every other source clears them,
      // so a PDF's pages can never be attributed to the text that replaced it.
      setPageSpans(data.page_spans || [])
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
      // Put the document on screen straight away, then read it in the
      // background — the user looks at their PDF, not at a wall of
      // extracted characters.
      showPreview(file, name)
      setInputText('')
      return extractFile('/api/extract-pdf', file, `Reading ${name}…`, (d) => `${name} · ${d.pages} page${d.pages === 1 ? '' : 's'}`)
    }
    if (type.startsWith('image/')) {
      return extractFile('/api/extract-image', file, `Reading text from ${name || 'image'}…`, () => `${name || 'Pasted image'}`)
    }
    if (type.startsWith('audio/') || type.startsWith('video/') || /\.(mp3|wav|m4a|ogg|webm)$/i.test(name)) {
      return extractFile('/api/transcribe', file, `Transcribing ${name || 'audio'}…`, () => `${name || 'Recording'} · transcript`)
    }
    if (type.startsWith('text/') || /\.(txt|md|csv|json)$/i.test(name)) {
      setBusy(`Reading ${name}…`)
      file.text().then((t) => {
        setInputText(t.slice(0, MAX_CHARS))
        setPageSpans([])
        done(name)
      }).catch(() => fail(`Could not read ${name}.`))
      return
    }
    fail(`Unsupported file type: ${name || type || 'unknown'}. Try a PDF, image, audio, or text file.`)
  }

  // ----- URL fetch ----------------------------------------------------------
  const fetchUrl = async (raw) => {
    let target = (raw || '').trim()
    if (!target) return
    if (!/^https?:\/\//i.test(target)) target = 'https://' + target
    if (!isUrl(target)) return
    setBusy('Fetching page…'); setError(null); setInfo(null)
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
      setPageSpans([])
      setDraft('')
      done(data.title || target)
    } catch (err) {
      fail(err.message)
    }
  }

  // ----- "Learn anything" bar submit ----------------------------------------
  const submitDraft = () => {
    const t = draft.trim()
    if (!t || isStreaming || busy) return
    if (looksLikeUrl(t)) return fetchUrl(t)
    setInputText(t.slice(0, MAX_CHARS))
    setPageSpans([])
    setDraft('')
    done('Pasted text')
  }

  // ----- paste: screenshots / files come through the clipboard --------------
  const onPaste = (e) => {
    const file = Array.from(e.clipboardData?.files || [])[0]
    if (file) { e.preventDefault(); handleFile(file) }
  }

  // "Paste" card: read the clipboard directly
  const pasteFromClipboard = async () => {
    try {
      const t = await navigator.clipboard.readText()
      if (t && t.trim()) {
        if (looksLikeUrl(t)) return fetchUrl(t)
        setInputText(t.slice(0, MAX_CHARS))
        setPageSpans([])
        done('Pasted text')
      } else {
        barRef.current?.focus()
      }
    } catch {
      barRef.current?.focus() // permission denied -> let them paste manually
    }
  }

  // "Link" card: prime the bar with a scheme and let them type
  const startLink = () => {
    setDraft((d) => (d ? d : 'https://'))
    requestAnimationFrame(() => barRef.current?.focus())
  }

  // ----- drag & drop over the whole panel -----------------------------------
  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files?.[0])
  }

  // ----- audio recording ----------------------------------------------------
  const startRecording = async () => {
    setError(null); setInfo(null)
    try {
      const streamObj = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(streamObj)
      chunksRef.current = []
      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data) }
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

  // ----- replace the loaded source ------------------------------------------
  const replaceSource = () => {
    if (isStreaming) return
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current)
      previewUrlRef.current = null
    }
    setPreview(null)
    setShowText(false)
    setInputText('')
    setPageSpans([])
    setInfo(null)
    setError(null)
    setDraft('')
    setEditing(false)
  }

  // ----- shared bits --------------------------------------------------------
  const locked = isStreaming || !!busy
  const atLimit = inputText.length >= MAX_CHARS
  const words = hasSource ? inputText.trim().split(/\s+/).length : 0

  const DropOverlay = () =>
    dragging ? (
      <div className="pointer-events-none absolute inset-0 z-20 flex flex-col items-center justify-center gap-2 rounded-3xl border-2 border-dashed border-neutral-900 bg-white/90 backdrop-blur-sm">
        <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-neutral-900 text-white shadow-lift">
          <Icon.Paperclip className="h-6 w-6" />
        </span>
        <p className="text-sm font-semibold text-neutral-700">Drop your file here</p>
      </div>
    ) : null

  const dragProps = {
    onDragOver: (e) => { e.preventDefault(); setDragging(true) },
    onDragLeave: (e) => { if (!e.currentTarget.contains(e.relatedTarget)) setDragging(false) },
    onDrop,
  }

  const fileInput = (
    <input
      ref={fileInputRef}
      type="file"
      accept="application/pdf,image/*,audio/*,.txt,.md,.csv,.json,.mp3,.wav,.m4a,.ogg,.webm"
      className="hidden"
      onChange={(e) => { handleFile(e.target.files?.[0]); e.target.value = '' }}
    />
  )

  // ==========================================================================
  // LOADED STATE — source viewer (also the workspace's left pane)
  // ==========================================================================
  if (hasSource) {
    // ---- Phase 1: a document is just an attachment ------------------------
    // Before Generate, the entry page should stay a clean composer — so a PDF
    // reads as a chip (icon + name + status), the way attaching a file to a
    // chat message does. The document itself belongs in the workspace, where
    // there is room to actually read it.
    if (preview && !fill) {
      return (
        <div className="relative mx-auto w-full max-w-[672px]" {...dragProps}>
          <DropOverlay />
          <div className="flex items-center gap-3 rounded-3xl border border-neutral-200 bg-white px-4 py-3 shadow-[0_4px_10px_rgba(0,0,0,0.04)]">
            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-red-50 text-red-500">
              <Icon.FileText className="h-5 w-5" />
            </span>

            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium text-neutral-900" title={preview.name}>
                {preview.name}
              </p>
              <p className="mt-0.5 truncate text-xs text-neutral-400">
                {busy ? (
                  <span className="inline-flex items-center gap-1.5 text-neutral-500">
                    <span className="h-2.5 w-2.5 shrink-0 animate-spin rounded-full border-2 border-neutral-400 border-t-transparent" />
                    Reading it…
                  </span>
                ) : error ? (
                  <span className="text-red-600">{error}</span>
                ) : (
                  <>PDF · {words.toLocaleString()} words ready</>
                )}
              </p>
            </div>

            <button
              onClick={replaceSource}
              disabled={locked}
              aria-label="Remove this file"
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-neutral-400 transition-colors hover:bg-neutral-100 hover:text-neutral-900 disabled:pointer-events-none disabled:opacity-40"
            >
              <Icon.X className="h-4 w-4" />
            </button>
          </div>
          {fileInput}
        </div>
      )
    }

    const actionBtn =
      'inline-flex items-center gap-1.5 rounded-full border border-neutral-200 px-3 py-1.5 text-xs ' +
      'font-medium text-neutral-600 transition-colors hover:border-neutral-300 hover:bg-neutral-50 ' +
      'hover:text-neutral-900 disabled:pointer-events-none disabled:opacity-40'

    return (
      <div
        className={`relative mx-auto flex w-full max-w-[672px] flex-col ${fill ? 'h-full min-h-0' : ''}`}
        {...dragProps}
      >
        <DropOverlay />

        <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-3xl border border-neutral-200 bg-white shadow-[0_4px_10px_rgba(0,0,0,0.04)]">
          {/* header — what's loaded, how big, what you can do with it */}
          <div className="flex shrink-0 items-start gap-3 border-b border-neutral-200/70 px-5 py-4">
            <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-neutral-100 text-neutral-500">
              <Icon.FileText className="h-[18px] w-[18px]" />
            </span>

            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium text-neutral-900" title={preview?.name || info?.label || 'Your source'}>
                {preview?.name || info?.label || 'Your source'}
              </p>
              <p className="mt-0.5 text-xs text-neutral-400">
                {busy ? (
                  <span className="text-neutral-500">Reading it in the background…</span>
                ) : (
                  <>
                    {inputText.length.toLocaleString()} characters · {words.toLocaleString()} words
                    {atLimit && (
                      <span className="text-amber-600"> · trimmed to the {MAX_CHARS.toLocaleString()}-character limit</span>
                    )}
                  </>
                )}
              </p>
            </div>

            <div className="flex shrink-0 items-center gap-2">
              {preview ? (
                // No Edit for a rendered document — you can't type into a PDF.
                // Offer a look at what the model will actually read instead.
                <button
                  onClick={() => setShowText((v) => !v)}
                  className={actionBtn}
                  disabled={locked || !inputText}
                  title="See the text the model will read"
                >
                  <Icon.Type className="h-3.5 w-3.5" />
                  {showText ? 'Document' : 'Text'}
                </button>
              ) : editing ? (
                <button onClick={() => setEditing(false)} className={actionBtn} disabled={isStreaming}>
                  <Icon.Check className="h-3.5 w-3.5" />
                  Done
                </button>
              ) : (
                <button onClick={() => setEditing(true)} className={actionBtn} disabled={locked}>
                  <Icon.Pencil className="h-3.5 w-3.5" />
                  Edit
                </button>
              )}
              <button onClick={replaceSource} className={actionBtn} disabled={locked}>
                <Icon.Refresh className="h-3.5 w-3.5" />
                Replace
              </button>
            </div>
          </div>

          {/* body — the document itself when we have one, else the text */}
          {preview && !showText ? (
            <div className={`min-h-0 flex-1 bg-neutral-100 ${fill ? '' : 'h-[32rem]'}`}>
              <iframe
                src={`${preview.url}#view=FitH&navpanes=0`}
                title={preview.name}
                className="h-full w-full border-0"
              />
            </div>
          ) : editing ? (
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value.slice(0, MAX_CHARS))}
              onPaste={onPaste}
              autoFocus
              disabled={isStreaming}
              className={`scroll-area w-full flex-1 resize-none bg-transparent px-5 py-4 text-sm leading-relaxed text-neutral-800 focus:outline-none ${
                fill ? 'min-h-0' : 'min-h-[16rem]'
              }`}
            />
          ) : (
            <div className={`scroll-area flex-1 overflow-y-auto px-5 py-4 ${fill ? 'min-h-0' : 'max-h-[22rem]'}`}>
              <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-neutral-600">
                {inputText}
              </p>
            </div>
          )}

          {/* status — extraction of a dropped replacement, or its failure */}
          {(busy || error) && (
            <div className="shrink-0 border-t border-neutral-200/70 px-5 py-2.5">
              {busy ? (
                <span className="flex items-center gap-2 truncate text-xs text-neutral-500">
                  <span className="h-3 w-3 shrink-0 animate-spin rounded-full border-2 border-neutral-900 border-t-transparent" />
                  {busy}
                </span>
              ) : (
                <span className="block truncate text-xs text-red-600">{error}</span>
              )}
            </div>
          )}
        </div>

        {fileInput}
      </div>
    )
  }

  // ==========================================================================
  // EMPTY STATE — four cards over one bar
  // ==========================================================================
  return (
    <div className="relative mx-auto flex w-full max-w-[672px] flex-col" {...dragProps}>
      <DropOverlay />

      {/* four ways in */}
      <div className="grid w-full grid-cols-2 gap-3 sm:grid-cols-4">
        <SourceCard
          icon={Icon.Paperclip}
          title="Upload"
          sub="File, audio, video"
          badge="Popular"
          onClick={() => fileInputRef.current?.click()}
          disabled={locked}
        />
        <SourceCard
          icon={Icon.Link}
          title="Link"
          sub="YouTube, website"
          onClick={startLink}
          disabled={locked}
        />
        <SourceCard
          icon={Icon.Copy}
          title="Paste"
          sub="Copied text"
          onClick={pasteFromClipboard}
          disabled={locked}
        />
        <SourceCard
          icon={recording ? Icon.Stop : Icon.Mic}
          title={recording ? 'Stop' : 'Record'}
          sub={recording ? 'Recording…' : 'Record a lecture'}
          onClick={recording ? stopRecording : startRecording}
          disabled={isStreaming || (!recording && !!busy)}
        />
      </div>

      {/* the one bar */}
      <div className="relative mt-3 rounded-3xl border border-neutral-200 bg-white pb-1 pl-3.5 pr-2 pt-1 shadow-[0_4px_10px_rgba(0,0,0,0.04)] transition-colors duration-200 focus-within:border-neutral-400 hover:border-neutral-300">
        <textarea
          ref={barRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onPaste={onPaste}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submitDraft()
            }
          }}
          disabled={locked}
          rows={1}
          placeholder="Learn anything"
          className="scroll-area max-h-40 w-full resize-none overflow-y-auto bg-transparent py-1.5 pr-10 text-base leading-6 text-neutral-900 placeholder:text-neutral-400 focus:outline-none disabled:opacity-60"
        />

        {/* toolbar — stays out of the way until there's something to send */}
        <div
          className={`overflow-hidden transition-all duration-300 ${
            draft.trim() ? 'max-h-12 opacity-100' : 'max-h-0 opacity-0'
          }`}
        >
          <div className="flex items-center gap-1 pb-1 pt-1">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={locked}
              className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1.5 text-sm font-medium text-neutral-500 transition-colors hover:bg-neutral-100 hover:text-neutral-900 disabled:pointer-events-none disabled:opacity-40"
            >
              <Icon.Paperclip className="h-4 w-4" />
              Attach
            </button>
          </div>
        </div>

        <button
          type="button"
          onClick={submitDraft}
          disabled={!draft.trim() || locked}
          aria-label="Use this as your source"
          className="absolute bottom-[0.3rem] right-2 flex h-[34px] w-[34px] items-center justify-center rounded-full bg-neutral-900 text-white transition-opacity hover:opacity-85 disabled:pointer-events-none disabled:opacity-25"
        >
          <Icon.Send className="h-[18px] w-[18px]" />
        </button>
      </div>

      {/* status */}
      <div className="min-h-[1.5rem] pt-2 text-center text-xs">
        {busy && (
          <span className="inline-flex items-center gap-2 text-neutral-500">
            <span className="h-3 w-3 shrink-0 animate-spin rounded-full border-2 border-neutral-900 border-t-transparent" />
            {busy}
          </span>
        )}
        {error && !busy && <span className="text-red-600">{error}</span>}
        {!busy && !error && (
          <span className="text-neutral-400">Or drop a file anywhere on this page</span>
        )}
      </div>

      {fileInput}
    </div>
  )
}
