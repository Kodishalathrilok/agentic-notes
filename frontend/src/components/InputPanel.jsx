import { useState, useRef } from 'react'

const TABS = [
  { id: 'text', label: 'Text' },
  { id: 'pdf', label: 'PDF' },
  { id: 'image', label: 'Image' },
  { id: 'url', label: 'URL' },
  { id: 'audio', label: 'Audio' },
]

const MAX_CHARS = 300000

export default function InputPanel({ inputText, setInputText, isStreaming }) {
  const [tab, setTab] = useState('text')

  // PDF
  const [pdfInfo, setPdfInfo] = useState(null) // { name, pages }
  const [pdfLoading, setPdfLoading] = useState(false)
  const [pdfError, setPdfError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const fileInputRef = useRef(null)

  // Image (OCR)
  const [imgLoading, setImgLoading] = useState(false)
  const [imgError, setImgError] = useState(null)
  const [imgInfo, setImgInfo] = useState(null)
  const [imgDragging, setImgDragging] = useState(false)
  const imgInputRef = useRef(null)

  const handleImageFile = async (file) => {
    if (!file) return
    setImgError(null)
    setImgInfo(null)
    setImgLoading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch('/api/extract-image', { method: 'POST', body: form })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Extraction failed (${res.status})`)
      }
      const data = await res.json()
      setInputText(data.text.slice(0, MAX_CHARS))
      setImgInfo({ name: file.name, chars: data.chars })
    } catch (err) {
      setImgError(err.message)
    } finally {
      setImgLoading(false)
    }
  }

  // URL
  const [url, setUrl] = useState('')
  const [urlLoading, setUrlLoading] = useState(false)
  const [urlError, setUrlError] = useState(null)
  const [urlInfo, setUrlInfo] = useState(null)

  // Audio
  const [recording, setRecording] = useState(false)
  const [audioStatus, setAudioStatus] = useState(null)
  const [audioError, setAudioError] = useState(null)
  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])

  // ----- URL handling ------------------------------------------------------
  const fetchUrl = async () => {
    if (!url.trim()) return
    setUrlError(null)
    setUrlInfo(null)
    setUrlLoading(true)
    try {
      const res = await fetch('/api/extract-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url.trim() }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Fetch failed (${res.status})`)
      }
      const data = await res.json()
      setInputText(data.text.slice(0, MAX_CHARS))
      setUrlInfo({ title: data.title, chars: data.chars })
    } catch (err) {
      setUrlError(err.message)
    } finally {
      setUrlLoading(false)
    }
  }

  // ----- PDF handling ------------------------------------------------------
  const handlePdfFile = async (file) => {
    if (!file) return
    setPdfError(null)
    setPdfInfo(null)
    setPdfLoading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch('/api/extract-pdf', { method: 'POST', body: form })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Extraction failed (${res.status})`)
      }
      const data = await res.json()
      setInputText(data.text.slice(0, MAX_CHARS))
      setPdfInfo({ name: file.name, pages: data.pages })
    } catch (err) {
      setPdfError(err.message)
    } finally {
      setPdfLoading(false)
    }
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) handlePdfFile(file)
  }

  // ----- Audio handling ----------------------------------------------------
  const startRecording = async () => {
    setAudioError(null)
    setAudioStatus(null)
    try {
      const streamObj = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(streamObj)
      chunksRef.current = []
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }
      recorder.onstop = async () => {
        streamObj.getTracks().forEach((t) => t.stop())
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        await transcribe(blob)
      }
      mediaRecorderRef.current = recorder
      recorder.start()
      setRecording(true)
    } catch (err) {
      setAudioError('Microphone access denied or unavailable.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop()
      setRecording(false)
      setAudioStatus('Transcribing…')
    }
  }

  const transcribe = async (blob) => {
    setAudioError(null)
    setAudioStatus('Transcribing…')
    try {
      const form = new FormData()
      form.append('file', blob, 'recording.webm')
      const res = await fetch('/api/transcribe', { method: 'POST', body: form })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Transcription failed (${res.status})`)
      }
      const data = await res.json()
      setInputText(data.text.slice(0, MAX_CHARS))
      setAudioStatus('Transcription complete ✓')
    } catch (err) {
      setAudioError(err.message)
      setAudioStatus(null)
    }
  }

  // ----- Render ------------------------------------------------------------
  return (
    <div className="card p-6 sm:p-7">
      <h2 className="mb-5 text-xs font-bold uppercase tracking-[0.2em] text-espresso-500">
        Source
      </h2>

      {/* Tab switcher — pills */}
      <div className="scroll-area mb-5 flex gap-1 overflow-x-auto rounded-full border border-espresso-900/10 bg-espresso-900/[0.06] p-1">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
 className={`whitespace-nowrap rounded-full px-4 py-1.5 text-sm font-semibold transition-all duration-200 ${
              tab === t.id
                ? 'bg-espresso-900 text-cream shadow-soft'
                : 'text-espresso-600 hover:text-espresso-900'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Text */}
      {tab === 'text' && (
        <div>
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value.slice(0, MAX_CHARS))}
            disabled={isStreaming}
            rows={10}
            placeholder="Paste lecture notes, article, or any text..."
 className="field resize-y p-3 leading-relaxed"
          />
          <div className="mt-1 text-right text-xs text-slate-400">
            {inputText.length.toLocaleString()} / {MAX_CHARS.toLocaleString()}
          </div>
        </div>
      )}

      {/* PDF */}
      {tab === 'pdf' && (
        <div>
          <div
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
 className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 text-center transition ${
              dragging
                ? 'border-brand-500 bg-brand-50 dark:bg-brand-900/20'
                : 'border-slate-300 hover:border-brand-400 dark:border-slate-600'
            }`}
          >
            <svg className="mb-2 h-10 w-10 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.9A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 13l3-3m0 0l3 3m-3-3v9" />
            </svg>
            <p className="text-sm text-slate-600 dark:text-slate-300">
              {pdfLoading ? 'Extracting text…' : 'Drag & drop a PDF, or click to browse'}
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept="application/pdf"
 className="hidden"
              onChange={(e) => handlePdfFile(e.target.files?.[0])}
            />
          </div>

          {pdfInfo && (
            <div className="mt-3 rounded-lg bg-green-50 px-3 py-2 text-sm text-green-700 dark:bg-green-900/20 dark:text-green-300">
              ✓ <span className="font-medium">{pdfInfo.name}</span> — {pdfInfo.pages} page
              {pdfInfo.pages === 1 ? '' : 's'} extracted
            </div>
          )}
          {pdfError && (
            <div className="mt-3 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-600 dark:bg-red-900/20 dark:text-red-300">
              {pdfError}
            </div>
          )}
        </div>
      )}

      {/* Image (OCR via Gemini vision) */}
      {tab === 'image' && (
        <div>
          <div
            onDragOver={(e) => {
              e.preventDefault()
              setImgDragging(true)
            }}
            onDragLeave={() => setImgDragging(false)}
            onDrop={(e) => {
              e.preventDefault()
              setImgDragging(false)
              handleImageFile(e.dataTransfer.files?.[0])
            }}
            onClick={() => imgInputRef.current?.click()}
 className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 text-center transition ${
              imgDragging
                ? 'border-brand-500 bg-brand-50 dark:bg-brand-900/20'
                : 'border-slate-300 hover:border-brand-400 dark:border-slate-600'
            }`}
          >
            <svg className="mb-2 h-10 w-10 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 5a2 2 0 012-2h14a2 2 0 012 2v14a2 2 0 01-2 2H5a2 2 0 01-2-2V5z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 16l5-5 4 4 3-3 6 6" />
              <circle cx="8.5" cy="8.5" r="1.5" fill="currentColor" />
            </svg>
            <p className="text-sm text-slate-600 dark:text-slate-300">
              {imgLoading ? 'Reading text from image…' : 'Drop a photo of notes/textbook, or click to upload'}
            </p>
            <input
              ref={imgInputRef}
              type="file"
              accept="image/*"
 className="hidden"
              onChange={(e) => handleImageFile(e.target.files?.[0])}
            />
          </div>

          {imgInfo && (
            <div className="mt-3 rounded-lg bg-green-50 px-3 py-2 text-sm text-green-700 dark:bg-green-900/20 dark:text-green-300">
              ✓ <span className="font-medium">{imgInfo.name}</span> — {imgInfo.chars.toLocaleString()} chars extracted
            </div>
          )}
          {imgError && (
            <div className="mt-3 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-600 dark:bg-red-900/20 dark:text-red-300">
              {imgError}
            </div>
          )}
          <p className="mt-2 text-xs text-slate-400">
            Snap a textbook page, slide, or handwritten notes — Gemini vision extracts the text.
          </p>
        </div>
      )}

      {/* URL */}
      {tab === 'url' && (
        <div>
          <div className="flex gap-2">
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && fetchUrl()}
              placeholder="https://example.com/article or YouTube link"
 className="field flex-1"
            />
            <button onClick={fetchUrl} disabled={urlLoading || !url.trim()} className="btn-primary shrink-0">
              {urlLoading ? 'Fetching…' : 'Fetch'}
            </button>
          </div>
          {urlInfo && (
            <div className="mt-3 rounded-lg bg-green-50 px-3 py-2 text-sm text-green-700 dark:bg-green-900/20 dark:text-green-300">
              ✓ <span className="font-medium">{urlInfo.title}</span> — {urlInfo.chars.toLocaleString()} chars extracted
            </div>
          )}
          {urlError && (
            <div className="mt-3 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-600 dark:bg-red-900/20 dark:text-red-300">
              {urlError}
            </div>
          )}
          <p className="mt-2 text-xs text-slate-400">
            Paste a link to an article or web page and we'll extract the readable text.
          </p>
        </div>
      )}

      {/* Audio */}
      {tab === 'audio' && (
        <div className="flex flex-col items-center gap-4 py-4">
          <div className="flex items-center gap-3">
            {!recording ? (
              <button
                onClick={startRecording}
 className="flex items-center gap-2 rounded-full bg-red-500 px-5 py-2.5 text-sm font-medium text-white shadow hover:bg-red-600"
              >
                <span className="h-3 w-3 rounded-full bg-white" />
                Record
              </button>
            ) : (
              <button
                onClick={stopRecording}
 className="flex items-center gap-2 rounded-full bg-slate-700 px-5 py-2.5 text-sm font-medium text-white shadow hover:bg-slate-800"
              >
                <span className="h-3 w-3 rounded-sm bg-white" />
                Stop
              </button>
            )}
            {recording && (
              <span className="flex items-center gap-2 text-sm text-red-500">
                <span className="h-3 w-3 animate-ping rounded-full bg-red-500" />
                Recording…
              </span>
            )}
          </div>

          {audioStatus && (
            <div className="text-sm text-slate-600 dark:text-slate-300">{audioStatus}</div>
          )}
          {audioError && (
            <div className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-600 dark:bg-red-900/20 dark:text-red-300">
              {audioError}
            </div>
          )}
          <p className="text-center text-xs text-slate-400">
            Audio transcription uses Groq Whisper and requires a GROQ_API_KEY.
          </p>
        </div>
      )}
    </div>
  )
}
