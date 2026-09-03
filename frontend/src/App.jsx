import { useState, useEffect, useCallback, useRef, useMemo, lazy, Suspense } from 'react'
import useStream from './hooks/useStream'
import InputPanel from './components/InputPanel'
import ControlPanel from './components/ControlPanel'
import PipelineStrip, { LABELS as STEP_LABELS } from './components/PipelineStrip'
import PipelineInsights from './components/PipelineInsights'
import NotesOutput from './components/NotesOutput'
import Icon from './components/Icons'
import Landing from './components/Landing'
import ErrorBoundary from './components/ErrorBoundary'

// Split out of the main bundle: none of these are on the first-paint path, and
// most visitors never open them at all. Landing, InputPanel and NotesOutput stay
// eager because they ARE the first paint.
const QuizPanel = lazy(() => import('./components/QuizPanel'))
const FlashcardPanel = lazy(() => import('./components/FlashcardPanel'))
const HistoryPanel = lazy(() => import('./components/HistoryPanel'))
const MenuOverlay = lazy(() => import('./components/MenuOverlay'))
const EvalDashboard = lazy(() => import('./components/EvalDashboard'))
const AuthModal = lazy(() => import('./components/AuthModal'))
const SharedNote = lazy(() => import('./components/SharedNote'))

// Quiet placeholder. A spinner here would flash and vanish on a warm cache,
// which reads as a glitch rather than as loading.
const PanelFallback = () => (
  <div className="h-40 animate-pulse rounded-3xl border border-neutral-200 bg-neutral-50" />
)
import useHistory from './hooks/useHistory'
import { supabase, supabaseEnabled } from './lib/supabase'
import { apiFetch } from './lib/api'
import { recordRun, estimateTotalSeconds, formatClock, formatRemaining } from './lib/timings'
import {
  INPUT_KEY,
  SETTINGS_KEY,
  readJSON,
  writeJSON,
  readText,
  writeText,
  clearAccountContent,
  purgeLegacyKeys,
} from './lib/storage'

// The account whose data is on screen. Signed-out use has its own namespace.
const GUEST = 'guest'
const accountOf = (user) => user?.id || GUEST

// Retire the pre-namespacing keys before anything reads storage, so a document
// left behind by the previous build can't show up in a fresh sign-in.
purgeLegacyKeys()

const INITIAL_STEPS = [
  { step: 'plan', message: '', status: 'pending' },
  { step: 'write', message: '', status: 'pending' },
  { step: 'critique', message: '', status: 'pending' },
  { step: 'revise', message: '', status: 'pending' },
  { step: 'quiz', message: '', status: 'pending' },
  { step: 'flashcards', message: '', status: 'pending' },
]

const TABS = [
  { id: 'notes', label: 'Notes' },
  { id: 'quiz', label: 'Quiz' },
  { id: 'flashcards', label: 'Flashcards' },
  { id: 'history', label: 'History' },
]

const DEFAULT_SETTINGS = {
  mode: 'exam',
  tone: 'academic',
  length: 'medium',
  format: 'bullet',
  model: '',
  instructions: '',
}

// Render [1,2,3,7] as "1-3, 7" so a long gap list stays readable.
function formatPageList(pages) {
  const runs = []
  for (const p of [...new Set(pages)].sort((a, b) => a - b)) {
    const last = runs[runs.length - 1]
    if (last && p === last[1] + 1) last[1] = p
    else runs.push([p, p])
  }
  return runs.map(([a, b]) => (a === b ? `${a}` : `${a}\u2013${b}`)).join(', ')
}

export default function App() {
  const { stream, isStreaming, cancel } = useStream()

  const [showLanding, setShowLanding] = useState(true)
  const [showEval, setShowEval] = useState(false)

  // Warm light theme: make sure no legacy `dark` class lingers from before.
  useEffect(() => {
    document.documentElement.classList.remove('dark')
  }, [])

  // Persisted settings + input. Both start in the guest namespace: the
  // Supabase session resolves asynchronously, so the account is unknown for
  // the first render or two. `scopeId` below tracks whose data is loaded.
  const [settings, setSettings] = useState(() => ({ ...DEFAULT_SETTINGS, ...readJSON(SETTINGS_KEY, GUEST, {}) }))
  const { mode, tone, length, format, model, instructions } = settings
  const setSetting = (key) => (value) => setSettings((s) => ({ ...s, [key]: value }))

  const [inputText, setInputText] = useState(() => readText(INPUT_KEY, GUEST))
  // [{page, start, end}] when the source is a PDF, so citations can name the
  // page they came from. Deliberately not persisted: it describes a file the
  // browser no longer has after a reload.
  const [pageSpans, setPageSpans] = useState([])

  const [models, setModels] = useState([])
  // The server's MAX_TEXT_CHARS. Until /api/health answers, assume the
  // documented default; InputPanel holds the source to whatever lands here.
  const [maxChars, setMaxChars] = useState(300000)
  // Page the left-hand PDF viewer is parked on, driven by citation clicks.
  const [pdfPage, setPdfPage] = useState(null)

  // The attached document, held here rather than in InputPanel: Generate swaps
  // the entry page for the workspace, remounting that panel, and the preview
  // has to outlive the swap — the workspace is the whole point of having it.
  const [preview, setPreview] = useState(null) // { kind: 'pdf', url, name }
  const previewUrlRef = useRef(null)

  const openPreview = useCallback((file, name) => {
    const url = URL.createObjectURL(file)
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current)
    previewUrlRef.current = url
    setPreview({ kind: 'pdf', url, name })
    setPdfPage(null)
  }, [])

  const clearPreview = useCallback(() => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current)
      previewUrlRef.current = null
    }
    setPreview(null)
    setPdfPage(null)
  }, [])

  // Object URLs leak until revoked.
  useEffect(() => () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current)
  }, [])

  const [agentSteps, setAgentSteps] = useState(INITIAL_STEPS)
  const [plan, setPlan] = useState(null)
  const [notes, setNotes] = useState('')
  // The source the notes on screen were built from. Compared against the live
  // input so the workspace can tell "regenerate the same thing" apart from
  // "you swapped the document and these notes are now stale".
  const [generatedFrom, setGeneratedFrom] = useState('')
  const [notesBefore, setNotesBefore] = useState(null)
  const [sources, setSources] = useState([])
  // Durable record of whether every page of the source was actually
  // written. A run that lost a window used to be indistinguishable from
  // a complete one once the status stream ended.
  const [coverage, setCoverage] = useState(null)
  const [critique, setCritique] = useState(null)
  const [quiz, setQuiz] = useState('')
  const [flashcards, setFlashcards] = useState('')

  const [user, setUser] = useState(null)
  const [authOpen, setAuthOpen] = useState(false)
  const [sharedView, setSharedView] = useState(null) // read-only shared session
  const { history, cloud, addSession, updateSession, deleteSession, shareSession } = useHistory(user)
  const [activeTab, setActiveTab] = useState('notes')
  const [provider, setProvider] = useState(null)
  const [toast, setToast] = useState(null)
  const [error, setError] = useState(null)
  const [blocked, setBlocked] = useState(null)
  const [menuOpen, setMenuOpen] = useState(false)
  const [showOptions, setShowOptions] = useState(false)

  const [rewriting, setRewriting] = useState(null)
  const [quizRegen, setQuizRegen] = useState(false)
  const [cardsRegen, setCardsRegen] = useState(false)

  const bufferRef = useRef('') // accumulates streamed note deltas
  const titleRef = useRef('') // AI-generated session title

  // ----- Account scoping ----------------------------------------------------
  // `account` is who is signed in now; `scopeId` is whose data is currently in
  // state. They differ for exactly one render after the session changes, which
  // is the window the effect below uses to swap everything over.
  const account = accountOf(user)
  const [scopeId, setScopeId] = useState(GUEST)

  // Wipe the workspace and re-read storage whenever the signed-in account
  // changes. Without this, signing out and signing in as someone else left
  // their predecessor's source text in the input panel and their notes, quiz,
  // flashcards and citations on screen — none of it belonged to the new user.
  // Everything downstream of a source: the notes written from it, the
  // citations that point into it, and the quiz and cards derived from those.
  // Kept separate from the source itself because the account switch below
  // reloads a different source, while "new document" clears it outright.
  const clearWorkspace = useCallback(() => {
    setNotes('')
    setNotesBefore(null)
    setSources([])
    setQuiz('')
    setFlashcards('')
    setCritique(null)
    setPlan(null)
    setError(null)
    setBlocked(null)
    setAgentSteps(INITIAL_STEPS.map((s) => ({ ...s })))
    setActiveTab('notes')
    setShowOptions(false)
    setGeneratedFrom('')
    bufferRef.current = ''
    titleRef.current = ''
  }, [])

  // Going Home is how you start another document, and it used to reset
  // nothing: `showLanding` only swaps which view renders, App never unmounts,
  // and `inputText` is mirrored to localStorage. So the next document began
  // holding the previous PDF's extracted text, its page spans and its preview
  // — and pressing Generate would have sent that stale source.
  //
  // History is untouched: finished runs were already saved to it, and this
  // clears only what is currently active.
  const startNewDocument = useCallback(() => {
    cancel() // a run still streaming would keep writing into the cleared state
    setInputText('')
    setPageSpans([])
    setPdfPage(null)
    clearPreview()
    clearWorkspace()
    setShowLanding(true)
  }, [cancel, clearPreview, clearWorkspace])

  useEffect(() => {
    if (scopeId === account) return

    cancel() // a stream started by the previous account must not keep writing

    setSettings({ ...DEFAULT_SETTINGS, ...readJSON(SETTINGS_KEY, account, {}) })
    setInputText(readText(INPUT_KEY, account))
    setPageSpans([])
    setPdfPage(null)
    clearPreview()
    clearWorkspace()

    setScopeId(account)
  }, [account, scopeId, cancel, clearPreview, clearWorkspace])

  // ----- Persist settings + input -----------------------------------------
  // Only once `scopeId` has caught up with the signed-in account. In the
  // render right after a sign-in, `account` is already the new user while the
  // state still holds the previous one's — writing then would file account
  // A's source text under account B.
  const hydrated = scopeId === account

  useEffect(() => {
    if (!hydrated) return
    writeJSON(SETTINGS_KEY, scopeId, settings)
  }, [settings, scopeId, hydrated])

  useEffect(() => {
    if (!hydrated) return
    writeText(INPUT_KEY, scopeId, inputText)
  }, [inputText, scopeId, hydrated])

  // ----- Health + models ---------------------------------------------------
  useEffect(() => {
    fetch('/api/health')
      .then((r) => r.json())
      .then((d) => {
        setProvider(d.provider)
        if (d.max_text_chars) setMaxChars(d.max_text_chars)
      })
      .catch(() => setProvider(null))

    fetch('/api/models')
      .then((r) => r.json())
      .then((d) => {
        setModels(d.models || [])
        setSettings((s) => (s.model ? s : { ...s, model: d.default || '' }))
      })
      .catch(() => setModels([]))
  }, [])

  // One timer at a time: a second toast used to be cut short by the first
  // one's pending timeout.
  const toastTimer = useRef(null)
  const showToast = (msg) => {
    setToast(msg)
    clearTimeout(toastTimer.current)
    toastTimer.current = setTimeout(() => setToast(null), 2000)
  }
  useEffect(() => () => clearTimeout(toastTimer.current), [])

  // ----- Step helpers ------------------------------------------------------
  const setStep = useCallback((stepName, status, message) => {
    setAgentSteps((prev) =>
      prev.map((s) => (s.step === stepName ? { ...s, status, message: message ?? s.message } : s))
    )
  }, [])

  // Roving-tabindex support for the workspace tablist.
  const tabRefs = useRef({})
  const onTabKeyDown = useCallback(
    (e) => {
      const ids = TABS.map((t) => t.id)
      const i = ids.indexOf(activeTab)
      const next =
        e.key === 'ArrowRight' ? ids[(i + 1) % ids.length]
        : e.key === 'ArrowLeft' ? ids[(i - 1 + ids.length) % ids.length]
        : e.key === 'Home' ? ids[0]
        : e.key === 'End' ? ids[ids.length - 1]
        : null
      if (!next) return
      e.preventDefault()
      setActiveTab(next)
      // Focus has to follow selection or the arrow keys strand the user on a
      // button that is no longer the active tab.
      tabRefs.current[next]?.focus()
    },
    [activeTab]
  )

  // ----- Generation progress ----------------------------------------------
  // Elapsed is ticked here rather than derived during render so the clock
  // advances while the model is quiet (which, on the 550B, is most of a run).
  const [elapsed, setElapsed] = useState(0)
  const startedAtRef = useRef(null)
  const stepStartRef = useRef(null)   // { step, at } for the step now running
  const runTimingsRef = useRef({})    // step -> seconds, for this run

  useEffect(() => {
    if (!isStreaming) return
    const id = setInterval(() => {
      setElapsed(startedAtRef.current ? (Date.now() - startedAtRef.current) / 1000 : 0)
    }, 1000)
    return () => clearInterval(id)
  }, [isStreaming])

  const markActive = useCallback((stepName, message) => {
    // A step going active closes out the previous one.
    const now = Date.now()
    const prevStep = stepStartRef.current
    if (prevStep && prevStep.step !== stepName) {
      runTimingsRef.current[prevStep.step] = (now - prevStep.at) / 1000
    }
    if (!prevStep || prevStep.step !== stepName) stepStartRef.current = { step: stepName, at: now }

    setAgentSteps((prev) => {
      const idx = prev.findIndex((s) => s.step === stepName)
      return prev.map((s, i) => {
        if (i < idx && s.status !== 'done') return { ...s, status: 'done' }
        if (s.step === stepName) return { ...s, status: 'active', message: message ?? s.message }
        return s
      })
    })
  }, [])

  // ----- Auth (Supabase) ---------------------------------------------------
  useEffect(() => {
    if (!supabase) return
    supabase.auth.getSession().then(({ data }) => {
      const signedIn = data.session?.user ?? null
      setUser(signedIn)
      // Already signed in on arrival -> open the app, don't park them on the
      // marketing page. This is the only path a Google sign-in can take: the
      // OAuth redirect reloads the page, so handing the user back from the
      // sign-in call (see handleAuthed) never runs. It also covers simply
      // returning with a live session. Only on this first resolution, so the
      // logo's "back to home" keeps working afterwards.
      if (signedIn) setShowLanding(false)
    })
    const { data: sub } = supabase.auth.onAuthStateChange((_e, session) => setUser(session?.user ?? null))
    return () => sub.subscription.unsubscribe()
  }, [])

  const signOut = async () => {
    if (!supabase) return
    // Drop this account's source text and local history from the browser
    // before releasing the session — whoever signs in next on this machine
    // should not be able to read it. Settings (a preference, not content)
    // stay, still namespaced to the account.
    clearAccountContent(account)
    await supabase.auth.signOut()
  }

  // Auth gate: with Supabase enabled, only signed-in users may use the app.
  // (Backend enforces this too — this just keeps the UI honest.)
  const authRequired = supabaseEnabled && !user
  useEffect(() => {
    if (authRequired && !showLanding && !sharedView && !showEval) setShowLanding(true)
  }, [authRequired, showLanding, sharedView, showEval])

  // Signing in has to open the workspace and register the user in one go.
  // Doing only the former let the gate above fire in the render before
  // onAuthStateChange arrived — `user` was still null, so a fresh sign-in was
  // thrown straight back to the landing page and had to press Launch again.
  const handleAuthed = useCallback((signedIn) => {
    if (signedIn?.id) setUser(signedIn)
    setShowLanding(false)
  }, [])

  // Load a shared (public) session if the URL has ?share=<id>.
  useEffect(() => {
    if (!supabase) return
    const id = new URLSearchParams(window.location.search).get('share')
    if (!id) return
    supabase
      .from('sessions')
      .select('*')
      .eq('id', id)
      .single()
      .then(({ data }) => {
        if (data) setSharedView(data)
      })
  }, [])

  // ----- History -----------------------------------------------------------
  const saveToHistory = useCallback(
    (finalNotes, finalQuiz, finalCards, finalSources, finalCoverage) => {
      if (!finalNotes) return
      addSession({
        id: Date.now().toString(),
        date: new Date().toISOString(),
        mode,
        title: titleRef.current || '',
        tags: [],
        notes_preview: finalNotes.slice(0, 100),
        notes: finalNotes,
        quiz: finalQuiz,
        flashcards: finalCards,
        sources: finalSources || [],
        coverage: finalCoverage || null,
      })
      showToast('Saved to history ✓')
    },
    [mode, addSession]
  )

  const loadSession = (s) => {
    setNotes(s.notes || '')
    setQuiz(s.quiz || '')
    setFlashcards(s.flashcards || '')
    setSources(s.sources || [])
    setNotesBefore(null)
    setCritique(null)
    setPlan(null)
    setActiveTab('notes')
    // These notes came out of history, not out of whatever is in the input
    // panel — so the workspace should offer to generate, not to regenerate.
    setGeneratedFrom('')
  }

  const shareLink = async (id) => {
    const sid = await shareSession(id)
    if (sid) {
      const url = `${window.location.origin}/?share=${sid}`
      try {
        await navigator.clipboard.writeText(url)
        showToast('Share link copied ✓')
      } catch {
        showToast('Link: ' + url)
      }
    }
  }

  // ----- Generate ----------------------------------------------------------
  const handleGenerate = useCallback(() => {
    if (!inputText.trim() || isStreaming) return

    setNotes('')
    setNotesBefore(null)
    setSources([])
    setPdfPage(null) // citations from the previous run no longer apply
    setQuiz('')
    setFlashcards('')
    setCritique(null)
    setPlan(null)
    setError(null)
    setBlocked(null)
    setActiveTab('notes')
    setGeneratedFrom(inputText)
    bufferRef.current = ''
    titleRef.current = ''

    startedAtRef.current = Date.now()
    stepStartRef.current = null
    runTimingsRef.current = {}
    setElapsed(0)

    setAgentSteps(INITIAL_STEPS.map((s) => ({ ...s })))

    let latestQuiz = ''
    let latestCards = ''
    let latestSources = []

    stream(
      { text: inputText, mode, tone, length, format, model, instructions, page_spans: pageSpans },
      {
        onStatus: (step, message) => {
          if (step === 'gate') return // pre-pipeline gatekeeper, no visible step
          if (step === 'revise' && /no revision/i.test(message)) {
            setStep('revise', 'done', message)
          } else {
            markActive(step, message)
          }
        },
        onBlocked: (message, data) => {
          setBlocked({ message, subject: data?.subject })
          setAgentSteps(INITIAL_STEPS.map((s) => ({ ...s })))
        },
        onPlanDone: (data) => {
          setPlan(data)
          setStep('plan', 'done')
        },
        onSources: (data) => {
          latestSources = data || []
          setSources(latestSources)
        },
        onNotesDelta: (step, delta) => {
          bufferRef.current += delta
          setNotes(bufferRef.current)
          if (step === 'write') setStep('write', 'active')
          if (step === 'revise') setStep('revise', 'active')
        },
        onNotesDone: () => {
          setNotes(bufferRef.current)
          setStep('write', 'done')
        },
        onReviseStart: () => {
          setNotesBefore(bufferRef.current) // snapshot pre-revision notes
          bufferRef.current = ''
          setNotes('')
        },
        onNotesRevised: (text) => {
          bufferRef.current = text
          setNotes(text)
          setStep('revise', 'done')
        },
        onCritiqueDone: (data, message) => {
          setCritique(data)
          setStep('critique', 'done', message)
        },
        onTitleDone: (t) => {
          titleRef.current = t
        },
        onQuizDone: (text) => {
          latestQuiz = text
          setQuiz(text)
          setStep('quiz', 'done')
        },
        onFlashcardsDone: (text) => {
          latestCards = text
          setFlashcards(text)
          setStep('flashcards', 'done')
        },
        onDone: (data) => {
          const cov = data?.coverage || null
          setCoverage(cov)
          setAgentSteps((prev) =>
            prev.map((s) => (s.status === 'pending' || s.status === 'active' ? { ...s, status: 'done' } : s))
          )
          // Close out the step that was still running, then bank the run so the
          // next generation on this model can estimate its remaining time.
          const last = stepStartRef.current
          if (last) runTimingsRef.current[last.step] = (Date.now() - last.at) / 1000
          stepStartRef.current = null
          recordRun(model || 'default', runTimingsRef.current)
          saveToHistory(bufferRef.current, latestQuiz, latestCards, latestSources, cov)
        },
        onError: (message) => {
          setError(message)
          setAgentSteps((prev) => prev.map((s) => (s.status === 'active' ? { ...s, status: 'error' } : s)))
        },
      }
    )
  }, [inputText, pageSpans, isStreaming, mode, tone, length, format, model, instructions, stream, setStep, markActive, saveToHistory])

  // ----- Ctrl/Cmd+Enter to generate ---------------------------------------
  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault()
        handleGenerate()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [handleGenerate])

  // ----- Section regeneration ----------------------------------------------
  const regenQuiz = async () => {
    if (!notes) return
    setQuizRegen(true)
    try {
      const res = await apiFetch('/api/quiz', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes, model }),
      })
      const data = await res.json()
      if (data.quiz) setQuiz(data.quiz)
    } catch {
      /* ignore */
    } finally {
      setQuizRegen(false)
    }
  }

  const regenFlashcards = async () => {
    if (!notes) return
    setCardsRegen(true)
    try {
      const res = await apiFetch('/api/flashcards', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes, model }),
      })
      const data = await res.json()
      if (data.flashcards) setFlashcards(data.flashcards)
    } catch {
      /* ignore */
    } finally {
      setCardsRegen(false)
    }
  }

  const editSelection = async (selection, instruction) => {
    if (!notes || !selection) return
    const res = await apiFetch('/api/edit-selection', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ notes, selection, instruction, model }),
    })
    const data = await res.json()
    if (data.notes) {
      setNotesBefore(notes) // enable the diff view
      setNotes(data.notes)
      showToast('Notes updated ✓')
    }
    return data.notes
  }

  const rewriteNotes = async (direction) => {
    if (!notes) return
    setRewriting(direction)
    try {
      const res = await apiFetch('/api/rewrite', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes, direction, mode, tone, format, model }),
      })
      const data = await res.json()
      if (data.notes) {
        setNotesBefore(notes) // enable diff vs the previous version
        setNotes(data.notes)
        showToast(`Notes made ${direction} ✓`)
      }
    } catch {
      /* ignore */
    } finally {
      setRewriting(null)
    }
  }

  // Progress readout. `estimate` is null until this model has finished a run
  // on this machine, in which case we show elapsed only rather than guess.
  const progress = useMemo(() => {
    const total = agentSteps.length
    const doneCount = agentSteps.filter((s) => s.status === 'done').length
    const active = agentSteps.find((s) => s.status === 'active')
    const estimate = estimateTotalSeconds(model || 'default')
    return {
      total,
      current: Math.min(total, doneCount + (active ? 1 : 0)) || 1,
      label: active ? STEP_LABELS[active.step] || active.step : null,
      message: active?.message || '',
      remaining: estimate != null ? Math.max(0, estimate - elapsed) : null,
    }
  }, [agentSteps, elapsed, model])

  // One polite sentence per step change. Deliberately NOT wired to the
  // streaming note text — announcing every token would make the app unusable
  // with a screen reader.
  const liveMessage = isStreaming && progress.label
    ? `Step ${progress.current} of ${progress.total}: ${progress.label}. ${progress.message}`
    : ''

  // Source in the panel differs from the one the visible notes came from.
  const sourceChanged = !!inputText.trim() && inputText !== generatedFrom

  const providerBadge =
    provider === 'nvidia'
      ? { label: 'NVIDIA', cls: 'bg-lime-50 text-lime-700 ring-1 ring-lime-200 dark:bg-lime-900/30 dark:text-lime-300 dark:ring-lime-800/50' }
      : provider === 'gemini'
      ? { label: 'Gemini', cls: 'bg-sky-50 text-sky-600 ring-1 ring-sky-200 dark:bg-sky-900/30 dark:text-sky-300 dark:ring-sky-800/50' }
      : provider === 'ollama'
        ? { label: 'Ollama', cls: 'bg-violet-50 text-violet-600 ring-1 ring-violet-200 dark:bg-violet-900/30 dark:text-violet-300 dark:ring-violet-800/50' }
        : { label: '…', cls: 'bg-slate-100 text-slate-500 dark:bg-slate-700 dark:text-slate-300' }

  if (sharedView) {
    return (
      <Suspense fallback={<PanelFallback />}>
      <SharedNote
        session={sharedView}
        onClose={() => {
          setSharedView(null)
          window.history.replaceState({}, '', '/')
          setShowLanding(false)
        }}
      />
      </Suspense>
    )
  }

  if (showEval) {
    return (
      <Suspense fallback={<PanelFallback />}>
        <EvalDashboard onBack={() => setShowEval(false)} />
      </Suspense>
    )
  }

  if (showLanding) {
    return (
      <>
        <Landing
          authRequired={authRequired}
          onLaunch={() => setShowLanding(false)}
          onAuthed={handleAuthed}
          onEval={() => {
            setShowLanding(false)
            setShowEval(true)
          }}
          user={user}
          supabaseEnabled={supabaseEnabled}
          onSignIn={() => setAuthOpen(true)}
          onSignOut={signOut}
        />
        {authOpen && (
          <Suspense fallback={null}>
            <AuthModal onClose={() => setAuthOpen(false)} onAuthed={handleAuthed} />
          </Suspense>
        )}
      </>
    )
  }

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-latte-100 text-espresso-800">
      {/* Ambient background glow */}
      <div className="pointer-events-none fixed inset-0 -z-10">
        <div className="absolute left-[8%] top-[-12%] h-[420px] w-[420px] rounded-full bg-brand-400/25 blur-3xl" />
        <div className="absolute bottom-[-15%] right-[5%] h-[380px] w-[380px] rounded-full bg-rose-300/30 blur-3xl" />
      </div>

      {/* Minimal nav: logo left · Eval + auth + CTA right */}
      {/* Floating header: simple fixed buttons, pipeline strip top-center */}
      <div className="fixed inset-x-0 top-4 z-40 flex items-start justify-between gap-3 px-4">
        {/* left: logo */}
        <button
          onClick={startNewDocument}
          title="Back to home"
          className="glass-pill flex items-center gap-2.5 rounded-full py-1.5 pl-2 pr-4"
        >
          <span className="flex h-8 w-8 items-center justify-center rounded-full bg-espresso-900 text-white">
            <Icon.Book className="h-4 w-4" />
          </span>
          <span className="text-sm font-extrabold tracking-tight text-espresso-900">Agentic Notes</span>
        </button>

        {/* center: THE pipeline — the only one in the app */}
        <div className="hidden md:block">
          <PipelineStrip steps={agentSteps} />
        </div>

        {/* right: auth + menu */}
        <div className="flex items-center gap-2">
          {supabaseEnabled &&
            (user ? (
              <button
                onClick={signOut}
                title={user.email}
                className="glass-pill rounded-full px-4 py-2 text-sm font-semibold text-espresso-900"
              >
                Sign out
              </button>
            ) : (
              <button
                onClick={() => setAuthOpen(true)}
                className="rounded-full bg-espresso-900 px-4 py-2 text-sm font-semibold text-white shadow-soft transition-transform hover:scale-[1.03]"
              >
                Sign in
              </button>
            ))}
          <button
            onClick={() => setMenuOpen(true)}
            aria-label="Open menu"
            className="glass-pill flex h-10 w-10 items-center justify-center rounded-full text-espresso-900"
          >
            <Icon.Menu className="h-5 w-5" />
          </button>
        </div>
      </div>

      {/* Screen-reader progress. sr-only + polite: it narrates the pipeline
          without stealing focus and without reading the notes themselves. */}
      <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {liveMessage}
      </div>

      {menuOpen && (
      <Suspense fallback={null}>
      <MenuOverlay
        open={menuOpen}
        onClose={() => setMenuOpen(false)}
        items={[
          { label: 'Home', onClick: startNewDocument },
          { label: 'Eval', onClick: () => setShowEval(true) },
          { label: 'GitHub', href: 'https://github.com/Kodishalathrilok/agentic-notes' },
          ...(supabaseEnabled
            ? [user
                ? { label: 'Sign out', onClick: signOut }
                : { label: 'Sign in', onClick: () => setAuthOpen(true) }]
            : []),
        ]}
      />
      </Suspense>
      )}

      {/* Main */}
      <main className="mx-auto max-w-7xl px-5 pb-16 pt-24 sm:pt-28">
        {/* pipeline on mobile (header center is hidden there) */}
        <div className="mb-6 flex justify-center md:hidden">
          <PipelineStrip steps={agentSteps} />
        </div>

        {!notes && !isStreaming ? (
          /* ------------------------------------------------------------------
             PHASE 1 — input only. Upload/paste a source, then Generate.
             The output panel does not exist yet.
          ------------------------------------------------------------------ */
          <div className="animate-fade-in pt-6 sm:pt-16">
            {!inputText.trim() && (
              <h2 className="mb-5 text-center font-display text-[1.75rem] font-normal leading-tight tracking-tight text-neutral-900 sm:text-4xl">
                What do you want to learn?
              </h2>
            )}

            <div className="mx-auto w-full max-w-[672px]">
              {/* keyed on the account: a switch remounts the panel, dropping the
                  previous user's PDF preview, file name and draft. */}
              <InputPanel
                key={scopeId}
                inputText={inputText}
                setInputText={setInputText}
                setPageSpans={setPageSpans}
                isStreaming={isStreaming}
                preview={preview}
                onPreview={openPreview}
                onClearPreview={clearPreview}
                maxChars={maxChars}
              />
            </div>

            {/* Generate appears once a source is loaded */}
            {inputText.trim() && (
              <div className="mx-auto mt-3 w-full max-w-[672px] animate-slide-up">
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleGenerate}
                    disabled={isStreaming}
                    className="flex flex-1 items-center justify-center gap-2 rounded-full bg-neutral-900 px-6 py-3 text-sm font-medium text-white transition-opacity hover:opacity-85 disabled:pointer-events-none disabled:opacity-40"
                  >
                    <Icon.Zap className="h-4 w-4" />
                    Generate notes
                  </button>
                  <button
                    onClick={() => setShowOptions((v) => !v)}
                    className={`rounded-full border px-5 py-3 text-sm font-medium transition-colors ${
                      showOptions
                        ? 'border-neutral-900 bg-neutral-900 text-white'
                        : 'border-neutral-200 bg-white text-neutral-700 hover:border-neutral-300 hover:bg-neutral-50'
                    }`}
                  >
                    Options
                  </button>
                </div>

                {showOptions && (
                  <div className="mt-4 animate-fade-in">
                    <ControlPanel
                      mode={mode}
                      setMode={setSetting('mode')}
                      tone={tone}
                      setTone={setSetting('tone')}
                      length={length}
                      setLength={setSetting('length')}
                      format={format}
                      setFormat={setSetting('format')}
                      model={model}
                      setModel={setSetting('model')}
                      models={models}
                      instructions={instructions}
                      setInstructions={setSetting('instructions')}
                      onGenerate={handleGenerate}
                      isStreaming={isStreaming}
                      canGenerate={!!inputText.trim()}
                    />
                  </div>
                )}
              </div>
            )}

            {error && (
              <div className="mx-auto mt-4 w-full max-w-[672px] rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
                {error}
              </div>
            )}
            {blocked && (
              <div className="mx-auto mt-4 w-full max-w-[672px] rounded-2xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-800">
                🎓 <span className="font-semibold">This tool is for academic study material only.</span>{' '}
                {blocked.message} Try a study topic or source — e.g. Biology, History, Computer
                Science, or Economics.
              </div>
            )}

            {/* Recents on the empty home only */}
            {!inputText.trim() && history.length > 0 && (
              <div className="mx-auto mt-12 w-full max-w-[672px]">
                <h3 className="mb-3 text-sm font-medium text-neutral-500">Recents</h3>
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                  {history.slice(0, 3).map((s) => (
                    <button
                      key={s.id}
                      onClick={() => loadSession(s)}
                      className="group rounded-3xl border border-neutral-200 bg-white p-4 text-left shadow-[0_4px_10px_rgba(0,0,0,0.04)] transition-colors duration-200 hover:border-neutral-300 hover:bg-neutral-50"
                    >
                      <Icon.Clock className="mb-3 h-5 w-5 text-neutral-400 transition-colors group-hover:text-neutral-900" />
                      <p className="truncate text-sm font-medium text-neutral-900">{s.title || 'Untitled notes'}</p>
                      <p className="mt-0.5 text-xs text-neutral-400">
                        {/* sessions carry `date` (see useHistory); `createdAt`
                            never existed, so this always read "Saved session". */}
                        {s.date ? new Date(s.date).toLocaleDateString() : 'Saved session'}
                      </p>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          /* ------------------------------------------------------------------
             PHASE 2 — after Generate: input left, output slides in from the
             right with the Learning tab on top.
          ------------------------------------------------------------------ */
          <div className="grid grid-cols-1 items-stretch gap-6 lg:h-[calc(100vh-11rem)] lg:grid-cols-2">
            {/* Left: the source — fills the viewport, scrolls inside */}
            <div className="flex flex-col gap-3 lg:h-full lg:min-h-0">
              <div className="min-h-0 flex-1">
                <InputPanel
                  key={scopeId}
                  inputText={inputText}
                  setInputText={setInputText}
                  setPageSpans={setPageSpans}
                  isStreaming={isStreaming}
                  pdfPage={pdfPage}
                  preview={preview}
                  onPreview={openPreview}
                  onClearPreview={clearPreview}
                  maxChars={maxChars}
                  fill
                />
              </div>

              {/* Generate belongs here too, not only on the entry page. Once
                  notes exist the layout never goes back, so replacing the
                  source in the workspace used to leave a new document loaded
                  with nothing to press — the only way to run the pipeline
                  again was the unlabelled arrow in the far panel. */}
              {inputText.trim() && (
                <div className="mx-auto w-full max-w-[672px] shrink-0">
                  <button
                    onClick={handleGenerate}
                    disabled={isStreaming}
                    className={`flex w-full items-center justify-center gap-2 rounded-full px-6 py-3 text-sm font-medium transition-all disabled:pointer-events-none disabled:opacity-40 ${
                      sourceChanged
                        ? 'bg-neutral-900 text-white hover:opacity-85'
                        : 'border border-neutral-200 bg-white text-neutral-700 hover:border-neutral-300 hover:bg-neutral-50'
                    }`}
                  >
                    <Icon.Zap className="h-4 w-4" />
                    {isStreaming
                      ? 'Generating…'
                      : sourceChanged
                        ? 'Generate notes from this source'
                        : 'Regenerate notes'}
                  </button>
                  {sourceChanged && !isStreaming && (
                    <p className="mt-1.5 text-center text-xs text-amber-600">
                      The notes on the right are from your previous source.
                    </p>
                  )}
                </div>
              )}
            </div>

            {/* Right: OUTPUT PANEL — pinned to the viewport, scrolls inside */}
            <div className="flex min-h-[420px] animate-slide-in-right flex-col gap-3 lg:h-full lg:min-h-0">
              {/* Tabs and the panel's one action, on a single row.
                  There used to be a "Learning tab" caption above this naming
                  the container, directly on top of four tabs that already say
                  what the container is — two rows of chrome to say one thing. */}
              <div className="shrink-0 rounded-3xl border border-neutral-200 bg-white p-1.5">
                <div className="flex items-center gap-2">
                  <span
                    className={`ml-2 h-1.5 w-1.5 shrink-0 rounded-full ${isStreaming ? 'animate-pulse bg-amber-500' : 'bg-green-500'}`}
                    title={isStreaming ? 'Agents are working' : 'Idle'}
                  />
                  {isStreaming && (
                    <span className="hidden shrink-0 whitespace-nowrap text-xs text-neutral-500 sm:inline">
                      {progress.label && (
                        <span className="font-medium text-neutral-700">
                          {progress.current}/{progress.total} {progress.label}
                        </span>
                      )}
                      <span className="ml-1.5 tabular-nums">{formatClock(elapsed)}</span>
                      {progress.remaining != null && (
                        <span className="ml-1.5">· {formatRemaining(progress.remaining)}</span>
                      )}
                    </span>
                  )}
                  <div
                    role="tablist"
                    aria-label="Workspace"
                    onKeyDown={onTabKeyDown}
                    className="scroll-area flex min-w-0 flex-1 gap-1 overflow-x-auto"
                  >
                    {TABS.map((t) => (
                      <button
                        key={t.id}
                        id={`tab-${t.id}`}
                        role="tab"
                        aria-selected={activeTab === t.id}
                        aria-controls={`tabpanel-${t.id}`}
                        // Roving tabindex: one stop for the whole group, then
                        // the arrow keys move within it. Four separate tab
                        // stops is what the plain buttons gave before.
                        tabIndex={activeTab === t.id ? 0 : -1}
                        ref={(el) => (tabRefs.current[t.id] = el)}
                        onClick={() => setActiveTab(t.id)}
                        className={`whitespace-nowrap rounded-full px-4 py-2 text-sm font-medium transition-colors ${
                          activeTab === t.id
                            ? 'bg-neutral-900 text-white'
                            : 'text-neutral-500 hover:bg-neutral-100 hover:text-neutral-900'
                        }`}
                      >
                        {t.label}
                        {t.id === 'history' && history.length > 0 && (
                          <span className="ml-1 text-xs opacity-60">({history.length})</span>
                        )}
                      </button>
                    ))}
                  </div>
                  {isStreaming ? (
                    <button
                      onClick={cancel}
                      className="shrink-0 rounded-full border border-red-200 bg-red-50 px-3.5 py-1.5 text-xs font-medium text-red-600 transition-colors hover:bg-red-100"
                    >
                      Cancel
                    </button>
                  ) : (
                    <button
                      onClick={() => setShowOptions((v) => !v)}
                      className={`shrink-0 rounded-full px-3.5 py-1.5 text-xs font-medium transition-colors ${
                        showOptions
                          ? 'bg-neutral-900 text-white'
                          : 'text-neutral-500 hover:bg-neutral-100 hover:text-neutral-900'
                      }`}
                    >
                      Options
                    </button>
                  )}
                </div>
              </div>

              {/* everything below the Learning tab scrolls inside the column */}
              <div className="scroll-area flex flex-1 flex-col gap-3 lg:min-h-0 lg:overflow-y-auto lg:pr-1">
              {showOptions && !isStreaming && (
                <div className="animate-fade-in">
                  <ControlPanel
                    mode={mode}
                    setMode={setSetting('mode')}
                    tone={tone}
                    setTone={setSetting('tone')}
                    length={length}
                    setLength={setSetting('length')}
                    format={format}
                    setFormat={setSetting('format')}
                    model={model}
                    setModel={setSetting('model')}
                    models={models}
                    instructions={instructions}
                    setInstructions={setSetting('instructions')}
                    onGenerate={handleGenerate}
                    isStreaming={isStreaming}
                    canGenerate={!!inputText.trim()}
                  />
                </div>
              )}

              <PipelineInsights plan={plan} critique={critique} />

              {error && (
                <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
                  {error}
                </div>
              )}
              {blocked && (
                <div className="rounded-2xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-800">
                  🎓 <span className="font-semibold">This tool is for academic study material only.</span>{' '}
                  {blocked.message}
                </div>
              )}

              {/* Tab content.
                  The ErrorBoundary sits HERE, not around <App />, because
                  `notes` lives in this component: a render crash in a panel
                  now fails just the panel and leaves a finished generation
                  intact. Keying it on activeTab clears a stuck error when the
                  user switches away and back. */}
              <ErrorBoundary key={activeTab} label={TABS.find((t) => t.id === activeTab)?.label}>
              <div
                className="animate-fade-in"
                role="tabpanel"
                id={`tabpanel-${activeTab}`}
                aria-labelledby={`tab-${activeTab}`}
                tabIndex={0}
                aria-busy={isStreaming}
              >
                {activeTab === 'notes' && coverage && coverage.complete === false && (
                  <div
                    role="status"
                    className="mb-3 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900"
                  >
                    <strong>Incomplete coverage.</strong>{' '}
                    {coverage.processed_pages} of {coverage.total_pages} pages were
                    written.{' '}
                    {coverage.failed_pages?.length > 0
                      ? `Missing: page${coverage.failed_pages.length > 1 ? 's' : ''} ${formatPageList(coverage.failed_pages)}.`
                      : `${coverage.failed_windows?.length || 0} part(s) could not be generated.`}{' '}
                    Re-generating usually fixes this.
                  </div>
                )}
                {activeTab === 'notes' && (
                  <NotesOutput
                    notes={notes}
                    setNotes={setNotes}
                    quiz={quiz}
                    flashcards={flashcards}
                    notesBefore={notesBefore}
                    onRewrite={rewriteNotes}
                    rewriting={rewriting}
                    sources={sources}
                    onEditSelection={editSelection}
                    onCitePage={setPdfPage}
                    streaming={isStreaming}
                  />
                )}
                {activeTab === 'quiz' && (
                  <Suspense fallback={<PanelFallback />}>
                    <QuizPanel quiz={quiz} onRegenerate={notes ? regenQuiz : null} regenerating={quizRegen} />
                  </Suspense>
                )}
                {activeTab === 'flashcards' && (
                  <Suspense fallback={<PanelFallback />}>
                    <FlashcardPanel
                      flashcards={flashcards}
                      onRegenerate={notes ? regenFlashcards : null}
                      regenerating={cardsRegen}
                      accountId={scopeId}
                    />
                  </Suspense>
                )}
                {activeTab === 'history' && (
                  <Suspense fallback={<PanelFallback />}>
                    <HistoryPanel
                      history={history}
                      onLoad={loadSession}
                      onDelete={deleteSession}
                      onUpdate={updateSession}
                      onShare={cloud ? shareLink : null}
                      cloud={cloud}
                    />
                  </Suspense>
                )}
              </div>
              </ErrorBoundary>

              </div>

              {/* Ask anything — pinned at the bottom of the output panel */}
              <div className="flex shrink-0 items-center gap-2 rounded-2xl border border-neutral-200 bg-white py-2 pl-5 pr-2 transition-colors focus-within:border-neutral-400">
                <input
                  value={instructions}
                  onChange={(e) => setSetting('instructions')(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && inputText.trim() && !isStreaming) handleGenerate()
                  }}
                  disabled={isStreaming}
                  placeholder="Ask anything — e.g. focus on definitions, add examples…"
                  className="min-w-0 flex-1 bg-transparent text-sm text-neutral-800 placeholder:text-neutral-400 focus:outline-none"
                />
                <button
                  onClick={handleGenerate}
                  disabled={isStreaming || !inputText.trim()}
                  aria-label="Generate with these instructions"
                  className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-neutral-900 text-white transition-opacity hover:opacity-85 disabled:pointer-events-none disabled:opacity-30"
                >
                  <Icon.Send className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>
        )}
      </main>

      {authOpen && (
        <Suspense fallback={null}>
          <AuthModal onClose={() => setAuthOpen(false)} onAuthed={handleAuthed} />
        </Suspense>
      )}

      {/* Toast */}
      {toast && (
        <div className="glass-pill fixed bottom-6 left-1/2 z-50 flex -translate-x-1/2 animate-slide-up items-center gap-2 rounded-full px-4 py-2.5 text-sm font-semibold text-espresso-900 shadow-lift">
          <Icon.Check className="h-4 w-4 text-green-400 dark:text-green-600" />
          {toast.replace(' ✓', '')}
        </div>
      )}
    </div>
  )
}