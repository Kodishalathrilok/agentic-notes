import { useState, useEffect, useCallback, useRef } from 'react'
import useStream from './hooks/useStream'
import InputPanel from './components/InputPanel'
import ControlPanel from './components/ControlPanel'
import AgentStatus from './components/AgentStatus'
import PipelineInsights from './components/PipelineInsights'
import NotesOutput from './components/NotesOutput'
import QuizPanel from './components/QuizPanel'
import FlashcardPanel from './components/FlashcardPanel'
import HistoryPanel from './components/HistoryPanel'
import Icon from './components/Icons'
import Landing from './components/Landing'
import EvalDashboard from './components/EvalDashboard'
import AuthModal from './components/AuthModal'
import SharedNote from './components/SharedNote'
import useHistory from './hooks/useHistory'
import { supabase, supabaseEnabled } from './lib/supabase'

const HISTORY_KEY = 'agentic-notes-history'
const SETTINGS_KEY = 'agentic-notes-settings'
const INPUT_KEY = 'agentic-notes-input'
const THEME_KEY = 'theme'

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

function loadJSON(key, fallback) {
  try {
    const raw = localStorage.getItem(key)
    return raw ? JSON.parse(raw) : fallback
  } catch {
    return fallback
  }
}

export default function App() {
  const { stream, isStreaming, cancel } = useStream()

  const [darkMode, setDarkMode] = useState(() => localStorage.getItem(THEME_KEY) === 'dark')
  const [showLanding, setShowLanding] = useState(true)
  const [showEval, setShowEval] = useState(false)

  // Persisted settings + input
  const [settings, setSettings] = useState(() => ({ ...DEFAULT_SETTINGS, ...loadJSON(SETTINGS_KEY, {}) }))
  const { mode, tone, length, format, model, instructions } = settings
  const setSetting = (key) => (value) => setSettings((s) => ({ ...s, [key]: value }))

  const [inputText, setInputText] = useState(() => localStorage.getItem(INPUT_KEY) || '')

  const [models, setModels] = useState([])

  const [agentSteps, setAgentSteps] = useState(INITIAL_STEPS)
  const [plan, setPlan] = useState(null)
  const [notes, setNotes] = useState('')
  const [notesBefore, setNotesBefore] = useState(null)
  const [sources, setSources] = useState([])
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

  const [rewriting, setRewriting] = useState(null)
  const [quizRegen, setQuizRegen] = useState(false)
  const [cardsRegen, setCardsRegen] = useState(false)

  const bufferRef = useRef('') // accumulates streamed note deltas
  const titleRef = useRef('') // AI-generated session title

  // ----- Persist settings + input -----------------------------------------
  useEffect(() => {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings))
  }, [settings])

  useEffect(() => {
    localStorage.setItem(INPUT_KEY, inputText)
  }, [inputText])

  // ----- Dark mode ---------------------------------------------------------
  useEffect(() => {
    const root = document.documentElement
    if (darkMode) {
      root.classList.add('dark')
      localStorage.setItem(THEME_KEY, 'dark')
    } else {
      root.classList.remove('dark')
      localStorage.setItem(THEME_KEY, 'light')
    }
  }, [darkMode])

  // ----- Health + models ---------------------------------------------------
  useEffect(() => {
    fetch('/api/health')
      .then((r) => r.json())
      .then((d) => setProvider(d.provider))
      .catch(() => setProvider(null))

    fetch('/api/models')
      .then((r) => r.json())
      .then((d) => {
        setModels(d.models || [])
        setSettings((s) => (s.model ? s : { ...s, model: d.default || '' }))
      })
      .catch(() => setModels([]))
  }, [])

  const showToast = (msg) => {
    setToast(msg)
    setTimeout(() => setToast(null), 2000)
  }

  // ----- Step helpers ------------------------------------------------------
  const setStep = useCallback((stepName, status, message) => {
    setAgentSteps((prev) =>
      prev.map((s) => (s.step === stepName ? { ...s, status, message: message ?? s.message } : s))
    )
  }, [])

  const markActive = useCallback((stepName, message) => {
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
    supabase.auth.getSession().then(({ data }) => setUser(data.session?.user ?? null))
    const { data: sub } = supabase.auth.onAuthStateChange((_e, session) => setUser(session?.user ?? null))
    return () => sub.subscription.unsubscribe()
  }, [])

  const signOut = async () => {
    if (supabase) await supabase.auth.signOut()
  }

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
    (finalNotes, finalQuiz, finalCards, finalSources) => {
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
    setQuiz('')
    setFlashcards('')
    setCritique(null)
    setPlan(null)
    setError(null)
    setBlocked(null)
    setActiveTab('notes')
    bufferRef.current = ''
    titleRef.current = ''

    setAgentSteps(INITIAL_STEPS.map((s) => ({ ...s })))

    let latestQuiz = ''
    let latestCards = ''
    let latestSources = []

    stream(
      { text: inputText, mode, tone, length, format, model, instructions },
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
        onDone: () => {
          setAgentSteps((prev) =>
            prev.map((s) => (s.status === 'pending' || s.status === 'active' ? { ...s, status: 'done' } : s))
          )
          saveToHistory(bufferRef.current, latestQuiz, latestCards, latestSources)
        },
        onError: (message) => {
          setError(message)
          setAgentSteps((prev) => prev.map((s) => (s.status === 'active' ? { ...s, status: 'error' } : s)))
        },
      }
    )
  }, [inputText, isStreaming, mode, tone, length, format, model, instructions, stream, setStep, markActive, saveToHistory])

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
      const res = await fetch('/api/quiz', {
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
      const res = await fetch('/api/flashcards', {
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
    const res = await fetch('/api/edit-selection', {
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
      const res = await fetch('/api/rewrite', {
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

  const providerBadge =
    provider === 'groq'
      ? { label: 'Groq', cls: 'bg-orange-50 text-orange-600 ring-1 ring-orange-200 dark:bg-orange-900/30 dark:text-orange-300 dark:ring-orange-800/50' }
      : provider === 'ollama'
        ? { label: 'Ollama', cls: 'bg-violet-50 text-violet-600 ring-1 ring-violet-200 dark:bg-violet-900/30 dark:text-violet-300 dark:ring-violet-800/50' }
        : { label: '…', cls: 'bg-slate-100 text-slate-500 dark:bg-slate-700 dark:text-slate-300' }

  if (sharedView) {
    return (
      <SharedNote
        session={sharedView}
        darkMode={darkMode}
        setDarkMode={setDarkMode}
        onClose={() => {
          setSharedView(null)
          window.history.replaceState({}, '', '/')
          setShowLanding(false)
        }}
      />
    )
  }

  if (showEval) {
    return <EvalDashboard onBack={() => setShowEval(false)} darkMode={darkMode} setDarkMode={setDarkMode} />
  }

  if (showLanding) {
    return (
      <>
        <Landing
          onLaunch={() => setShowLanding(false)}
          onEval={() => {
            setShowLanding(false)
            setShowEval(true)
          }}
          darkMode={darkMode}
          setDarkMode={setDarkMode}
          user={user}
          supabaseEnabled={supabaseEnabled}
          onSignIn={() => setAuthOpen(true)}
          onSignOut={signOut}
        />
        {authOpen && <AuthModal onClose={() => setAuthOpen(false)} />}
      </>
    )
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 transition-colors dark:bg-slate-950 dark:text-slate-100">
      {/* Header */}
      <header className="sticky top-0 z-30 border-b border-slate-200/70 bg-white/70 backdrop-blur-xl dark:border-slate-800 dark:bg-slate-900/70">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-2 px-4 py-3">
          <button
            onClick={() => setShowLanding(true)}
            className="flex items-center gap-3 text-left"
            title="Back to home"
          >
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 text-white shadow-lift">
              <Icon.Book className="h-5 w-5" />
            </span>
            <div className="leading-tight">
              <h1 className="text-base font-extrabold tracking-tight sm:text-lg">Agentic Notes</h1>
              <p className="hidden text-xs text-slate-400 sm:block">Multi-agent study generator</p>
            </div>
          </button>
          <div className="flex items-center gap-2.5">
            <button
              onClick={() => setShowEval(true)}
              className="hidden rounded-xl px-3 py-2 text-sm font-semibold text-slate-600 transition-colors hover:bg-slate-100 hover:text-slate-900 sm:inline-flex dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white"
            >
              Eval
            </button>
            <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ${providerBadge.cls}`}>
              <Icon.Zap className="h-3 w-3" />
              {providerBadge.label}
            </span>
            <button
              onClick={() => setDarkMode((d) => !d)}
              className="flex h-9 w-9 items-center justify-center rounded-xl border border-slate-200 text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-900 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-white"
              title="Toggle dark mode"
              aria-label="Toggle dark mode"
            >
              {darkMode ? <Icon.Sun className="h-4 w-4" /> : <Icon.Moon className="h-4 w-4" />}
            </button>
            {supabaseEnabled &&
              (user ? (
                <div className="flex items-center gap-2">
                  <span className="hidden max-w-[140px] truncate text-xs text-slate-500 sm:inline dark:text-slate-400">
                    {user.email}
                  </span>
                  <button
                    onClick={signOut}
                    className="rounded-xl border border-slate-200 px-3 py-1.5 text-sm font-semibold text-slate-600 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
                  >
                    Sign out
                  </button>
                </div>
              ) : (
                <button onClick={() => setAuthOpen(true)} className="btn-primary px-3.5 py-1.5 text-sm">
                  Sign in
                </button>
              ))}
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="mx-auto max-w-7xl px-4 py-6">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
          {/* Left column */}
          <div className="space-y-6 lg:col-span-2">
            <InputPanel inputText={inputText} setInputText={setInputText} isStreaming={isStreaming} />
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
            <p className="text-center text-xs text-slate-400">
              Tip: press{' '}
              <kbd className="rounded-md border border-slate-200 bg-slate-100 px-1.5 py-0.5 font-mono text-[10px] dark:border-slate-700 dark:bg-slate-800">
                Ctrl
              </kbd>{' '}
              +{' '}
              <kbd className="rounded-md border border-slate-200 bg-slate-100 px-1.5 py-0.5 font-mono text-[10px] dark:border-slate-700 dark:bg-slate-800">
                Enter
              </kbd>{' '}
              to generate
            </p>
            {isStreaming && (
              <button
                onClick={cancel}
                className="w-full rounded-xl border border-red-200 px-4 py-2 text-sm font-semibold text-red-600 transition-colors hover:bg-red-50 dark:border-red-900/60 dark:hover:bg-red-900/20"
              >
                Cancel generation
              </button>
            )}
          </div>

          {/* Right column */}
          <div className="space-y-6 lg:col-span-3">
            <AgentStatus steps={agentSteps} critique={critique} />
            <PipelineInsights plan={plan} critique={critique} />

            {error && (
              <div className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-600 dark:bg-red-900/20 dark:text-red-300">
                {error}
              </div>
            )}

            {blocked && (
              <div className="rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-800 dark:border-amber-800/60 dark:bg-amber-900/20 dark:text-amber-200">
                🎓 <span className="font-semibold">This tool is for academic study material only.</span>{' '}
                {blocked.message} Try a study topic or source — e.g. Biology, History, Computer
                Science, or Economics.
              </div>
            )}

            {/* Tab bar */}
            <div className="scroll-area flex w-full gap-1 overflow-x-auto rounded-xl border border-slate-200/70 bg-white p-1 dark:border-slate-800 dark:bg-slate-900/50">
              {TABS.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setActiveTab(t.id)}
                  className={`flex-1 whitespace-nowrap rounded-lg px-4 py-2 text-sm font-semibold transition-all duration-200 ${
                    activeTab === t.id
                      ? 'bg-brand-600 text-white shadow-soft'
                      : 'text-slate-500 hover:bg-slate-100 hover:text-slate-800 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-100'
                  }`}
                >
                  {t.label}
                  {t.id === 'history' && history.length > 0 && (
                    <span className="ml-1 text-xs opacity-70">({history.length})</span>
                  )}
                </button>
              ))}
            </div>

            {/* Tab content */}
            <div className="animate-fade-in" key={activeTab}>
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
                />
              )}
              {activeTab === 'quiz' && (
                <QuizPanel quiz={quiz} onRegenerate={notes ? regenQuiz : null} regenerating={quizRegen} />
              )}
              {activeTab === 'flashcards' && (
                <FlashcardPanel
                  flashcards={flashcards}
                  onRegenerate={notes ? regenFlashcards : null}
                  regenerating={cardsRegen}
                />
              )}
              {activeTab === 'history' && (
                <HistoryPanel
                  history={history}
                  onLoad={loadSession}
                  onDelete={deleteSession}
                  onUpdate={updateSession}
                  onShare={cloud ? shareLink : null}
                  cloud={cloud}
                />
              )}
            </div>
          </div>
        </div>
      </main>

      {authOpen && <AuthModal onClose={() => setAuthOpen(false)} />}

      {/* Toast */}
      {toast && (
        <div className="fixed bottom-6 left-1/2 z-50 flex -translate-x-1/2 animate-slide-up items-center gap-2 rounded-xl bg-slate-900 px-4 py-2.5 text-sm font-semibold text-white shadow-lift dark:bg-white dark:text-slate-900">
          <Icon.Check className="h-4 w-4 text-green-400 dark:text-green-600" />
          {toast.replace(' ✓', '')}
        </div>
      )}
    </div>
  )
}
