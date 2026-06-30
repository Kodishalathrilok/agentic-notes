import Icon from './Icons'
import NotesOutput from './NotesOutput'

export default function SharedNote({ session, darkMode, setDarkMode, onClose }) {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <header className="sticky top-0 z-30 border-b border-slate-200/70 bg-white/70 backdrop-blur-xl dark:border-slate-800 dark:bg-slate-900/70">
        <div className="mx-auto flex max-w-3xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2.5">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 text-white shadow-lift">
              <Icon.Book className="h-5 w-5" />
            </span>
            <div className="leading-tight">
              <h1 className="text-base font-extrabold tracking-tight">{session.title || 'Shared notes'}</h1>
              <p className="text-xs text-slate-400">Shared · read-only</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setDarkMode((d) => !d)}
              className="flex h-9 w-9 items-center justify-center rounded-xl border border-slate-200 text-slate-500 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-slate-800"
              aria-label="Toggle dark mode"
            >
              {darkMode ? <Icon.Sun className="h-4 w-4" /> : <Icon.Moon className="h-4 w-4" />}
            </button>
            <button onClick={onClose} className="btn-primary px-3.5 py-1.5 text-sm">
              Open the app
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-4 py-8">
        <NotesOutput
          notes={session.notes || ''}
          setNotes={() => {}}
          quiz={session.quiz || ''}
          flashcards={session.flashcards || ''}
          sources={session.sources || []}
        />
      </main>
    </div>
  )
}
