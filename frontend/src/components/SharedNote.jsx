import Icon from './Icons'
import NotesOutput from './NotesOutput'

export default function SharedNote({ session, onClose }) {
  return (
    <div className="min-h-screen bg-ink-950 text-stone-300">
      <header className="sticky top-0 z-30 border-b border-white/[0.06] bg-ink-950/70 backdrop-blur-xl">
        <div className="mx-auto flex max-w-3xl items-center justify-between px-5 py-4">
          <div className="flex items-center gap-2.5">
            <span className="flex h-9 w-9 items-center justify-center rounded-2xl bg-gradient-to-br from-brand-500 to-brand-700 text-white shadow-lift">
              <Icon.Book className="h-5 w-5" />
            </span>
            <div className="leading-tight">
              <h1 className="text-base font-extrabold tracking-tight text-cream">{session.title || 'Shared notes'}</h1>
              <p className="text-xs text-stone-500">Shared · read-only</p>
            </div>
          </div>
          <button onClick={onClose} className="btn-primary px-4 py-2 text-sm">
            Open the app
          </button>
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
