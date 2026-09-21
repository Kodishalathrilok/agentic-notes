import Icon from './Icons'
import NotesOutput from './NotesOutput'

export default function SharedNote({ session, onClose }) {
  // `missing`: the link resolved to nothing (unshared, deleted or a bad id).
  const missing = !!session.missing
  return (
    <div className="min-h-screen bg-latte-100 text-espresso-800">
      <header className="fixed inset-x-0 top-0 z-40">
        <div className="mx-auto flex max-w-3xl items-center justify-between px-5 py-4">
          <div className="flex items-center gap-2.5">
            <span className="flex h-9 w-9 items-center justify-center rounded-2xl bg-gradient-to-br from-espresso-700 to-espresso-900 text-white shadow-lift">
              <Icon.Book className="h-5 w-5" />
            </span>
            <div className="leading-tight">
              <h1 className="text-base font-extrabold tracking-tight text-espresso-900">{missing ? 'Link unavailable' : session.title || 'Shared notes'}</h1>
              <p className="text-xs text-espresso-500">Shared · read-only</p>
            </div>
          </div>
          <button onClick={onClose} className="btn-primary px-4 py-2 text-sm">
            Open the app
          </button>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-4 pb-8 pt-28">
        {missing ? (
          <div className="card p-6 text-center text-sm text-espresso-600">
            This shared note isn't available. Its owner may have stopped sharing it or deleted it, or the link
            is incomplete.
          </div>
        ) : (
          <NotesOutput
            notes={session.notes || ''}
            setNotes={() => {}}
            quiz={session.quiz || ''}
            flashcards={session.flashcards || ''}
            sources={session.sources || []}
          />
        )}
      </main>
    </div>
  )
}
