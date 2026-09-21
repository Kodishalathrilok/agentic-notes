import { useState, useRef, useCallback } from 'react'
import { apiFetch } from '../lib/api'
import { readSseEvents } from '../lib/sse'

const STREAM_CLOSED_EARLY =
  "The connection closed before the notes finished — what's shown may be incomplete. Please generate again."

/**
 * Consume the /api/generate SSE stream.
 *
 * Usage:
 *   const { stream, isStreaming, error, cancel } = useStream()
 *   stream(payload, callbacks)
 *
 * Because the endpoint is a POST, we use fetch + ReadableStream rather than
 * the native EventSource (which only supports GET).
 */
export default function useStream() {
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState(null)
  const controllerRef = useRef(null)

  const cancel = useCallback(() => {
    if (controllerRef.current) {
      controllerRef.current.abort()
      controllerRef.current = null
    }
    setIsStreaming(false)
  }, [])

  const stream = useCallback(async (payload, callbacks = {}) => {
    const {
      onStatus,
      onBlocked,
      onPlanDone,
      onSources,
      onNotesDelta,
      onNotesDone,
      onReviseStart,
      onNotesRevised,
      onCritiqueDone,
      onTitleDone,
      onQuizDone,
      onFlashcardsDone,
      onDone,
      onError,
    } = callbacks

    setError(null)
    setIsStreaming(true)

    const controller = new AbortController()
    controllerRef.current = controller

    const dispatch = (event) => {
      switch (event.type) {
        case 'status':
          onStatus && onStatus(event.step, event.content)
          break
        case 'blocked':
          onBlocked && onBlocked(event.content, event.data)
          break
        case 'plan_done':
          onPlanDone && onPlanDone(event.data)
          break
        case 'sources':
          onSources && onSources(event.data)
          break
        case 'notes_delta':
          onNotesDelta && onNotesDelta(event.step, event.content)
          break
        case 'notes_done':
          onNotesDone && onNotesDone(event.content)
          break
        case 'revise_start':
          onReviseStart && onReviseStart()
          break
        case 'notes_revised':
          onNotesRevised && onNotesRevised(event.content)
          break
        case 'critique_done':
          onCritiqueDone && onCritiqueDone(event.data, event.content)
          break
        case 'title_done':
          onTitleDone && onTitleDone(event.content)
          break
        case 'quiz_done':
          onQuizDone && onQuizDone(event.content)
          break
        case 'flashcards_done':
          onFlashcardsDone && onFlashcardsDone(event.content)
          break
        case 'done':
          onDone && onDone(event.data)
          break
        case 'error':
          setError(event.content)
          onError && onError(event.content)
          break
        default:
          break
      }
    }

    try {
      const res = await apiFetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
      })

      if (!res.ok) {
        let detail = `Request failed (${res.status})`
        try {
          const body = await res.json()
          if (body.detail) detail = body.detail
        } catch {
          /* ignore */
        }
        throw new Error(detail)
      }

      const { terminal } = await readSseEvents(res.body, dispatch)

      // The backend always ends a run with `done`, `error` or `blocked`. If
      // the connection closed cleanly without one (proxy idle timeout, server
      // restart, deploy), the run is incomplete: surface it as an error so
      // steps stop spinning and the partial notes aren't saved as finished.
      // (A user cancel normally rejects with AbortError; the signal check is
      // a belt-and-braces guard so cancelling is never reported as an error.)
      if (!terminal && !controller.signal.aborted) throw new Error(STREAM_CLOSED_EARLY)
    } catch (err) {
      if (err.name === 'AbortError') {
        // user cancelled — not an error
      } else {
        setError(err.message)
        onError && onError(err.message)
      }
    } finally {
      controllerRef.current = null
      setIsStreaming(false)
    }
  }, [])

  return { stream, isStreaming, error, cancel }
}
