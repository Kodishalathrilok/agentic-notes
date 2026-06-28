import { useState, useRef, useCallback } from 'react'

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
      onPlanDone,
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
        case 'plan_done':
          onPlanDone && onPlanDone(event.data)
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
          onDone && onDone()
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
      const res = await fetch('/api/generate', {
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

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      // SSE frames are separated by a blank line. Each frame has one or more
      // `data:` lines; we concatenate them and JSON.parse the result.
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        // Normalise CRLF -> LF so frames split consistently regardless of
        // whether the server uses \n\n or \r\n\r\n between events.
        buffer += decoder.decode(value, { stream: true }).replace(/\r/g, '')

        let sepIndex
        while ((sepIndex = buffer.indexOf('\n\n')) !== -1) {
          const frame = buffer.slice(0, sepIndex)
          buffer = buffer.slice(sepIndex + 2)

          const dataLines = frame
            .split('\n')
            .filter((l) => l.startsWith('data:'))
            .map((l) => l.slice(5).trimStart())

          if (dataLines.length === 0) continue
          const raw = dataLines.join('\n')
          if (!raw) continue

          try {
            dispatch(JSON.parse(raw))
          } catch {
            /* ignore non-JSON keep-alive frames */
          }
        }
      }
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
