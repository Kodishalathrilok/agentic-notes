/**
 * Minimal Server-Sent Events reader for fetch() response bodies.
 *
 * The /api/generate endpoint is a POST, so we can't use the native
 * EventSource (GET only); instead we read the ReadableStream ourselves.
 * Every frame's `data:` payload is expected to be JSON; anything else
 * (keep-alive pings, comments) is ignored.
 */

// Event types the backend uses to end a run. run_agent always finishes with
// exactly one of these; `blocked` returns without a trailing `done`.
export const TERMINAL_EVENTS = new Set(['done', 'error', 'blocked'])

/**
 * Parse one SSE frame (already split off, CRLF-normalised) and dispatch it.
 * Returns the parsed event, or null if the frame carried no JSON payload.
 */
function handleFrame(frame, onEvent) {
  const dataLines = frame
    .split('\n')
    .filter((l) => l.startsWith('data:'))
    .map((l) => l.slice(5).trimStart())

  if (dataLines.length === 0) return null
  const raw = dataLines.join('\n')
  if (!raw) return null

  let event
  try {
    event = JSON.parse(raw)
  } catch {
    /* ignore non-JSON keep-alive frames */
    return null
  }
  // Callback errors are swallowed, as the original inline loop did, so one
  // bad handler can't abort the whole stream.
  try {
    onEvent(event)
  } catch {
    /* ignore */
  }
  return event
}

/**
 * Read an SSE byte stream to the end, calling onEvent(event) for each parsed
 * JSON event in order.
 *
 * Resolves to { terminal } where `terminal` is true if a terminal event
 * (`done` | `error` | `blocked`) was seen. A stream that closes cleanly
 * without one (proxy idle timeout, server restart, deploy) resolves with
 * terminal: false so the caller can report the run as incomplete.
 * Read errors (including AbortError on cancel) propagate to the caller.
 */
export async function readSseEvents(body, onEvent) {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let terminal = false

  const dispatch = (frame) => {
    const event = handleFrame(frame, onEvent)
    if (event && TERMINAL_EVENTS.has(event.type)) terminal = true
  }

  // SSE frames are separated by a blank line. Each frame has one or more
  // `data:` lines; we concatenate them and JSON.parse the result.
  while (true) {
    const { value, done } = await reader.read()
    // Normalise CRLF -> LF so frames split consistently regardless of
    // whether the server uses \n\n or \r\n\r\n between events.
    // On EOF, call decode() with no args to flush any bytes of a multi-byte
    // character still held by the decoder.
    buffer += (done ? decoder.decode() : decoder.decode(value, { stream: true })).replace(/\r/g, '')

    let sepIndex
    while ((sepIndex = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sepIndex)
      buffer = buffer.slice(sepIndex + 2)
      dispatch(frame)
    }

    if (done) {
      // A final frame without a trailing blank line would otherwise be
      // dropped at EOF (and a lost `done` misreported as a dropped
      // connection), so dispatch whatever is left.
      if (buffer.trim()) dispatch(buffer)
      break
    }
  }

  return { terminal }
}
