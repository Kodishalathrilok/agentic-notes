import { describe, it, expect } from 'vitest'
import { readSseEvents } from './sse'

const enc = new TextEncoder()

// Build a ReadableStream that yields each item as one chunk. Strings are
// UTF-8 encoded; Uint8Arrays are passed through (for byte-level splits).
function streamOf(chunks) {
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(typeof c === 'string' ? enc.encode(c) : c)
      controller.close()
    },
  })
}

const frame = (obj, sep = '\n\n') => `data: ${JSON.stringify(obj)}${sep}`

async function collect(chunks) {
  const events = []
  const result = await readSseEvents(streamOf(chunks), (e) => events.push(e))
  return { events, ...result }
}

describe('readSseEvents', () => {
  it('dispatches all events in order and reports terminal on done', async () => {
    const { events, terminal } = await collect([
      frame({ type: 'status', step: 'plan', content: 'Planning' }),
      frame({ type: 'notes_delta', step: 'write', content: 'Hello' }),
      frame({ type: 'notes_done', content: 'Hello' }),
      frame({ type: 'done', data: { coverage: 1 } }),
    ])
    expect(events.map((e) => e.type)).toEqual(['status', 'notes_delta', 'notes_done', 'done'])
    expect(events[3].data).toEqual({ coverage: 1 })
    expect(terminal).toBe(true)
  })

  it('reports not terminal when the stream closes mid-run', async () => {
    const { events, terminal } = await collect([
      frame({ type: 'status', step: 'write', content: 'Writing' }),
      frame({ type: 'notes_delta', step: 'write', content: 'Partial notes' }),
    ])
    expect(events).toHaveLength(2)
    expect(terminal).toBe(false)
  })

  it('reports not terminal for an empty stream', async () => {
    const { events, terminal } = await collect([])
    expect(events).toEqual([])
    expect(terminal).toBe(false)
  })

  it('parses events split across chunks, including inside a multi-byte char', async () => {
    const text = frame({ type: 'notes_delta', step: 'write', content: 'a — b' }) + frame({ type: 'done' })
    const bytes = enc.encode(text)
    // "—" is 3 bytes (E2 80 94); split between its first and second byte.
    const dash = bytes.findIndex((b) => b === 0xe2)
    const parts = [bytes.slice(0, 5), bytes.slice(5, dash + 1), bytes.slice(dash + 1, dash + 2), bytes.slice(dash + 2)]
    const { events, terminal } = await collect(parts)
    expect(events[0].content).toBe('a — b')
    expect(events.map((e) => e.type)).toEqual(['notes_delta', 'done'])
    expect(terminal).toBe(true)
  })

  it('handles one byte per chunk', async () => {
    const bytes = enc.encode(frame({ type: 'title_done', content: 'Café — notes' }) + frame({ type: 'done' }))
    const { events, terminal } = await collect(Array.from(bytes, (b) => new Uint8Array([b])))
    expect(events[0].content).toBe('Café — notes')
    expect(terminal).toBe(true)
  })

  it('handles CRLF frame separators, including a CRLF split across chunks', async () => {
    const { events, terminal } = await collect([
      'data: {"type":"status","step":"plan"}\r\n\r',
      '\ndata: {"type":"done"}\r\n\r\n',
    ])
    expect(events.map((e) => e.type)).toEqual(['status', 'done'])
    expect(terminal).toBe(true)
  })

  it('joins multiple data: lines in one frame', async () => {
    const { events } = await collect(['data: {"type":\ndata: "done"}\n\n'])
    expect(events).toEqual([{ type: 'done' }])
  })

  it('dispatches a final frame without a trailing blank line', async () => {
    const { events, terminal } = await collect([
      frame({ type: 'notes_done', content: 'x' }),
      frame({ type: 'done', data: {} }, ''),
    ])
    expect(events.map((e) => e.type)).toEqual(['notes_done', 'done'])
    expect(terminal).toBe(true)
  })

  it('dispatches a final unterminated non-terminal frame without counting it terminal', async () => {
    const { events, terminal } = await collect([frame({ type: 'status', step: 'write' }, '\n')])
    expect(events.map((e) => e.type)).toEqual(['status'])
    expect(terminal).toBe(false)
  })

  it('ignores non-JSON keep-alive and comment frames', async () => {
    const { events, terminal } = await collect([
      ': ping\n\n',
      'data: ping\n\n',
      'event: keepalive\ndata:\n\n',
      frame({ type: 'status', step: 'plan' }),
      ': ping - 2026-09-21 00:00:00\r\n\r\n',
      frame({ type: 'done' }),
      ': trailing comment\n\n',
    ])
    expect(events.map((e) => e.type)).toEqual(['status', 'done'])
    expect(terminal).toBe(true)
  })

  it.each(['error', 'blocked'])('treats %s as terminal', async (type) => {
    const { events, terminal } = await collect([frame({ type: 'status', step: 'gate' }), frame({ type, content: 'msg' })])
    expect(events.at(-1).type).toBe(type)
    expect(terminal).toBe(true)
  })

  it('keeps reading when a handler throws', async () => {
    const seen = []
    const { terminal } = await readSseEvents(streamOf([frame({ type: 'status' }), frame({ type: 'done' })]), (e) => {
      seen.push(e.type)
      if (e.type === 'status') throw new Error('boom')
    })
    expect(seen).toEqual(['status', 'done'])
    expect(terminal).toBe(true)
  })

  it('propagates read errors (e.g. AbortError on cancel)', async () => {
    const body = new ReadableStream({
      start(c) {
        c.enqueue(enc.encode(frame({ type: 'status' })))
        c.error(new DOMException('The operation was aborted.', 'AbortError'))
      },
    })
    await expect(readSseEvents(body, () => {})).rejects.toMatchObject({ name: 'AbortError' })
  })
})
