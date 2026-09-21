import { describe, it, expect } from 'vitest'
import { lineRangeOf, resolveLineAnchors } from './lineAnchors'

// Minimal fake DOM: elements have attributes + parentNode; text nodes have
// only parentNode (no getAttribute), like the real thing.
function el(attrs = {}, parent = null) {
  return { parentNode: parent, getAttribute: (k) => (k in attrs ? String(attrs[k]) : null) }
}
const text = (parent) => ({ parentNode: parent, nodeType: 3 })

const root = el()
const line4 = el({ 'data-line': 4 }, root)
const strong4 = el({}, el({}, line4)) // <span><strong>…</strong></span>
const line9 = el({ 'data-line': 9 }, root)
const code = el({ 'data-line-start': 11, 'data-line-end': 15 }, root)
const codeInner = el({}, code)

describe('lineRangeOf', () => {
  it('walks up from a text node to the nearest data-line', () => {
    expect(lineRangeOf(text(strong4), root)).toEqual([4, 4])
  })
  it('uses start/end for multi-line blocks', () => {
    expect(lineRangeOf(text(codeInner), root)).toEqual([11, 15])
  })
  it('stops at the root and returns null outside anchored lines', () => {
    expect(lineRangeOf(text(root), root)).toBeNull()
    expect(lineRangeOf(text(el({}, root)), root)).toBeNull()
    expect(lineRangeOf(null, root)).toBeNull()
  })
  it('treats line 0 as a valid anchor and rejects junk values', () => {
    expect(lineRangeOf(text(el({ 'data-line': 0 }, root)), root)).toEqual([0, 0])
    expect(lineRangeOf(text(el({ 'data-line': 'x' }, root)), root)).toBeNull()
  })
})

describe('resolveLineAnchors', () => {
  it('returns one line for a selection inside a line', () => {
    expect(resolveLineAnchors(text(strong4), text(line4), root)).toEqual({ lineStart: 4, lineEnd: 4 })
  })
  it('orders a backwards (focus before anchor) selection', () => {
    expect(resolveLineAnchors(text(line9), text(strong4), root)).toEqual({ lineStart: 4, lineEnd: 9 })
  })
  it('covers the whole of a code block at either end', () => {
    expect(resolveLineAnchors(text(line9), text(codeInner), root)).toEqual({ lineStart: 9, lineEnd: 15 })
  })
  it('is null when either end is not anchored', () => {
    expect(resolveLineAnchors(text(line4), text(root), root)).toBeNull()
  })
})
