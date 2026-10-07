import { describe, it, expect } from 'vitest'
import { resolveModel } from './models'

// Seen on the live app: the Model dropdown showed the first option while the
// request carried an id saved weeks earlier, which the server no longer listed.
// It answered "Unknown model", and nothing on screen said which model was sent.

const LISTED = [
  { id: 'nvidia/ultra', label: 'Ultra' },
  { id: 'gemini-flash', label: 'Flash' },
]

describe('resolveModel', () => {
  it('keeps a saved model the server still offers', () => {
    expect(resolveModel('gemini-flash', LISTED, 'nvidia/ultra')).toBe('gemini-flash')
  })

  it('replaces a saved model the server no longer offers with its default', () => {
    expect(resolveModel('nvidia/retired-model', LISTED, 'gemini-flash')).toBe('gemini-flash')
  })

  it('uses the default when nothing was saved', () => {
    expect(resolveModel('', LISTED, 'nvidia/ultra')).toBe('nvidia/ultra')
    expect(resolveModel(undefined, LISTED, 'nvidia/ultra')).toBe('nvidia/ultra')
  })

  it('falls back to the first listed model when the default is not on the list', () => {
    expect(resolveModel('nvidia/retired-model', LISTED, 'also-not-listed')).toBe('nvidia/ultra')
    expect(resolveModel('nvidia/retired-model', LISTED, '')).toBe('nvidia/ultra')
    expect(resolveModel('nvidia/retired-model', LISTED)).toBe('nvidia/ultra')
  })

  it('changes nothing while the list is not known', () => {
    // Not loaded yet, or the request failed: a guess here would throw away a
    // perfectly good saved choice on every slow page load.
    expect(resolveModel('nvidia/retired-model', [], 'nvidia/ultra')).toBe('nvidia/retired-model')
    expect(resolveModel('gemini-flash', undefined, 'nvidia/ultra')).toBe('gemini-flash')
    expect(resolveModel('', [], 'nvidia/ultra')).toBe('')
  })

  it('always answers with an id the list contains, once the list is known', () => {
    const ids = LISTED.map((m) => m.id)
    for (const saved of ['', 'nvidia/ultra', 'gemini-flash', 'x', null]) {
      for (const fallback of ['', 'nvidia/ultra', 'y', undefined]) {
        expect(ids).toContain(resolveModel(saved, LISTED, fallback))
      }
    }
  })
})
