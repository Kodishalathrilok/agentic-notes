// Map a DOM selection back to SOURCE lines of the notes markdown.
//
// NotesOutput renders each markdown line as one element tagged data-line={idx}
// (multi-line blocks such as code carry data-line-start/data-line-end). The
// highlighted text itself is rendered - bold markers, bullets and [n]
// citations are gone - so it can't be found in the markdown reliably; the
// line range can. The server edits those raw lines and splices them back.

function num(v) {
  if (v === null || v === undefined || v === '') return null
  const n = Number(v)
  return Number.isInteger(n) && n >= 0 ? n : null
}

// [start, end] of the nearest anchored ancestor of `node` (inclusive), or
// null if none is found before `root`. Text nodes have no attributes, so the
// walk simply moves on to their parent.
export function lineRangeOf(node, root) {
  for (let n = node; n && n !== root; n = n.parentNode) {
    if (typeof n.getAttribute !== 'function') continue
    const line = num(n.getAttribute('data-line'))
    if (line !== null) return [line, line]
    const start = num(n.getAttribute('data-line-start'))
    const end = num(n.getAttribute('data-line-end'))
    if (start !== null && end !== null && start <= end) return [start, end]
  }
  return null
}

// { lineStart, lineEnd } (ordered, inclusive) covering both ends of a
// selection, or null when either end isn't inside an anchored line - the
// caller then sends no anchors and the server falls back to text matching.
export function resolveLineAnchors(anchorNode, focusNode, root) {
  const a = lineRangeOf(anchorNode, root)
  const f = lineRangeOf(focusNode, root)
  if (!a || !f) return null
  return { lineStart: Math.min(a[0], f[0]), lineEnd: Math.max(a[1], f[1]) }
}
