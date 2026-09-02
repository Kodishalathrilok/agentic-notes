/**
 * Per-account localStorage.
 *
 * WHY THIS EXISTS
 *
 * Everything the app persisted — the extracted source text, the settings, the
 * local history, the flashcard schedule — used to live under one global key.
 * localStorage is per-BROWSER, not per-account, so signing out and signing in
 * as somebody else handed the new account the previous one's data: their
 * uploaded document was still sitting in the input panel.
 *
 * Every key is now suffixed with the account it belongs to, so two accounts on
 * the same browser can never read each other's state. Signed-out use gets its
 * own `guest` namespace.
 */

const GUEST = 'guest'

// The keys as they were before namespacing. Read once at startup and retired,
// so data written by the old build can't outlive it.
const LEGACY_KEYS = [
  'agentic-notes-history',
  'agentic-notes-settings',
  'agentic-notes-input',
  'agentic-notes-srs',
]

export const INPUT_KEY = 'agentic-notes-input'
export const SETTINGS_KEY = 'agentic-notes-settings'
export const HISTORY_KEY = 'agentic-notes-history'
export const SRS_KEY = 'agentic-notes-srs'
// Per-model step timings, used to estimate how long a generation has left.
// Not account-scoped in spirit — it describes the models, not the user —
// but it goes through the same helpers, so it carries a fixed scope.
export const TIMINGS_KEY = 'agentic-notes-timings'

/** The storage key `base` resolves to for one account. */
export function scopedKey(base, accountId) {
  return `${base}:${accountId || GUEST}`
}

export function readText(base, accountId, fallback = '') {
  try {
    const raw = localStorage.getItem(scopedKey(base, accountId))
    return raw === null ? fallback : raw
  } catch {
    return fallback
  }
}

/**
 * Writes are best-effort. A 300k-character source plus a history of notes can
 * exceed the ~5 MB quota, and an uncaught QuotaExceededError inside a render
 * effect takes the whole app down — losing a draft is the better failure.
 */
export function writeText(base, accountId, value) {
  try {
    localStorage.setItem(scopedKey(base, accountId), value)
    return true
  } catch {
    return false
  }
}

export function readJSON(base, accountId, fallback) {
  try {
    const raw = localStorage.getItem(scopedKey(base, accountId))
    return raw ? JSON.parse(raw) : fallback
  } catch {
    return fallback
  }
}

export function writeJSON(base, accountId, value) {
  return writeText(base, accountId, JSON.stringify(value))
}

export function removeKey(base, accountId) {
  try {
    localStorage.removeItem(scopedKey(base, accountId))
  } catch {
    /* ignore */
  }
}

/**
 * Drop the content one account left behind, on explicit sign-out. Settings are
 * deliberately kept: they are preferences, not somebody's study material.
 */
export function clearAccountContent(accountId) {
  removeKey(INPUT_KEY, accountId)
  removeKey(HISTORY_KEY, accountId)
  removeKey(SRS_KEY, accountId)
}

/**
 * One-time cleanup of the pre-namespacing keys. History, settings and the
 * flashcard schedule are handed to the guest namespace (they were written
 * while the browser had no account scoping, so that is where they belong);
 * the input draft is dropped outright — it is the one that leaked a document
 * into the next person's session, and a stale draft is worth nothing.
 */
export function purgeLegacyKeys() {
  let store
  try {
    store = localStorage
  } catch {
    return
  }
  for (const base of LEGACY_KEYS) {
    let raw
    try {
      raw = store.getItem(base)
    } catch {
      continue
    }
    if (raw === null) continue
    if (base !== INPUT_KEY) {
      try {
        if (store.getItem(scopedKey(base, GUEST)) === null) {
          store.setItem(scopedKey(base, GUEST), raw)
        }
      } catch {
        /* full or unavailable — the removal below still matters most */
      }
    }
    try {
      store.removeItem(base)
    } catch {
      /* ignore */
    }
  }
}
