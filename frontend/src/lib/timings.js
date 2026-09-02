import { TIMINGS_KEY, readJSON, writeJSON } from './storage'

/**
 * Remembers how long each pipeline step took, per model, so the workspace can
 * estimate the time left on a generation.
 *
 * Deliberately learned rather than hard-coded: the same six steps take a few
 * seconds on Gemini Flash-Lite and several minutes on Nemotron 550B, so any
 * fixed table would be wrong for half the model picker. With no history for a
 * model we return null and the UI shows elapsed time only — an honest "no idea
 * yet" beats a confident wrong number.
 */

// Timings describe the models, not the person, so every account on this
// machine shares one bucket.
const SCOPE = 'shared'
const KEEP_PER_STEP = 5

function median(xs) {
  if (!xs.length) return 0
  const sorted = [...xs].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

/** Merge one finished run's per-step seconds into the stored history. */
export function recordRun(modelId, durations) {
  if (!modelId || !durations) return
  const entries = Object.entries(durations).filter(([, d]) => d > 0)
  if (!entries.length) return
  try {
    const all = readJSON(TIMINGS_KEY, SCOPE, {}) || {}
    const forModel = { ...(all[modelId] || {}) }
    for (const [step, seconds] of entries) {
      forModel[step] = [...(forModel[step] || []), seconds].slice(-KEEP_PER_STEP)
    }
    writeJSON(TIMINGS_KEY, SCOPE, { ...all, [modelId]: forModel })
  } catch {
    /* a full or disabled localStorage must never break a generation */
  }
}

/**
 * Expected total seconds for a full run on `modelId`, or null when we haven't
 * seen one finish yet. Sums the median of each step we have data for; steps we
 * have never timed simply contribute nothing, so the estimate is a floor that
 * tightens as history builds.
 */
export function estimateTotalSeconds(modelId) {
  if (!modelId) return null
  try {
    const forModel = (readJSON(TIMINGS_KEY, SCOPE, {}) || {})[modelId]
    if (!forModel) return null
    const total = Object.values(forModel).reduce((sum, xs) => sum + median(xs), 0)
    return total > 0 ? total : null
  } catch {
    return null
  }
}

/** 134 -> "2:14" */
export function formatClock(seconds) {
  const s = Math.max(0, Math.round(seconds))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

/**
 * Coarse "time left" wording. Rounded hard on purpose — a to-the-second
 * countdown on an estimate this soft would imply precision we don't have.
 */
export function formatRemaining(seconds) {
  if (seconds <= 5) return 'almost done'
  if (seconds < 60) return `~${Math.ceil(seconds / 10) * 10}s left`
  return `~${Math.ceil(seconds / 60)} min left`
}
