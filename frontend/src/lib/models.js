/**
 * Which model id to send, given the one the browser saved and what the server
 * offers now.
 *
 * The saved choice is kept across reloads and sign-outs, and the server's list
 * changes: models are retired, deployments are reconfigured. A saved id that is
 * no longer listed used to be sent anyway, and the server (rightly) answered
 * "Unknown model" - while the dropdown, which shows its first option when its
 * value matches none, displayed a model that was never the one sent.
 *
 * `models` is /api/models' list; `fallback` is its `default`. While the list is
 * not known - not loaded yet, or the request failed - the saved id is returned
 * unchanged: a guess there would discard a good choice on every slow page load.
 */
export function resolveModel(saved, models, fallback = '') {
  const ids = (models || []).map((m) => m.id)
  if (!ids.length) return saved || ''
  if (ids.includes(saved)) return saved
  return ids.includes(fallback) ? fallback : ids[0]
}
