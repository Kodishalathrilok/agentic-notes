import { Component } from 'react'

/**
 * Catches render errors in a subtree and shows a recoverable message instead of
 * unmounting the whole app.
 *
 * Placement is the whole point. A generation takes minutes, and the notes live
 * in App's state — so a boundary wrapped around <App /> would "handle" a crash
 * in the notes renderer by throwing that work away. The boundary that matters is
 * the one INSIDE App, around the tab content: it fails just the panel and leaves
 * `notes` intact, so "Try again" can re-render the same data.
 *
 * Has to be a class — React still exposes no hook for componentDidCatch.
 */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
    this.reset = this.reset.bind(this)
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    // Keep the stack in the console — the card below stays deliberately vague
    // because it's shown to end users, not to whoever is debugging.
    console.error(`[ErrorBoundary${this.props.label ? `: ${this.props.label}` : ''}]`, error, info)
  }

  reset() {
    this.setState({ error: null })
    this.props.onReset?.()
  }

  render() {
    if (!this.state.error) return this.props.children

    return (
      <div
        role="alert"
        className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600"
      >
        <p className="font-semibold">
          {this.props.label ? `${this.props.label} couldn't be displayed.` : 'Something went wrong.'}
        </p>
        <p className="mt-1 text-red-500">
          The rest of the page still works, and nothing you generated has been lost.
        </p>
        <button
          onClick={this.reset}
          className="mt-3 rounded-full border border-red-300 bg-white px-4 py-1.5 text-xs font-semibold text-red-700 transition-colors hover:bg-red-100"
        >
          Try again
        </button>
      </div>
    )
  }
}
