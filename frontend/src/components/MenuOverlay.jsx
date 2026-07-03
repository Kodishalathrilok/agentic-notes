import { useEffect } from 'react'
import { createPortal } from 'react-dom'
import Icon from './Icons'

/**
 * Fullscreen espresso menu overlay with giant handwritten links (à la the
 * coffee-shop reference). items: [{ label, onClick } | { label, href }]
 */
export default function MenuOverlay({ open, onClose, items = [] }) {
  useEffect(() => {
    if (!open) return
    const onKey = (e) => e.key === 'Escape' && onClose()
    document.addEventListener('keydown', onKey)
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = ''
    }
  }, [open, onClose])

  if (!open) return null

  return createPortal(
    <div className="fixed inset-0 z-[60] flex flex-col bg-gradient-to-b from-espresso-800 to-espresso-900 animate-fade-in">
      {/* Top bar */}
      <div className="flex items-center justify-between px-6 py-5 sm:px-10">
        <span className="font-display text-3xl font-bold text-cream">Agentic Notes</span>
        <button
          onClick={onClose}
          aria-label="Close menu"
 className="flex h-12 w-12 items-center justify-center rounded-full bg-cream text-espresso-900 transition-transform hover:rotate-90"
        >
          <Icon.X className="h-5 w-5" />
        </button>
      </div>

      {/* Giant links */}
      <nav className="flex flex-1 flex-col items-center justify-center gap-2 px-6">
        {items.map((item, i) => {
          const cls =
            'group flex items-baseline gap-3 font-display text-6xl font-bold leading-tight text-cream transition-all duration-200 hover:translate-x-2 hover:text-brand-300 sm:text-8xl'
          const arrow = (
            <Icon.ArrowUpRight className="h-8 w-8 -translate-y-1 opacity-50 transition-all duration-200 group-hover:translate-x-1 group-hover:-translate-y-2 group-hover:opacity-100 sm:h-12 sm:w-12" />
          )
          const style = { animationDelay: `${80 + i * 90}ms` }
          return item.href ? (
            <a
              key={item.label}
              href={item.href}
              target="_blank"
              rel="noreferrer"
 className={`${cls} animate-wiggle-in`}
              style={style}
            >
              {item.label} {arrow}
            </a>
          ) : (
            <button
              key={item.label}
              onClick={() => {
                onClose()
                item.onClick && item.onClick()
              }}
 className={`${cls} animate-wiggle-in`}
              style={style}
            >
              {item.label} {arrow}
            </button>
          )
        })}
      </nav>

      <p className="pb-8 text-center text-xs font-semibold uppercase tracking-[0.3em] text-cream/40">
        Multi-agent · self-correcting · faithfulness-checked
      </p>
    </div>,
    document.body
  )
}
