import { useEffect } from 'react'

/**
 * One rAF loop driving both hero cursor behaviours:
 *
 *  1. the liquid mask reveal on .hero-art-color (CSS vars --hx/--hy/--hr/--ox/--oy)
 *  2. the hero heading as a rigid 3D panel: the cursor orients it in space —
 *     the edge under the pointer comes forward, the far edge falls back, with
 *     real perspective foreshortening. Rotation is the effect; a small inverse
 *     translation and a hair of recession back it up. Supporting layers opt in
 *     with data-parallax="<weight>" and only translate, so the heading reads as
 *     one object rather than a group.
 *
 * Disabled entirely on touch devices and for prefers-reduced-motion.
 */
// degrees at the hero edge. Signs live in the tick: CSS rotateY(+d) pushes the
// RIGHT edge away, rotateX(+d) brings the BOTTOM edge forward, so both are
// negated below to put the near edge under the cursor.
const ROT_Y = 8
const ROT_X = 6
// secondary, inverse translation — deliberately smaller than the rotation
const TRANS_X = 10
const TRANS_Y = 7
// how far the panel recedes at the corners (1 -> 0.985)
const RECESS = 0.015

export default function useHeroCursor(heroRef) {
  useEffect(() => {
    const el = heroRef.current
    if (!el) return

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const fine = window.matchMedia('(hover: hover) and (pointer: fine)').matches
    // No cursor (or the user asked for calm) -> leave the hero exactly as designed.
    if (reduced || !fine) return

    const layers = Array.from(el.querySelectorAll('[data-parallax]')).map((node) => ({
      node,
      w: parseFloat(node.dataset.parallax) || 0,
      // the heading is the panel that rotates; everything else only drifts
      panel: node.classList.contains('hero-heading'),
    }))

    // The entry animations (animate-slide-up / te-slide) are `fill-mode: both`,
    // so they keep holding transform:translateY(0) forever and would override
    // ours. Once they've played out their held state is the default anyway, so
    // dropping the class is a visual no-op that frees `transform`.
    const release = setTimeout(() => {
      layers.forEach(({ node }) => node.classList.remove('animate-slide-up'))
    }, 1200)

    let amp = window.innerWidth < 1024 ? 0.55 : 1
    const onResize = () => {
      amp = window.innerWidth < 1024 ? 0.55 : 1
    }
    window.addEventListener('resize', onResize)

    let raf
    const pos = { x: el.offsetWidth / 2, y: el.offsetHeight * 0.4 }
    const tgt = { x: pos.x, y: pos.y }
    let r = 0
    let tr = 0
    // parallax: normalized -1..+1, lerped so the layers trail the cursor
    const n = { x: 0, y: 0 }
    const nt = { x: 0, y: 0 }
    let intensity = 0
    let intensityTarget = 0

    const onMove = (e) => {
      const rect = el.getBoundingClientRect()
      tgt.x = e.clientX - rect.left
      tgt.y = e.clientY - rect.top
      tr = 400
      intensityTarget = 1
      nt.x = (tgt.x / rect.width) * 2 - 1
      nt.y = (tgt.y / rect.height) * 2 - 1
      // instant vars for the existing spotlight glow
      el.style.setProperty('--mx', `${tgt.x}px`)
      el.style.setProperty('--my', `${tgt.y}px`)
    }
    const onLeave = () => {
      tr = 0
      intensityTarget = 0
      nt.x = 0
      nt.y = 0
    }

    const tick = (t) => {
      // ---- hero art mask (unchanged) ----
      pos.x += (tgt.x - pos.x) * 0.095
      pos.y += (tgt.y - pos.y) * 0.095
      r += (tr - r) * 0.055
      const w = t * 0.0016
      const wob = 1 + 0.05 * Math.sin(w * 2.1)
      el.style.setProperty('--hx', pos.x.toFixed(1))
      el.style.setProperty('--hy', pos.y.toFixed(1))
      el.style.setProperty('--hr', (r * wob).toFixed(1))
      el.style.setProperty('--ox', (Math.cos(w * 0.9) * r * 0.28).toFixed(1))
      el.style.setProperty('--oy', (Math.sin(w * 1.3) * r * 0.24).toFixed(1))

      // ---- heading as a 3D panel ----
      n.x += (nt.x - n.x) * 0.12
      n.y += (nt.y - n.y) * 0.12
      intensity += (intensityTarget - intensity) * 0.09
      const k = intensity * amp
      // negated so the edge nearest the cursor is the one that comes forward
      const rotY = (-n.x * ROT_Y * k).toFixed(3)
      const rotX = (n.y * ROT_X * k).toFixed(3)
      const tx = -n.x * TRANS_X * k
      const ty = -n.y * TRANS_Y * k
      // recession grows with distance from the hero centre, not with either axis
      const sc = 1 - RECESS * Math.min(1, Math.hypot(n.x, n.y)) * k

      for (let i = 0; i < layers.length; i++) {
        const l = layers[i]
        const base = `translate3d(${(tx * l.w).toFixed(2)}px, ${(ty * l.w).toFixed(2)}px, 0)`
        l.node.style.transform = l.panel
          ? `${base} rotateX(${rotX}deg) rotateY(${rotY}deg) scale(${sc.toFixed(5)})`
          : `${base} scale(${(1 - (1 - sc) * l.w).toFixed(5)})`
      }
      raf = requestAnimationFrame(tick)
    }

    el.addEventListener('pointermove', onMove)
    el.addEventListener('pointerleave', onLeave)
    raf = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(raf)
      clearTimeout(release)
      window.removeEventListener('resize', onResize)
      el.removeEventListener('pointermove', onMove)
      el.removeEventListener('pointerleave', onLeave)
      layers.forEach(({ node }) => (node.style.transform = ''))
    }
  }, [heroRef])
}
