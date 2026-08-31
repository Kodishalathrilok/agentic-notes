/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Questrial', 'Century Gothic', 'sans-serif'],
        // Landing hero only. Didone serif — the weight axis is loaded as a
        // 400..700 range so bold is a real cut, not a synthesised one.
        bodoni: ['"Bodoni Moda"', 'Didot', '"Bodoni MT"', 'Georgia', 'serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      colors: {
        // Pure white backgrounds (monochrome theme)
        latte: {
          50: '#ffffff',
          100: '#ffffff',
          200: '#f5f5f5',
          300: '#e8e8e8',
        },
        // Pure black for all text & CTAs
        espresso: {
          300: '#000000',
          400: '#000000',
          500: '#000000',
          600: '#000000',
          700: '#000000',
          800: '#000000',
          900: '#000000',
        },
        cream: '#ffffff',
        // Accent — neutral grayscale (monochrome like the reference)
        brand: {
          50: '#fafafa',
          100: '#f5f5f5',
          200: '#e5e5e5',
          300: '#d4d4d4',
          400: '#737373',
          500: '#000000',
          600: '#000000',
          700: '#000000',
          800: '#000000',
          900: '#000000',
        },
      },
      boxShadow: {
        soft: '0 2px 8px -2px rgb(0 0 0 / 0.12)',
        card: '0 2px 6px -1px rgb(0 0 0 / 0.06), 0 16px 40px -16px rgb(0 0 0 / 0.14)',
        lift: '0 18px 44px -14px rgb(0 0 0 / 0.28)',
        // Neumorphism — soft extrusion on the white surface
        neu: '10px 10px 24px rgb(0 0 0 / 0.07), -10px -10px 24px rgb(255 255 255 / 0.95)',
        'neu-sm': '5px 5px 12px rgb(0 0 0 / 0.06), -5px -5px 12px rgb(255 255 255 / 0.9)',
        'neu-inset': 'inset 4px 4px 10px rgb(0 0 0 / 0.05), inset -4px -4px 10px rgb(255 255 255 / 0.9)',
      },
      borderRadius: {
        xl: '0.875rem',
        '2xl': '1.125rem',
      },
      keyframes: {
        'fade-in': {
          '0%': { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-up': {
          '0%': { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '100%': { transform: 'translateX(100%)' },
        },
        marquee: {
          '0%': { transform: 'translateX(0)' },
          '100%': { transform: 'translateX(-50%)' },
        },
        'wiggle-in': {
          '0%': { opacity: '0', transform: 'translateY(24px) rotate(-2deg)' },
          '100%': { opacity: '1', transform: 'translateY(0) rotate(0deg)' },
        },
        'slide-in-right': {
          '0%': { opacity: '0', transform: 'translateX(48px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
      },
      animation: {
        'fade-in': 'fade-in 0.25s ease-out both',
        'slide-up': 'slide-up 0.3s ease-out both',
        marquee: 'marquee 26s linear infinite',
        'wiggle-in': 'wiggle-in 0.5s cubic-bezier(0.2, 0.7, 0.2, 1) both',
        'slide-in-right': 'slide-in-right 0.45s cubic-bezier(0.2, 0.7, 0.2, 1) both',
      },
    },
  },
  plugins: [],
}