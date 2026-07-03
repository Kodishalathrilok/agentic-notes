/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Plus Jakarta Sans', 'Inter', 'system-ui', 'sans-serif'],
        display: ['Caveat', 'cursive'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      colors: {
        // Warm cream backgrounds (coffee-shop light theme)
        latte: {
          50: '#faf7f1',
          100: '#f3ecdf',
          200: '#eadfcc',
          300: '#ddceb4',
        },
        // Warm dark brown for text & CTAs
        espresso: {
          500: '#8a7263',
          600: '#6f594c',
          700: '#523d31',
          800: '#3b2b21',
          900: '#2b1d15',
        },
        cream: '#f6f1e7',
        // Accent — burnt caramel (replaces the old indigo scale)
        brand: {
          50: '#fdf3ec',
          100: '#f9e4d4',
          200: '#f2c9a8',
          300: '#e8a678',
          400: '#dc8350',
          500: '#c96733',
          600: '#a94f22',
          700: '#87401f',
          800: '#6b341d',
          900: '#4a2515',
        },
      },
      boxShadow: {
        soft: '0 2px 8px -2px rgb(59 43 33 / 0.15)',
        card: '0 2px 6px -1px rgb(59 43 33 / 0.08), 0 16px 40px -16px rgb(59 43 33 / 0.18)',
        lift: '0 18px 44px -14px rgb(59 43 33 / 0.35)',
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
      },
      animation: {
        'fade-in': 'fade-in 0.25s ease-out both',
        'slide-up': 'slide-up 0.3s ease-out both',
        marquee: 'marquee 26s linear infinite',
        'wiggle-in': 'wiggle-in 0.5s cubic-bezier(0.2, 0.7, 0.2, 1) both',
      },
    },
  },
  plugins: [],
}
