/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          dark: '#0B0F17',
          navy: '#111827',
          card: '#1E293B',
          border: '#334155',
          cyan: '#06B6D4',
          cyanGlow: '#22D3EE',
          teal: '#14B8A6',
          accent: '#38BDF8'
        }
      }
    },
  },
  plugins: [],
}
