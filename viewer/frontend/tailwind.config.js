/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,html}",
    "./*.js",
  ],
  theme: {
    extend: {
      colors: {
        brand: { DEFAULT: "#2563eb", accent: "#F57D15", teal: "#1D728A" },
      },
      fontFamily: {
        sans: ['"Noto Sans TC"', "sans-serif"],
        mono: ['"Roboto Mono"', "monospace"],
      },
    },
  },
  plugins: [],
}