import { defineConfig } from 'vite'

export default defineConfig({
  // Dev server: serve frontend on :5173, proxy /api and /viewer paths to FastAPI on :8000
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
      '/waterfalls': 'http://localhost:8000',
    },
  },
  // Build output: viewer/frontend/dist/
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})