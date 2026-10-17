/// <reference types="vitest/config" />
import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const api = process.env.COGNIMEM_API ?? 'http://localhost:8000'

// Served by FastAPI at /app in production; `pnpm dev` proxies the API so no CORS setup is needed.
export default defineConfig({
  base: '/app/',
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': path.resolve(import.meta.dirname, './src') } },
  server: { proxy: { '/api': api, '/health': api, '/metrics': api } },
  test: { environment: 'jsdom', setupFiles: ['./src/test/setup.ts'], include: ['src/**/*.test.{ts,tsx}'] },
})
