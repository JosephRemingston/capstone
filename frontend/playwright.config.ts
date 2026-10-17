import os from 'node:os'
import path from 'node:path'
import { defineConfig, devices } from '@playwright/test'

// One fresh store per run, outside the repository.
const store = path.join(os.tmpdir(), `cognimem-e2e-${process.pid}`)
const port = Number(process.env.E2E_PORT ?? 8765)
const python = process.env.COGNIMEM_PYTHON ?? path.join(import.meta.dirname, '..', '.venv', 'bin', 'python')

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  retries: 0,
  use: { baseURL: `http://127.0.0.1:${port}`, trace: 'retain-on-failure' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  // Serves the built app at /app from the real API (run `pnpm build` first). The heuristic scorer needs no model files.
  webServer: {
    command: `"${python}" -m uvicorn api.app:app --host 127.0.0.1 --port ${port}`,
    cwd: path.join(import.meta.dirname, '..'),
    url: `http://127.0.0.1:${port}/health/live`,
    reuseExistingServer: false,
    timeout: 120_000,
    env: {
      COGNIMEM_JWT_SECRET: 'e2e-secret-e2e-secret-e2e-secret-e2e-secret',
      COGNIMEM_STORE: path.join(store, 'memories.jsonl'),
      COGNIMEM_AUTH_DB: path.join(store, 'auth.sqlite3'),
      COGNIMEM_SCORER: 'heuristic',
      COGNIMEM_LOG_LEVEL: 'WARNING',
    },
  },
})
