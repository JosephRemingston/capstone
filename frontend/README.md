# CogniMem web app

React + TypeScript client for the CogniMem HTTP API (`api/`). It holds no business logic: every number it shows (tiers, expiry, importance parts, retrieval scores, citations, graph relations) comes from the API.

| Command | What it does |
| --- | --- |
| `pnpm install` | Install dependencies (Node 20+, pnpm 10). |
| `pnpm dev` | Vite dev server at `http://localhost:5173/app/`, proxying `/api`, `/health` and `/metrics` to `COGNIMEM_API` (default `http://localhost:8000`). Start the API first. |
| `pnpm build` | Type-check and build to `dist/`, which the API serves at `/app` (override the folder with `COGNIMEM_WEB_DIR`). |
| `pnpm typecheck` / `pnpm lint` / `pnpm test` | TypeScript, oxlint, Vitest unit tests. |
| `pnpm e2e` | Playwright smoke test: starts the API on port 8765 with a temporary store and the heuristic scorer, then drives the built app. Run `pnpm build` first; install the browser once with `pnpm exec playwright install chromium`. |

Screens: Chat, Memories, Memory detail, Search & Ask, Graph, Evaluation and System. The Evaluation page reads the tracked snapshots through `/api/v1/evaluation/*`; its per-layer metric list lives in `src/lib/evaluation.ts`, and `src/lib/evaluation.test.ts` checks every metric against the real `evaluation/reports` files and the README's numbers.

Layout: `src/api` (typed client and TanStack Query hooks), `src/auth` (session in `sessionStorage`; a 401 or token expiry signs out, as the API has no refresh endpoint), `src/components` (shadcn/ui in `ui/`, memory and search widgets), `src/pages` (one per screen), `e2e/` (Playwright).

Privacy: answers are extractive (local) by default. Turning on Gemini on the Ask tab asks for confirmation first, because the retrieved memories are then sent to Google.
