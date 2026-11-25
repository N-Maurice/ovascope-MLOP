# OvaScope web interface

React + TypeScript single-page app for the OvaScope API. It runs on laptops and phones (responsive layout,
bottom tab bar on small screens, camera capture on mobile) and has light and dark themes.

## Pages

| Route | What it shows | API endpoints |
|---|---|---|
| `/` | overview, live benchmark numbers, per-class F1 | `/results/evaluation`, `/dataset/summary`, `/gate`, `/classes/{id}/examples/{n}` |
| `/classify` | upload or camera, square crop around one egg, top-3 result, egg check, reference and look-alike crops, session history | `/predict` |
| `/classes` | the 11 classes with images, features, metrics and test-set mix-ups | `/classes`, `/results/evaluation`, `/dataset/summary` |
| `/dataset` | split and class balance, audit checks, crop sizes, camera formats | `/dataset/summary` |
| `/performance` | 7 metrics (val vs test), generalisation gap, accuracy by camera, model comparison, confusion matrix, per-class table, training curves per version, egg-check results | `/results/evaluation`, `/results/history/{version}`, `/model/versions`, `/gate` |
| `/status` | live health, uptime, request counts, latency, recent requests, retraining readiness (refreshes every 5 s) | `/health`, `/metrics`, `/retrain/status` |
| `/about` | method, limitations, API usage | – |
| `/sign-in` | sign in with username or email + password, create an account, or use the operator admin key | `/auth/login`, `/auth/register`, `/auth/status`, `/auth/verify` |
| `/account` | profile, role, change password, sign out | `/auth/me`, `/auth/password`, `/auth/logout` |
| `/admin` (admins only) | upload labelled images or class-folder ZIPs; database records (with uploader); start and follow retraining; activate or roll back model versions; manage users (promote, deactivate, delete) | `/upload`, `/uploads`, `/retrain`, `/retrain/jobs`, `/retrain/status/{id}`, `/model/*`, `/users` |

All data is fetched live through React Query, which polls health, metrics, jobs and uploads in the background so
every page stays in sync with the API, the SQLite database and the model registry. After an upload, a retraining
job or a model switch, the affected data is refreshed immediately.

## Run locally

```bash
# terminal 1: the API (from the project root)
OVASCOPE_ADMIN_KEY=choose-a-key uvicorn api.main:app --port 8000

# terminal 2: the interface
cd web
npm install
npm run dev            # http://localhost:5173  (phones on the same network: npm run dev -- --host)
```

The dev server forwards `/api/*` to `http://localhost:8000`; set `OVASCOPE_API_URL` to use another API address.
Open `/sign-in` → **Create account**. The first account becomes the administrator; later accounts start as normal
users until an admin promotes them in **Admin → Users**. "Keep me signed in" stores the session token in
localStorage (7 days); otherwise it lasts until the tab is closed. Or start both servers at once: `bash scripts/dev.sh`.

## Production

```bash
npm run build          # static files in web/dist
npm run preview        # serve the build locally on :4173 (same /api proxy)
```

With Docker, `docker compose up -d --build` from the project root also starts the `web` service on
**http://localhost:8080**: nginx serves the build and forwards `/api/*` to the API gateway.

## Structure

```
src/
  api/        client.ts (fetch wrapper, admin key), hooks.ts (React Query), types.ts (API shapes)
  components/ Layout (navigation), ConfusionMatrix, shared UI, toasts
  pages/      one file per page; admin/ holds the four admin panels
  lib/        formatting, class descriptions and colours, in-memory classification history
  styles.css  design tokens (light/dark) and components
```
