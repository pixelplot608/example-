## Healthcare Access Inequality Analysis & Planning System

Web application for analysing healthcare access inequality, forecasting overload, and planning new facilities/mobile routes.

### 1. Project structure

- `backend/`
  - `app/main.py` – FastAPI app with demo in‑memory data (regions, facilities, demand, planning, scenarios, admin uploads, CSV export) and Supabase‑backed auth profile lookup.
  - `app/auth.py` – Supabase JWT verification (`get_current_user`, `get_current_user_optional`).
  - `app/db.py` – PostgreSQL connection helper (uses `DATABASE_URL`).
  - `requirements.txt` – backend dependencies.
  - `.env.example` – example env for DB + Supabase JWT secret.
- `frontend/`
  - Vite + TypeScript SPA (single‑page app).
  - `src/main.ts` – all UI logic (Dashboard, Access & Inequality, Demand & Overload, Facility Planning, Mobile Units, Scenarios, Reports, Admin, Help).
  - `src/supabase.ts` – Supabase JS client.
  - `.env.example` – example env for Supabase URL + anon key.

### 2. How to run locally

#### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then edit .env
# Set at least:
# - DATABASE_URL   (Supabase Postgres connection string)
# - SUPABASE_JWT_SECRET (from Supabase Project → Settings → API → JWT secret)

python -m uvicorn app.main:app --reload --port 8000
```

#### Frontend

```bash
cd frontend
npm install

cp .env.example .env            # then edit .env
# Set:
# - VITE_SUPABASE_URL
# - VITE_SUPABASE_ANON_KEY

npm run dev                     # opens http://localhost:5173
```

You should see the dashboard and be able to log in with a Supabase email/password account.

### 3. Supabase setup (once per project)

1. Create Supabase project.
2. In **Settings → API** copy:
   - Project URL → `VITE_SUPABASE_URL`
   - anon/public key → `VITE_SUPABASE_ANON_KEY`
   - JWT secret → `SUPABASE_JWT_SECRET`.
3. In **Authentication → Providers**, enable **Email**.
4. In Supabase SQL editor, run:
   - `backend/supabase_phases_7_8.sql` (schema for scenarios, profiles, etc.).
   - `backend/supabase_auth_trigger.sql` (creates `profiles` rows with role = `viewer` on new users).
5. Make one admin user manually:

```sql
UPDATE public.profiles
SET role = 'admin'
WHERE email = 'your-admin-email@example.com';
```

### 4. Current behaviour

- Backend serves demo data from memory so the app works even if Supabase tables are empty.
- Frontend uses Supabase Auth:
  - If there is no session → login/signup screen.
  - If session exists → calls `/auth/me` to get `{ user_id, email, role }` and then loads the app.
- All API calls attach `Authorization: Bearer <access_token>` when a session exists, but read‑only endpoints also work without auth (using `get_current_user_optional`).

### 5. Division of work

We have **two contributors**:

- **You + this AI (Cursor agent A)** – focus on backend logic, core algorithms, and cross‑page wiring.
- **Your friend + their Cursor (agent B)** – focus on data, content, and UI/UX polish.

#### Agent A (you + this AI) – responsibilities

- **Backend & APIs**
  - Keep `backend/app/main.py`, `auth.py`, `db.py` as the **single source of truth** for APIs.
  - Add/adjust endpoints when new frontend features are needed (access models, planning, routing, scenarios, audit).
  - Ensure endpoints never crash (handle DB errors, missing tables, bad params).
  - Introduce new optimization/AI logic (better planning, demand forecasting, equity metrics) behind existing routes.
- **Auth & roles**
  - Maintain Supabase JWT verification, `/auth/me`, and role‑based behaviour (`viewer`, `planner`, `admin`).
  - Apply role checks to sensitive endpoints (uploads, config) without breaking read‑only access.
- **Data/algorithm design**
  - Translate research papers/ideas into simplified algorithms (2SFCA, equity, planning, VRP) and expose via APIs that the frontend already uses.
- **Documentation for other agents**
  - Update this README when backend endpoints or contracts change so agent B always knows what is safe to call.

#### Agent B (friend + their AI) – responsibilities

- **Frontend UI/UX (React/Vite)**
  - Work only inside `frontend/`.
  - Improve layout, styling, and components without changing API URLs or response shapes.
  - Add new panels, filters, or visualisations that **consume existing endpoints**:
    - `/dashboard/summary`
    - `/regions`, `/facilities`
    - `/access/summary`, `/access/metrics`
    - `/demand/summary`, `/demand/timeseries/{id}`
    - `/planning/facility`, `/planning/mobile-routes`
    - `/scenarios`, `/scenarios/{id}`
    - `/admin/audit-logs`, `/admin/users`
    - `/export/*`
- **Data upload & examples**
  - Prepare example CSVs for regions/facilities/demand matching the Admin upload formats.
  - Document how non‑technical users can upload their own data.
- **User‑facing help & copy**
  - Improve text in Help/About page, tooltips, labels.
  - Add explanations of metrics (access score, Gini, overload, etc.) using content only (no backend changes).

### 6. Collaboration rules (for both A and B)

- **Do not rename or remove API routes** that `frontend/src/main.ts` already calls.
- When adding new features:
  - Agent A adds backend route + updates README “API” section.
  - Agent B then consumes that route in frontend.
- Prefer **additive** changes:
  - New functions/files/components instead of refactoring everything at once.
- Keep env‑specific values **only in `.env` files**, never hard‑coded into source.

This README is the contract both Cursor agents should follow when working on this project.

