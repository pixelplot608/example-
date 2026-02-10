# Auth setup (Supabase Email/Password)

## 1. Enable Email auth in Supabase

1. Open **Supabase Dashboard** → your project → **Authentication** → **Providers**.
2. Enable **Email** and leave “Confirm email” off if you want immediate sign-in (or enable it and handle confirmation in your app).

## 2. Backend env

In `backend/.env` add (in addition to `DATABASE_URL`):

- `SUPABASE_JWT_SECRET` — from **Project Settings** → **API** → **JWT Secret**.

Optional: `SUPABASE_URL` (used only if you add more Supabase server-side features).

## 3. Frontend env

In `frontend/.env` add:

- `VITE_SUPABASE_URL` — Project URL from **Project Settings** → **API**.
- `VITE_SUPABASE_ANON_KEY` — anon/public key from the same page.

Restart the Vite dev server after changing `.env`.

## 4. Database: create profile on signup

Run the trigger in Supabase **SQL Editor** (once):

- File: `backend/supabase_auth_trigger.sql`

This creates a row in `public.profiles` with `role = 'viewer'` for every new user.

## 5. Set one admin (optional)

In Supabase SQL Editor:

```sql
UPDATE public.profiles SET role = 'admin' WHERE email = 'your@email.com';
```

## Flow

- **Sign up / Sign in**: frontend uses Supabase Auth (email + password); backend does not store passwords.
- **API calls**: frontend sends `Authorization: Bearer <access_token>`; backend verifies the JWT with `SUPABASE_JWT_SECRET` and uses `GET /auth/me` to return `user_id`, `email`, `role` from `profiles`.
