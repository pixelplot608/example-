-- Run this in Supabase SQL Editor (Phases 7 & 8).
-- Creates: scenarios, scenario results, audit_logs, config, profiles (for auth).

-- Scenarios (Phase 7)
CREATE TABLE IF NOT EXISTS public.scenarios (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  description TEXT,
  type TEXT NOT NULL CHECK (type IN ('facility_plan', 'mobile_routes', 'combined')),
  created_by UUID,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  inputs JSONB,
  results_summary JSONB
);

CREATE TABLE IF NOT EXISTS public.scenario_facility_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  scenario_id UUID NOT NULL REFERENCES public.scenarios(id) ON DELETE CASCADE,
  region_id TEXT NOT NULL,
  region_name TEXT,
  centroid_lat DOUBLE PRECISION,
  centroid_lon DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS public.scenario_route_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  scenario_id UUID NOT NULL REFERENCES public.scenarios(id) ON DELETE CASCADE,
  vehicle_id INT NOT NULL,
  stop_sequence INT NOT NULL,
  region_id TEXT NOT NULL,
  region_name TEXT,
  latitude DOUBLE PRECISION,
  longitude DOUBLE PRECISION,
  leg_distance_km DOUBLE PRECISION
);

-- Audit & config (Phase 8)
CREATE TABLE IF NOT EXISTS public.audit_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID,
  action_type TEXT NOT NULL,
  details JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.config (
  key TEXT PRIMARY KEY,
  value JSONB
);

-- Insert default config
INSERT INTO public.config (key, value) VALUES
  ('poor_access_threshold_km', '15'),
  ('overload_utilization_threshold', '1.0')
ON CONFLICT (key) DO NOTHING;

-- Profiles for auth (Phase 8). Link to Supabase auth.users when you enable Auth.
CREATE TABLE IF NOT EXISTS public.profiles (
  id UUID PRIMARY KEY,
  email TEXT,
  role TEXT NOT NULL DEFAULT 'viewer' CHECK (role IN ('viewer', 'planner', 'admin'))
);

-- Allow anonymous read for now; RLS can be added later.
ALTER TABLE public.scenarios ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.audit_logs ENABLE ROW LEVEL SECURITY;

-- Policies: allow all for demo (you can tighten later)
CREATE POLICY "Allow all scenarios" ON public.scenarios FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all profiles" ON public.profiles FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all audit_logs" ON public.audit_logs FOR ALL USING (true) WITH CHECK (true);
