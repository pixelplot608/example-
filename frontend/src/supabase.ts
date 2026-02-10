import { createClient } from '@supabase/supabase-js'

const url = import.meta.env.VITE_SUPABASE_URL
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

if (!url || !anonKey) {
  console.warn('VITE_SUPABASE_URL or VITE_SUPABASE_ANON_KEY missing. Auth will not work.')
}

export const supabase = createClient(url || '', anonKey || '')
