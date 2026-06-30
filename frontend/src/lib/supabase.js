import { createClient } from '@supabase/supabase-js'

const url = import.meta.env.VITE_SUPABASE_URL
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

// Only create a client if both env vars are configured. When unset, the app
// runs in local-only mode (history in localStorage, no auth).
export const supabase = url && anonKey ? createClient(url, anonKey) : null
export const supabaseEnabled = !!supabase
