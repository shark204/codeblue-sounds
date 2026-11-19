import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    persistSession: true,      // ✅ يحفظ الجلسة في localStorage عشان ميسجلش دخول كل مرة
    autoRefreshToken: true,    // ✅ يجدد التوكن تلقائياً قبل ما ينتهي
    detectSessionInUrl: true,  // ✅ يقرأ التوكن من الرابط (مهم لـ Password Reset و Magic Link)
    storage: typeof window !== 'undefined' ? window.localStorage : undefined, // ✅ حماية للـ SSR
  }
})