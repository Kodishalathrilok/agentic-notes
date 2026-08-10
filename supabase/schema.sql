-- ---------------------------------------------------------------------------
-- Agentic Notes — `sessions` table and its access rules.
--
-- WHY THIS FILE EXISTS
--
-- The frontend looks up "the session with this id" without filtering by user
-- (the share lookup in App.jsx). That is only safe because Row Level Security
-- makes Postgres apply the filter itself, per caller. If RLS is off, or a
-- policy is written permissively, that query returns EVERY user's notes to
-- ANYONE who is signed in.
--
-- Note the asymmetry in the SELECT policies below: "read public sessions"
-- intentionally lets any caller read an is_public row, which means an
-- unfiltered `select *` returns your own rows PLUS everybody's shared ones.
-- History therefore filters on user_id itself (useHistory.js) — RLS is the
-- boundary, but it is not by itself a per-user listing.
--
-- That boundary used to live only in the Supabase dashboard: not in the repo,
-- not reviewable, not testable, and impossible to rebuild if the project were
-- ever recreated. This file is the source of truth for it.
--
-- HOW TO RUN
--   Supabase dashboard -> SQL Editor -> paste -> Run.
--   Safe to re-run: everything is idempotent.
-- ---------------------------------------------------------------------------

create extension if not exists "pgcrypto";

create table if not exists public.sessions (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null references auth.users (id) on delete cascade
                default auth.uid(),
  created_at  timestamptz not null default now(),
  title       text        not null default '',
  mode        text,
  tags        jsonb       not null default '[]'::jsonb,
  notes       text        not null default '',
  quiz        text        not null default '',
  flashcards  text        not null default '',
  sources     jsonb       not null default '[]'::jsonb,
  is_public   boolean     not null default false
);

-- The client never sends user_id; the column default fills it from the caller's
-- JWT. Existing tables may predate that, so make sure it is set either way.
alter table public.sessions
  alter column user_id set default auth.uid();

-- History is listed newest-first per user; share links look up by id.
create index if not exists sessions_user_created_idx
  on public.sessions (user_id, created_at desc);

-- ---------------------------------------------------------------------------
-- Row Level Security
--
-- Without `enable row level security`, policies below are inert and the table
-- is world-readable to any signed-in user. This line is the whole boundary.
-- ---------------------------------------------------------------------------

alter table public.sessions enable row level security;

-- Belt and braces: even the table owner goes through the policies.
alter table public.sessions force row level security;

drop policy if exists "read own sessions"     on public.sessions;
drop policy if exists "read public sessions"  on public.sessions;
drop policy if exists "insert own sessions"   on public.sessions;
drop policy if exists "update own sessions"   on public.sessions;
drop policy if exists "delete own sessions"   on public.sessions;

-- SELECT — you see your own rows...
create policy "read own sessions"
  on public.sessions for select
  to authenticated
  using (auth.uid() = user_id);

-- ...and anyone, signed in or not, may read a row explicitly shared.
-- Scoped to is_public so an unlisted id is not enough on its own.
create policy "read public sessions"
  on public.sessions for select
  to anon, authenticated
  using (is_public = true);

-- INSERT — you may only create rows owned by you. Without the WITH CHECK a
-- caller could hand-craft a request that writes rows under someone else's id.
create policy "insert own sessions"
  on public.sessions for insert
  to authenticated
  with check (auth.uid() = user_id);

-- UPDATE — USING picks which rows you may target, WITH CHECK stops you
-- reassigning one to another user on the way out. Both are required.
create policy "update own sessions"
  on public.sessions for update
  to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy "delete own sessions"
  on public.sessions for delete
  to authenticated
  using (auth.uid() = user_id);

-- ---------------------------------------------------------------------------
-- Verify (run these after, and read the output — do not assume)
-- ---------------------------------------------------------------------------

-- 1. RLS must be ON. rowsecurity = true, or none of the above matters.
--    select relname, relrowsecurity, relforcerowsecurity
--      from pg_class where relname = 'sessions';

-- 2. Exactly the five policies above, and no leftover permissive one.
--    Anything with qual = "true" for select is a hole.
--    select policyname, cmd, roles, qual, with_check
--      from pg_policies where tablename = 'sessions' order by cmd, policyname;

-- 3. Any pre-existing rows with no owner are invisible under these policies
--    and can never be read again. Check before assuming the table is clean:
--    select count(*) from public.sessions where user_id is null;
