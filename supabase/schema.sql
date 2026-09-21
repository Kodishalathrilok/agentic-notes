-- ---------------------------------------------------------------------------
-- Agentic Notes — `sessions` table and its access rules.
--
-- WHY THIS FILE EXISTS
--
-- The browser talks to this table directly with the public anon key (it ships
-- in the frontend bundle). Row Level Security is therefore the ONLY thing that
-- stands between any visitor and every user's notes. If RLS is off, or a
-- policy is written permissively, a plain `GET /rest/v1/sessions?select=*`
-- returns EVERY user's notes to ANYONE.
--
-- Through the table, you only ever see your own rows. Shared notes are NOT
-- readable through the table at all: an earlier "read public sessions" policy
-- (`using (is_public = true)`) let anyone holding the anon key LIST every
-- user's shared notes — full content plus user_id — because RLS cannot see a
-- query's WHERE clause, so "is_public rows" meant "all of them", not "the one
-- whose link you were given". A share link now resolves through
-- public.get_shared_session(id), which returns one row, by exact id, with
-- display columns only.
--
-- That boundary used to live only in the Supabase dashboard: not in the repo,
-- not reviewable, not testable, and impossible to rebuild if the project were
-- ever recreated. This file is the source of truth for it.
--
-- HOW TO RUN
--   Supabase dashboard -> SQL Editor -> paste -> Run.
--   Safe to re-run: everything is idempotent.
--
--   DEPLOY ORDER MATTERS on a project that already has the old policy. The
--   old frontend reads shared notes straight from the table; the new one
--   calls get_shared_session(). So:
--     1. Run STEP 1 (everything above the STEP 2 banner). Purely additive:
--        safe at any time, old share links keep working.
--     2. Deploy the new frontend.
--     3. Run STEP 2 (drops "read public sessions"). Until you do, the
--        enumeration hole is still open.
--   On a brand-new project, just run the whole file.
-- ---------------------------------------------------------------------------

-- ===========================================================================
-- STEP 1 — additive; safe to run at any time
-- ===========================================================================

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

-- ("read public sessions" is dropped in STEP 2, not here, so that re-running
-- STEP 1 on a live project never breaks share links for the old frontend.)
drop policy if exists "read own sessions"     on public.sessions;
drop policy if exists "insert own sessions"   on public.sessions;
drop policy if exists "update own sessions"   on public.sessions;
drop policy if exists "delete own sessions"   on public.sessions;
drop policy if exists "share reader sees public rows" on public.sessions;

-- SELECT — through the table, you see your own rows and nothing else.
-- Shared notes are read via get_shared_session() below, never via a policy:
-- a policy like `using (is_public)` cannot tell "fetch the row whose link I
-- was given" from "list every shared row", so it lets anyone enumerate.
create policy "read own sessions"
  on public.sessions for select
  to authenticated
  using (auth.uid() = user_id);

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
-- Share links: public.get_shared_session(share_id)
--
-- A share link only has to answer "give me THIS row, if its owner shared it".
-- This function answers exactly that and nothing more:
--   * one row, by exact uuid — no listing, no filtering, no search. Ids are
--     uuid v4 (~122 random bits), so they cannot be guessed or enumerated;
--     the link itself is the capability;
--   * only if is_public — revoking a share (is_public = false) kills the link;
--   * display columns only — NOT user_id, tags or is_public, so a viewer
--     learns nothing about who wrote it or what else they have.
--
-- Trade-off, stated honestly: Supabase's guidance is to keep SECURITY DEFINER
-- functions OUT of exposed schemas, because anything in `public` is callable
-- over the REST API (/rest/v1/rpc/...). This one is deliberately exposed —
-- anonymous viewers must be able to call it — so it is locked down instead:
-- the single-row exact-id lookup and narrow column list above, a pinned empty
-- search_path, and fully schema-qualified names (so nobody can shadow
-- `sessions` with an object of their own). Keep it that small.
--
-- LEAST PRIVILEGE: SECURITY DEFINER runs the function as its OWNER, so the
-- owner is the real boundary. It is NOT `postgres` (which may bypass RLS and
-- could then read every row) but a dedicated role, `share_reader`, that:
--   * cannot log in and does not bypass RLS;
--   * can SELECT only the display columns plus is_public — not user_id or
--     tags (column-level grant);
--   * sees only is_public rows, via its own RLS policy below.
-- So even a future bug in the WHERE clause could only ever return rows that
-- are already public, and never user_id. Private rows are out of its reach.
-- ---------------------------------------------------------------------------

-- CREATE ROLE is not idempotent on its own; guard it. NOBYPASSRLS is stated
-- explicitly even though it is the default (Verify #5 checks it).
do $$
begin
  if not exists (select 1 from pg_catalog.pg_roles where rolname = 'share_reader') then
    create role share_reader nologin nobypassrls noinherit;
  end if;
end
$$;

-- The role running this file must be a member of share_reader to hand it the
-- function (ALTER ... OWNER) and, on later re-runs, to `create or replace` a
-- function share_reader owns. Re-granting an existing membership is a no-op
-- (it only raises a NOTICE).
grant share_reader to current_user;

grant usage on schema public to share_reader;
revoke all on public.sessions from share_reader;
grant select (id, title, created_at, mode, notes, quiz, flashcards, sources, is_public)
  on public.sessions to share_reader;

-- The only rows share_reader can ever see. This policy is for share_reader
-- alone: anon/authenticated get no public-read policy on the table.
create policy "share reader sees public rows"
  on public.sessions for select
  to share_reader
  using (is_public);

create or replace function public.get_shared_session(share_id uuid)
returns table (
  id          uuid,
  title       text,
  created_at  timestamptz,
  mode        text,
  notes       text,
  quiz        text,
  flashcards  text,
  sources     jsonb
)
language sql
stable
security definer
set search_path = ''
as $$
  select s.id, s.title, s.created_at, s.mode,
         s.notes, s.quiz, s.flashcards, s.sources
    from public.sessions as s
   where s.id = share_id
     and s.is_public
$$;

-- Hand the function to share_reader. PostgreSQL requires the new owner to
-- have CREATE on the function's schema at the moment of the ALTER; it is
-- granted for that one statement and taken back — ownership stays.
grant create on schema public to share_reader;
alter function public.get_shared_session(uuid) owner to share_reader;
revoke create on schema public from share_reader;

-- New functions are executable by PUBLIC (every role) by default. Take that
-- away, then grant only the two API roles.
revoke all on function public.get_shared_session(uuid) from public;
grant execute on function public.get_shared_session(uuid) to anon, authenticated;

-- ===========================================================================
-- STEP 2 — run only AFTER the frontend that calls get_shared_session() is live
--
-- Removes the old policy that let anyone with the anon key list every shared
-- session. Not recreated. Old frontends' share links stop working here.
-- ===========================================================================

drop policy if exists "read public sessions" on public.sessions;

-- ---------------------------------------------------------------------------
-- Verify (run these after, and read the output — do not assume)
-- ---------------------------------------------------------------------------

-- 1. RLS must be ON. rowsecurity = true, or none of the above matters.
--    select relname, relrowsecurity, relforcerowsecurity
--      from pg_class where relname = 'sessions';

-- 2. Exactly the five policies above (read/insert/update/delete own, plus
--    "share reader sees public rows", whose roles must be {share_reader}
--    only), and no leftover permissive one. "read public sessions" must NOT
--    be listed.
--    Anything with qual = "true" or "is_public" for select is a hole.
--    select policyname, cmd, roles, qual, with_check
--      from pg_policies where tablename = 'sessions' order by cmd, policyname;

-- 3. Any pre-existing rows with no owner are invisible under these policies
--    and can never be read again. Check before assuming the table is clean:
--    select count(*) from public.sessions where user_id is null;

-- 4. The anon key cannot list sessions. Expect 0, even with shared rows:
--    begin;
--      set local role anon;
--      select count(*) from public.sessions;
--    rollback;
--    And from outside, with the anon key, expect `[]`:
--      GET /rest/v1/sessions?select=*&is_public=eq.true

-- 5. The share function: search_path pinned, no PUBLIC execute, owned by
--    share_reader, and share_reader is neither superuser nor BYPASSRLS.
--    select p.proconfig, p.proacl, r.rolname, r.rolsuper, r.rolbypassrls,
--           r.rolcanlogin
--      from pg_proc p join pg_roles r on r.oid = p.proowner
--     where p.oid = 'public.get_shared_session(uuid)'::regprocedure;
--    Expect proconfig = {search_path=""}, no "=X/" entry (PUBLIC) in proacl,
--    rolname = share_reader, and rolsuper / rolbypassrls / rolcanlogin all
--    false. share_reader must not be able to read user_id:
--    select has_column_privilege('share_reader', 'public.sessions', 'user_id', 'select');
--    -- expect false
--    Then, as anon, a shared id returns one row and an unshared id none:
--    begin;
--      set local role anon;
--      select * from public.get_shared_session('<a shared session id>');
--    rollback;
