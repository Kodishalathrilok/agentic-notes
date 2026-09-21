"""Row Level Security on `sessions`, proven against a real PostgreSQL.

These tests run the ACTUAL supabase/schema.sql (not a copy of its SQL) inside
a throwaway Postgres 16 started by `pgserver`, on top of a minimal
Supabase-like harness:

  * roles `anon` and `authenticated` (NOLOGIN), granted to the connecting
    superuser so a test can `set local role` to act as them;
  * an `auth` schema with `auth.users` and `auth.uid()` reading the
    `request.jwt.claim.sub` setting (Supabase's documented test pattern);
  * Supabase's default grants: table and function privileges in `public`
    for anon/authenticated, so RLS alone is what restricts them;
  * schema.sql is applied by `db_admin`, a NON-superuser with CREATEROLE
    that owns the database — the closest stand-in for Supabase's `postgres`
    role. (A superuser bypasses RLS and every privilege check, which would
    hide exactly the mistakes these tests exist to catch.)

The module is skipped when pgserver / psycopg are not installed.
"""
import uuid
from pathlib import Path

import pytest

pgserver = pytest.importorskip("pgserver")
psycopg = pytest.importorskip("psycopg")

SCHEMA_SQL = (Path(__file__).resolve().parents[2] / "supabase" / "schema.sql").read_text(encoding="utf-8")

# The one line of schema.sql this harness cannot run as-is: pgserver's
# Postgres build ships without contrib, so `pgcrypto` is unavailable. The
# schema only needs it for gen_random_uuid(), which is built into core
# Postgres since 13, so skipping the extension changes nothing tested here.
# (Supabase has pgcrypto installed, so the line is kept in the real file.)
_PGCRYPTO = 'create extension if not exists "pgcrypto";'
assert SCHEMA_SQL.count(_PGCRYPTO) == 1, "schema.sql changed; update the harness"
SCHEMA_SQL = SCHEMA_SQL.replace(_PGCRYPTO, "-- (pgcrypto skipped by the test harness)")

# Roles are cluster-wide; created before the database so db_admin can own it.
ROLES_SQL = """
do $$ begin
  if not exists (select 1 from pg_roles where rolname = 'db_admin') then
    create role db_admin nologin nosuperuser createrole nobypassrls;
  end if;
  if not exists (select 1 from pg_roles where rolname = 'anon') then
    create role anon nologin noinherit;
  end if;
  if not exists (select 1 from pg_roles where rolname = 'authenticated') then
    create role authenticated nologin noinherit;
  end if;
end $$;
grant anon, authenticated, db_admin to current_user;
"""

HARNESS_SQL = """

create schema if not exists auth;
create table if not exists auth.users (id uuid primary key);
create or replace function auth.uid() returns uuid
  language sql stable
  as $$ select nullif(current_setting('request.jwt.claim.sub', true), '')::uuid $$;
grant usage on schema auth to anon, authenticated;
grant execute on function auth.uid() to anon, authenticated;
grant usage on schema auth to db_admin;
grant select, references on auth.users to db_admin;
grant execute on function auth.uid() to db_admin;

-- Supabase's defaults: the API roles get broad privileges in `public`;
-- RLS policies are what actually narrow them down.
grant usage on schema public to anon, authenticated;
alter default privileges for role db_admin in schema public
  grant select, insert, update, delete on tables to anon, authenticated;
alter default privileges for role db_admin in schema public
  grant execute on functions to anon, authenticated;
"""

USER_A = uuid.UUID("aaaaaaaa-0000-4000-8000-000000000001")
USER_B = uuid.UUID("bbbbbbbb-0000-4000-8000-000000000002")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    srv = pgserver.get_server(tmp_path_factory.mktemp("pgdata"), cleanup_mode="stop")
    yield srv
    srv.cleanup()


def _fresh_db(server, name):
    with psycopg.connect(server.get_uri(), autocommit=True) as admin:
        admin.execute(ROLES_SQL)
        admin.execute(f'drop database if exists "{name}"')
        admin.execute(f'create database "{name}" owner db_admin')
    conn = psycopg.connect(server.get_uri(name), autocommit=True)
    conn.execute(HARNESS_SQL)
    return conn


def _apply_schema(conn):
    """Run the real schema.sql as db_admin (non-superuser, CREATEROLE)."""
    conn.execute("set role db_admin")
    try:
        conn.execute(SCHEMA_SQL)
    finally:
        conn.execute("reset role")


@pytest.fixture(scope="module")
def db(server):
    conn = _fresh_db(server, "rls_sharing")
    _apply_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def tx(db):
    """A transaction that is always rolled back, seeded with two users.

    A has one private and one shared session; B the same.
    """
    with db.transaction(force_rollback=True):
        cur = db.cursor()
        cur.execute("insert into auth.users (id) values (%s), (%s)", (USER_A, USER_B))
        ids = {}
        for owner, tag in ((USER_A, "a"), (USER_B, "b")):
            for public in (False, True):
                cur.execute(
                    "insert into public.sessions (user_id, title, notes, is_public)"
                    " values (%s, %s, %s, %s) returning id",
                    (owner, f"{tag}-{'shared' if public else 'private'}", f"notes of {tag}", public),
                )
                ids[(tag, public)] = cur.fetchone()[0]
        yield cur, ids


def as_anon(cur):
    cur.execute("select set_config('request.jwt.claim.sub', '', true)")
    cur.execute("set local role anon")


def as_user(cur, user_id):
    cur.execute("select set_config('request.jwt.claim.sub', %s, true)", (str(user_id),))
    cur.execute("set local role authenticated")


def test_anon_cannot_list_shared_sessions(tx):
    cur, _ = tx
    as_anon(cur)
    cur.execute("select id, user_id, title from public.sessions")
    rows = cur.fetchall()
    assert rows == [], f"anon enumerated sessions via the table: {rows}"
    cur.execute("select count(*) from public.sessions where is_public")
    assert cur.fetchone()[0] == 0


def _shared(cur, share_id):
    cur.execute("select * from public.get_shared_session(%s)", (share_id,))
    cols = [d.name for d in cur.description]
    return cols, cur.fetchall()


def test_anon_reads_one_shared_session_without_owner(tx):
    cur, ids = tx
    as_anon(cur)
    cols, rows = _shared(cur, ids[("b", True)])
    assert len(rows) == 1
    row = dict(zip(cols, rows[0]))
    assert row["id"] == ids[("b", True)]
    assert row["title"] == "b-shared"
    assert row["notes"] == "notes of b"
    for hidden in ("user_id", "tags", "is_public"):
        assert hidden not in cols


def test_anon_gets_nothing_for_unshared_or_unknown_id(tx):
    cur, ids = tx
    as_anon(cur)
    assert _shared(cur, ids[("b", False)])[1] == []
    assert _shared(cur, uuid.uuid4())[1] == []


def test_signed_in_user_is_confined_to_own_rows(tx):
    cur, ids = tx
    as_user(cur, USER_A)

    # Only A's rows, even though B has a shared one.
    cur.execute("select title from public.sessions order by title")
    assert [r[0] for r in cur.fetchall()] == ["a-private", "a-shared"]

    # B's rows (shared or not) can be neither updated nor deleted.
    for b_id in (ids[("b", False)], ids[("b", True)]):
        cur.execute("update public.sessions set title = 'pwned' where id = %s", (b_id,))
        assert cur.rowcount == 0
        cur.execute("delete from public.sessions where id = %s", (b_id,))
        assert cur.rowcount == 0

    # Nor can A reassign an own row to B.
    cur.execute("savepoint s")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        cur.execute(
            "update public.sessions set user_id = %s where id = %s", (USER_B, ids[("a", False)])
        )
    cur.execute("rollback to savepoint s")

    # Nor insert a row owned by B.
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        cur.execute("insert into public.sessions (user_id, title) values (%s, 'forged')", (USER_B,))
    cur.execute("rollback to savepoint s")

    # Nothing of B's changed.
    cur.execute("reset role")
    cur.execute("select count(*) from public.sessions where user_id = %s and title like 'b-%%'", (USER_B,))
    assert cur.fetchone()[0] == 2


def test_unsharing_revokes_the_link(tx):
    cur, ids = tx
    a_shared = ids[("a", True)]
    as_anon(cur)
    assert len(_shared(cur, a_shared)[1]) == 1

    as_user(cur, USER_A)
    cur.execute("update public.sessions set is_public = false where id = %s", (a_shared,))
    assert cur.rowcount == 1

    as_anon(cur)
    assert _shared(cur, a_shared)[1] == []


def test_share_function_is_locked_down(tx):
    cur, _ = tx
    cur.execute(
        "select proconfig, prosecdef from pg_proc"
        " where oid = 'public.get_shared_session(uuid)'::regprocedure"
    )
    proconfig, prosecdef = cur.fetchone()
    assert prosecdef is True
    assert proconfig == ['search_path=""']

    # grantee 0 in aclexplode() is PUBLIC.
    cur.execute(
        "select a.grantee::regrole::text, a.privilege_type from pg_proc p,"
        " aclexplode(p.proacl) a where p.oid = 'public.get_shared_session(uuid)'::regprocedure"
        " and a.privilege_type = 'EXECUTE'"
    )
    grantees = {g for g, _ in cur.fetchall()}
    assert "-" not in grantees and "0" not in grantees, grantees  # PUBLIC
    assert {"anon", "authenticated"} <= grantees


def test_share_function_is_owned_by_least_privilege_role(tx):
    cur, _ = tx
    cur.execute(
        "select r.rolname, r.rolsuper, r.rolbypassrls, r.rolcanlogin"
        " from pg_proc p join pg_roles r on r.oid = p.proowner"
        " where p.oid = 'public.get_shared_session(uuid)'::regprocedure"
    )
    assert cur.fetchone() == ("share_reader", False, False, False)
    for col, allowed in (("user_id", False), ("tags", False), ("notes", True), ("is_public", True)):
        cur.execute(
            "select has_column_privilege('share_reader', 'public.sessions', %s, 'select')", (col,)
        )
        assert cur.fetchone()[0] is allowed, col
    for priv in ("insert", "update", "delete"):
        cur.execute("select has_table_privilege('share_reader', 'public.sessions', %s)", (priv,))
        assert cur.fetchone()[0] is False, priv


def test_share_reader_cannot_read_user_id(tx):
    cur, _ = tx
    cur.execute("set local role share_reader")
    cur.execute("savepoint s")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        cur.execute("select user_id from public.sessions")
    cur.execute("rollback to savepoint s")
    # Even as share_reader, only shared rows are visible (its RLS policy).
    cur.execute("select title from public.sessions order by title")
    assert [r[0] for r in cur.fetchall()] == ["a-shared", "b-shared"]


def test_broken_where_clause_still_cannot_leak_private_rows(tx):
    """Replace the body with one that forgets `and s.is_public` — and even the
    id filter — as the schema's owner would. The owner role's RLS policy and
    column grants must still keep private rows out."""
    cur, ids = tx
    cur.execute("set local role db_admin")
    cur.execute(
        """
        create or replace function public.get_shared_session(share_id uuid)
        returns table (id uuid, title text, created_at timestamptz, mode text,
                       notes text, quiz text, flashcards text, sources jsonb)
        language sql stable security definer set search_path = ''
        as $$
          select s.id, s.title, s.created_at, s.mode,
                 s.notes, s.quiz, s.flashcards, s.sources
            from public.sessions as s
           where s.id = share_id or true
        $$
        """
    )
    cur.execute(
        "select pg_get_userbyid(proowner) from pg_proc"
        " where oid = 'public.get_shared_session(uuid)'::regprocedure"
    )
    assert cur.fetchone()[0] == "share_reader"  # replace keeps the owner

    as_anon(cur)
    cols, rows = _shared(cur, ids[("a", False)])
    titles = sorted(dict(zip(cols, r))["title"] for r in rows)
    assert titles == ["a-shared", "b-shared"]  # only already-public rows


def test_schema_sql_is_idempotent(server):
    conn = _fresh_db(server, "rls_idempotent")
    try:
        _apply_schema(conn)
        # An existing project still carries the old enumerable policy; a re-run
        # must remove it (STEP 2) and not bring it back.
        conn.execute(
            'create policy "read public sessions" on public.sessions'
            " for select to anon, authenticated using (is_public = true)"
        )
        _apply_schema(conn)
        _apply_schema(conn)
        cur = conn.execute(
            "select policyname from pg_policies where tablename = 'sessions' order by policyname"
        )
        assert [r[0] for r in cur.fetchall()] == [
            "delete own sessions",
            "insert own sessions",
            "read own sessions",
            "share reader sees public rows",
            "update own sessions",
        ]
    finally:
        conn.close()
