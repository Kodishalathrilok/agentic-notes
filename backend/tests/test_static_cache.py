"""Cache-Control on the static frontend, over real HTTP through the real app.

Starlette's StaticFiles sends ETag and Last-Modified but no Cache-Control, so
browsers cache heuristically and can keep running an old index.html after a
deploy. Such a client asks for lazy chunk names the server no longer has --
the QuizPanel-*.js / FlashcardPanel-*.js 404s seen in production, with the
entry bundle appearing to load fine because it came from cache.

These tests mount the production CachedStaticFiles over a directory shaped like
a real Vite build and assert the headers a browser would actually receive.
"""
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import main

# A miniature dist/: hashed output under assets/, public/ files passed through
# at the root, exactly as Vite lays it out.
HASHED_JS = "index-C5O09g9Z.js"
HASHED_CSS = "index-BAtH7Jvx.css"
QUIZ_CHUNK = "QuizPanel-Dpoo18Bd.js"
CARDS_CHUNK = "FlashcardPanel-D5cPLql-.js"
PUBLIC_IMG = "cloud-edge.png"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("dist")
    (root / "index.html").write_text(
        f'<!doctype html><script type="module" src="/assets/{HASHED_JS}"></script>',
        encoding="utf-8")
    (root / PUBLIC_IMG).write_bytes(b"\x89PNG\r\n\x1a\n fake")
    assets = root / "assets"
    assets.mkdir()
    for name in (HASHED_JS, QUIZ_CHUNK, CARDS_CHUNK):
        (assets / name).write_text("export default 1", encoding="utf-8")
    (assets / HASHED_CSS).write_text("body{}", encoding="utf-8")

    app = FastAPI()
    app.mount("/", main.CachedStaticFiles(directory=str(root), html=True),
              name="frontend")
    return TestClient(app)


def cc(response):
    return response.headers.get("cache-control")


# ---------------------------------------------------------------------------
# 1. the entry HTML must never pin a client to an old build
# ---------------------------------------------------------------------------

def test_index_html_is_never_cached(client):
    r = client.get("/")
    assert r.status_code == 200
    assert cc(r) == main.CACHE_ENTRY_HTML
    assert "no-store" in cc(r) and "must-revalidate" in cc(r)


def test_index_html_is_not_immutable(client):
    """The specific mistake that would recreate the bug."""
    assert "immutable" not in cc(client.get("/"))
    assert "immutable" not in cc(client.get("/index.html"))


def test_index_html_explicit_path_matches_root(client):
    assert cc(client.get("/index.html")) == cc(client.get("/"))


# ---------------------------------------------------------------------------
# 2-4. content-hashed assets are safe to keep
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [HASHED_JS, QUIZ_CHUNK, CARDS_CHUNK, HASHED_CSS])
def test_hashed_assets_are_immutable(client, name):
    r = client.get(f"/assets/{name}")
    assert r.status_code == 200
    assert cc(r) == main.CACHE_IMMUTABLE
    assert "immutable" in cc(r) and "max-age=31536000" in cc(r)


def test_lazy_chunks_still_load(client):
    """The two panels from the production report."""
    for name in (QUIZ_CHUNK, CARDS_CHUNK):
        r = client.get(f"/assets/{name}")
        assert r.status_code == 200
        assert r.text == "export default 1"


# ---------------------------------------------------------------------------
# 5. unhashed public/ files keep their name across builds -> must revalidate
# ---------------------------------------------------------------------------

def test_unhashed_public_file_revalidates(client):
    r = client.get(f"/{PUBLIC_IMG}")
    assert r.status_code == 200
    assert cc(r) == main.CACHE_REVALIDATE
    assert "immutable" not in cc(r), "a name that survives a rebuild cannot be immutable"


# ---------------------------------------------------------------------------
# 6. 404 behaviour is unchanged
# ---------------------------------------------------------------------------

def test_missing_asset_still_404s(client):
    """No SPA rewrite: a stale chunk name must 404, not silently return HTML."""
    r = client.get("/assets/QuizPanel-DEADBEEF.js")
    assert r.status_code == 404
    assert "<!doctype html" not in r.text.lower()


def test_missing_root_path_still_404s(client):
    assert client.get("/nope-does-not-exist.js").status_code == 404


# ---------------------------------------------------------------------------
# revalidation must not drop the header
# ---------------------------------------------------------------------------

def test_304_still_carries_cache_control(client):
    """A Not Modified response without Cache-Control puts the browser straight
    back on heuristics - the exact behaviour being fixed."""
    first = client.get(f"/assets/{QUIZ_CHUNK}")
    etag = first.headers["etag"]
    second = client.get(f"/assets/{QUIZ_CHUNK}", headers={"if-none-match": etag})
    assert second.status_code == 304
    assert cc(second) == main.CACHE_IMMUTABLE


def test_304_on_index_html_still_says_no_store(client):
    first = client.get("/index.html")
    etag = first.headers["etag"]
    second = client.get("/index.html", headers={"if-none-match": etag})
    assert second.status_code == 304
    assert cc(second) == main.CACHE_ENTRY_HTML


# ---------------------------------------------------------------------------
# the policy function itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,expected", [
    (os.path.join("static", "index.html"), main.CACHE_ENTRY_HTML),
    (os.path.join("static", "assets", HASHED_JS), main.CACHE_IMMUTABLE),
    (os.path.join("static", "assets", QUIZ_CHUNK), main.CACHE_IMMUTABLE),
    (os.path.join("static", "assets", CARDS_CHUNK), main.CACHE_IMMUTABLE),
    (os.path.join("static", "assets", HASHED_CSS), main.CACHE_IMMUTABLE),
    (os.path.join("static", PUBLIC_IMG), main.CACHE_REVALIDATE),
    (os.path.join("static", "hero-color.webp"), main.CACHE_REVALIDATE),
    (os.path.join("static", "assets", "notahash.js"), main.CACHE_REVALIDATE),
    (os.path.join("static", "robots.txt"), main.CACHE_REVALIDATE),
])
def test_cache_policy_by_filename(path, expected):
    assert main.cache_policy(path) == expected


def test_real_public_names_are_not_mistaken_for_hashed():
    """cloud-edge.png and hero-sketch.webp contain a dash; they are not hashed."""
    for name in ("cloud-edge.png", "hero-color.webp", "hero-sketch.webp"):
        assert main.cache_policy(os.path.join("static", name)) == main.CACHE_REVALIDATE
