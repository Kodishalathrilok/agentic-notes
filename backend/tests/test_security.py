"""
Server-hardening tests: SSRF guard, URL scheme checks, and request size limits.
No network access needed — private/loopback addresses are validated locally.
"""

import pytest
from fastapi import HTTPException

import main


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "host",
    ["localhost", "127.0.0.1", "169.254.169.254", "10.0.0.5", "192.168.1.1", "172.16.0.9", "0.0.0.0"],
)
def test_private_hosts_blocked(host):
    assert main._host_resolves_public(host) is False


def test_public_ip_allowed():
    assert main._host_resolves_public("8.8.8.8") is True


def test_non_http_scheme_rejected():
    with pytest.raises(HTTPException) as exc:
        main._fetch_url_safely("ftp://example.com/file")
    assert exc.value.status_code == 422


def test_loopback_url_rejected():
    with pytest.raises(HTTPException) as exc:
        main._fetch_url_safely("http://127.0.0.1:8000/api/health")
    assert exc.value.status_code == 422


def test_metadata_endpoint_rejected():
    with pytest.raises(HTTPException) as exc:
        main._fetch_url_safely("http://169.254.169.254/latest/meta-data/")
    assert exc.value.status_code == 422


# ---------------------------------------------------------------------------
# Request size limits (Pydantic max_length)
# ---------------------------------------------------------------------------

def test_chat_request_rejects_oversized_question():
    with pytest.raises(Exception):
        main.ChatRequest(question="x" * 5000)


def test_chat_request_rejects_oversized_notes():
    with pytest.raises(Exception):
        main.ChatRequest(notes="x" * 300001)


def test_url_request_rejects_oversized_url():
    with pytest.raises(Exception):
        main.UrlRequest(url="https://example.com/" + "a" * 2000)


def test_normal_sizes_accepted():
    main.ChatRequest(notes="some notes", question="what is this?")
    main.UrlRequest(url="https://example.com/article")
    main.RegenRequest(notes="notes body")