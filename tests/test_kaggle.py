"""Kaggle endpoint helpers must normalise URLs and read cloudflared output without a network."""
import pytest

from thermo_agent.teacher import kaggle


def test_normalize_adds_scheme_and_v1():
    assert kaggle.normalize_base_url("abc.trycloudflare.com") == "https://abc.trycloudflare.com/v1"
    assert kaggle.normalize_base_url("https://abc.trycloudflare.com/") == "https://abc.trycloudflare.com/v1"
    assert kaggle.normalize_base_url("https://abc.trycloudflare.com/v1") == "https://abc.trycloudflare.com/v1"
    assert kaggle.normalize_base_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000/v1"


def test_normalize_rejects_empty():
    for bad in ("", "   "):
        with pytest.raises(ValueError):
            kaggle.normalize_base_url(bad)


def test_tunnel_url_is_extracted_from_cloudflared_noise():
    line = ("2026-11-14T22:00:00Z INF |  https://quiet-river-1234.trycloudflare.com  |")
    assert kaggle.tunnel_url(line) == "https://quiet-river-1234.trycloudflare.com"
    assert kaggle.tunnel_url("no tunnel here") is None


def test_endpoint_from_env(monkeypatch):
    monkeypatch.delenv("TEACHER_BASE_URL", raising=False)
    assert kaggle.endpoint_from_env() is None
    monkeypatch.setenv("TEACHER_BASE_URL", "https://abc.trycloudflare.com")
    monkeypatch.setenv("TEACHER_MODEL", "teacher")
    assert kaggle.endpoint_from_env() == ("https://abc.trycloudflare.com/v1", "teacher")
