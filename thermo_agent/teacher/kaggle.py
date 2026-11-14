"""Helpers for pointing the teacher sampler at a Kaggle-hosted endpoint.

Kaggle does not expose a general inference API; it gives GPU notebooks. The teacher therefore runs in a Kaggle
notebook (see `kaggle/serve_teacher.py`), serves an OpenAI-compatible API with vLLM, and publishes a
cloudflared quick-tunnel URL. This module normalises that URL and checks the endpoint before a long sampling
run starts.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

_TUNNEL = re.compile(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com")
_BASE = re.compile(r"^https?://")


def normalize_base_url(url: str) -> str:
    """Accept `host`, `https://host`, `https://host/` or `https://host/v1` and return the `/v1` root."""
    u = (url or "").strip()
    if not u:
        raise ValueError("empty base url")
    if not _BASE.match(u):
        u = "https://" + u
    u = u.rstrip("/")
    return u if u.endswith("/v1") else u + "/v1"


def tunnel_url(text: str) -> str | None:
    """Pull the quick-tunnel URL out of cloudflared's stderr."""
    m = _TUNNEL.search(text or "")
    return m.group(0) if m else None


def models(base_url: str, timeout: int = 20) -> list[str]:
    """Model ids served by the endpoint; raises if it is not reachable."""
    url = normalize_base_url(base_url) + "/models"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return [m.get("id", "") for m in body.get("data", [])]


def endpoint_from_env() -> tuple[str, str] | None:
    """(base_url, model) from TEACHER_BASE_URL / TEACHER_MODEL, or None when the teacher is not configured."""
    base = os.environ.get("TEACHER_BASE_URL", "").strip()
    model = os.environ.get("TEACHER_MODEL", "").strip()
    if not base:
        return None
    return normalize_base_url(base), model or "teacher"


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", help="base URL to check for /v1/models")
    ap.add_argument("--normalize", help="print the normalised /v1 root for this URL")
    a = ap.parse_args()
    if a.normalize:
        print(normalize_base_url(a.normalize))
        return
    if a.probe:
        print(json.dumps({"base_url": normalize_base_url(a.probe), "models": models(a.probe)}))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
