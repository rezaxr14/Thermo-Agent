"""Serve a frontier teacher from a Kaggle notebook (2xT4) and print a public base URL.

Kaggle gives GPU notebooks, not an inference API. This script runs vLLM's OpenAI-compatible server on the two
T4s and puts a cloudflared quick tunnel in front of it, so the laptop can point `teacher.sample` at it.

Notebook setup (Accelerator: GPU T4 x2, Internet: on):

    !pip install -q vllm
    !python serve_teacher.py

Copy the printed `TEACHER_BASE_URL=...` line. Keep the notebook running while sampling; closing it kills the
tunnel and the server.

Environment overrides:
    TEACHER_MODEL     default Qwen/Qwen2.5-32B-Instruct-AWQ   (AWQ fits 2xT4; fp16 32B does not)
    TEACHER_TP        default 2       tensor-parallel ranks, one per T4
    TEACHER_MAX_LEN   default 8192    prompts are ~3.5k tokens plus an 800-token completion
    TEACHER_GPU_UTIL  default 0.90
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

MODEL = os.environ.get("TEACHER_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ")
PORT = int(os.environ.get("TEACHER_PORT", "8000"))
TP = os.environ.get("TEACHER_TP", "2")
MAX_LEN = os.environ.get("TEACHER_MAX_LEN", "8192")
GPU_UTIL = os.environ.get("TEACHER_GPU_UTIL", "0.90")
CLOUDFLARED = "/tmp/cloudflared"
TUNNEL = re.compile(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com")


def _wait_models(url: str, timeout: int = 2400) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url + "/models", timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            time.sleep(5)
    raise SystemExit("vLLM did not answer /v1/models; check the notebook log for an OOM")


def _ensure_cloudflared() -> str:
    if shutil.which("cloudflared"):
        return "cloudflared"
    if not os.path.exists(CLOUDFLARED):
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        urllib.request.urlretrieve(url, CLOUDFLARED)
        os.chmod(CLOUDFLARED, 0o755)
    return CLOUDFLARED


def main() -> None:
    try:
        import vllm  # noqa: F401
    except ImportError:
        raise SystemExit("vLLM is not installed in this notebook: run  !pip install -q vllm  first")
    print(f"model={MODEL} tp={TP} max_len={MAX_LEN} gpu_util={GPU_UTIL}", flush=True)
    server = subprocess.Popen([
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL, "--served-model-name", "teacher", "--port", str(PORT),
        "--tensor-parallel-size", TP, "--dtype", "float16", "--max-model-len", MAX_LEN,
        "--gpu-memory-utilization", GPU_UTIL, "--disable-log-requests",
    ])
    try:
        served = _wait_models(f"http://127.0.0.1:{PORT}/v1")
        print("served models:", [m.get("id") for m in served.get("data", [])], flush=True)
        tunnel = subprocess.Popen([_ensure_cloudflared(), "tunnel", "--url", f"http://127.0.0.1:{PORT}",
                                   "--no-autoupdate"], stderr=subprocess.PIPE, text=True)
        for line in tunnel.stderr:
            m = TUNNEL.search(line)
            if m:
                print(f"TEACHER_BASE_URL={m.group(0)}", flush=True)
                print("TEACHER_MODEL=teacher", flush=True)
                break
        tunnel.wait()
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
