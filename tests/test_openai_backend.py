"""The OpenAI-compatible backend in eval/model_eval.py is tested against a local stub server, so no model or GPU is needed."""
from __future__ import annotations
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from thermo_agent.eval.model_eval import OpenAIGenerator


class _Stub(BaseHTTPRequestHandler):
    payloads: list[dict] = []

    def log_message(self, *args):
        pass

    def _send(self, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"data": [{"id": "stub-model"}]})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        _Stub.payloads.append(json.loads(self.rfile.read(n) or b"{}"))
        self._send({"choices": [{"message": {"content": "<think>reasoning...</think> FIX T0 = 298.15"}}]})


def _serve():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_resolves_model_strips_thinking_and_is_greedy():
    srv = _serve()
    try:
        gen = OpenAIGenerator(f"http://127.0.0.1:{srv.server_port}/v1", timeout=30)
        out = gen([{"role": "user", "content": "fix the ledger"}])
    finally:
        srv.shutdown()
    assert out == "FIX T0 = 298.15"
    req = _Stub.payloads[-1]
    assert req["model"] == "stub-model"
    assert req["temperature"] == 0.0
    assert req["max_tokens"] == 400
    assert req["chat_template_kwargs"] == {"enable_thinking": False}
    assert req["messages"][0]["content"] == "fix the ledger"


def test_explicit_model_is_used_and_string_prompt_is_wrapped():
    srv = _serve()
    try:
        gen = OpenAIGenerator(f"http://127.0.0.1:{srv.server_port}/v1", model="my-model", max_new_tokens=64, timeout=30)
        gen("bare prompt")
    finally:
        srv.shutdown()
    req = _Stub.payloads[-1]
    assert req["model"] == "my-model"
    assert req["max_tokens"] == 64
    assert req["messages"] == [{"role": "user", "content": "bare prompt"}]
