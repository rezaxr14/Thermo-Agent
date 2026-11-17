"""Teacher progress must survive a dead Kaggle session and a new account, with costs and failures recorded."""
import json
import pathlib
import sys
import types

from thermo_agent.teacher import progress


class FakeApi:
    def __init__(self):
        self.files = {}
        self.uploads = []

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, token=None, commit_message=None):
        data = path_or_fileobj.read() if hasattr(path_or_fileobj, "read") else pathlib.Path(path_or_fileobj).read_bytes()
        self.files[path_in_repo] = data
        self.uploads.append(path_in_repo)


def test_dedupe_keeps_distinct_attempts_but_not_exact_retries():
    a = {"id": "r1", "completion": "good"}
    b = {"id": "r1", "error": "quota"}
    assert progress.dedupe([a, a, b, b]) == [a, b]
    assert progress.answered_ids([a, b]) == {"r1"}
    assert progress.answered_ids([b]) == set()


def test_sync_merges_local_and_remote_and_pushes_the_union(tmp_path, monkeypatch):
    api = FakeApi()
    api.files[progress.DEFAULT_PREFIX] = b'{"id":"remote","completion":"answer","model":"m"}\n'
    monkeypatch.setattr(progress, "pull", lambda *a, **k: [
        json.loads(l) for l in api.files[progress.DEFAULT_PREFIX].decode().splitlines()])
    local = tmp_path / "completions.jsonl"
    local.write_text('{"id":"local","completion":"answer"}\n', encoding="utf-8")
    merged = progress.sync(local, "Reza-Nadimi/thermo-agent-checkpoints", "hf_x", api=api)
    assert {r["id"] for r in merged} == {"remote", "local"}
    assert json.loads(local.read_text().splitlines()[0])["id"] in {"remote", "local"}
    assert progress.DEFAULT_PREFIX in api.uploads
    assert api.files[progress.DEFAULT_PREFIX].count(b"\n") == 2


def test_usage_totals_records_nanodollars_and_models():
    records = [
        {"id": "a", "model": "m1", "usage": {"input_tokens": 10, "output_tokens": 5,
                                              "total_cost_nanodollars": 300, "total_backend_latency_ms": 100}},
        {"id": "b", "model": "m2", "usage": {"total_cost_nanodollars": 700}},
    ]
    out = progress.usage_totals(records)
    assert out["cost_nanodollars"] == 1000 and out["cost_usd"] == 1e-6
    assert out["input_tokens"] == 10 and out["models"] == {"m1": 1, "m2": 1}


def test_load_missing_is_empty(tmp_path):
    assert progress.load(tmp_path / "missing.jsonl") == []
