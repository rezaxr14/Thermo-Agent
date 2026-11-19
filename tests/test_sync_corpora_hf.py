"""The corpora mirror must refuse foreign repos/paths, filter held-out answers, and leave a manifest.

All tests are offline: main() is driven with --dry-run, so no token is read and no network call is made.
"""
import hashlib
import json
import sys

import pytest

import scripts.sync_corpora_hf as sync


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "sft_tools_v2").mkdir(parents=True)
    (tmp_path / "data" / "heldout_v2").mkdir(parents=True)
    return tmp_path


def _argv(*args):
    return ["sync_corpora_hf.py", *args]


def test_foreign_or_local_repo_id_is_refused_before_any_call(workspace, monkeypatch):
    assert sync.resolve_repo(sync.PROJECT_REPO) == "Reza-Nadimi/thermo-agent-corpora"
    for bad in ("someone-else/thermo-agent-corpora", "Reza-Nadimi/thermo-agent-checkpoints", "org/repo"):
        with pytest.raises(ValueError, match="refusing HF dataset"):
            sync.resolve_repo(bad)
    # Even a full dry-run CLI must refuse a foreign repo before token/network setup.
    monkeypatch.setattr(sys, "argv", _argv("--repo", "someone-else/thermo-agent-corpora", "--dry-run"))
    with pytest.raises(SystemExit, match="refusing HF dataset"):
        sync.main()


def test_path_outside_allow_list_is_refused(workspace):
    (workspace / "data" / "teacher_runs").mkdir(parents=True)
    (workspace / "data" / "sft_tools").mkdir(parents=True)  # legacy dir, no longer allow-listed
    (workspace / ".env").mkdir()  # a dir named .env must still be refused, not silently included
    for bad in ("data/teacher_runs", ".env", str(workspace / "data" / "teacher_runs"),
                "data/../.env", "data/sft_tools"):
        with pytest.raises(SystemExit, match="refusing to upload"):
            sync.check_dirs([bad])
    # An allow-listed dir is accepted.
    assert sync.check_dirs(["data/sft_tools_v2"]) == [sync.Path("data/sft_tools_v2")]


def test_heldout_answers_reports_and_verdicts_are_filtered(workspace):
    secret_row = {
        "id": "r1", "split": "test_id", "prompt": [
            {"role": "user", "content": "PROMPT TEXT", "verdict": "NESTED SECRET"},
        ],
        "completion": [{"role": "assistant", "content": "SECRET ANSWER"}],
        "answer": "SECRET ANSWER", "verdict": {"accepted": True, "reason": "SECRET VERDICT"},
        "report": "SECRET REPORT", "plant_original": "SECRET SOLUTION",
        "plant_broken": "INPUT PLANT", "site": "INPUT SITE",
    }
    projected = sync.filter_heldout_row(secret_row)
    assert projected == {"id": "r1", "split": "test_id", "prompt": [
        {"role": "user", "content": "PROMPT TEXT"},
    ]}
    assert set(projected).issubset(sync.HELDOUT_KEEP_FIELDS)

    f = workspace / "data" / "heldout_v2" / "repair_test_id.jsonl"
    f.write_text(json.dumps(secret_row) + "\n", encoding="utf-8")
    transformed = sync.transform_bytes(f, f.read_bytes()).decode("utf-8")
    for token in ("SECRET ANSWER", "SECRET VERDICT", "SECRET REPORT", "SECRET SOLUTION",
                  "NESTED SECRET", "plant_original", "completion"):
        assert token not in transformed
    assert "PROMPT TEXT" in transformed

    # A non-held-out corpus is uploaded byte-for-byte (no filtering, no surprises).
    other = workspace / "data" / "sft_tools_v2" / "repair_train.jsonl"
    other.write_text('{"completion": "KEEP ME"}\n', encoding="utf-8")
    assert sync.transform_bytes(other, other.read_bytes()) == other.read_bytes()


def test_dry_run_produces_an_auditable_manifest(workspace, monkeypatch, capsys):
    (workspace / "data" / "sft_tools_v2" / "repair_train.jsonl").write_text(
        '{"id":"s1","prompt":[{"role":"user","content":"hi"}]}\n', encoding="utf-8")
    (workspace / "data" / "heldout_v2" / "repair_test_id.jsonl").write_text(
        '{"id":"h1","prompt":[{"role":"user","content":"hi"}],"completion":"LEAK"}\n', encoding="utf-8")
    out = workspace / "manifest.json"
    monkeypatch.setattr(sys, "argv", _argv(
        "--dry-run", "--message", "v2 rebuild", "--tag", "2026-11-19", "--manifest-out", str(out)))
    assert sync.main() == 0
    assert "nothing was sent" in capsys.readouterr().out

    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["repo_id"] == "Reza-Nadimi/thermo-agent-corpora"
    assert manifest["message"] == "v2 rebuild" and manifest["tag"] == "2026-11-19"
    assert manifest["mode"] == "dry-run" and manifest["additive_only"] is True
    assert manifest["heldout_prompts_only"] is True
    assert manifest["timestamp_utc"].endswith("Z")
    assert manifest["file_count"] == 2
    assert set(manifest["dirs"]) == {"data/sft_tools_v2", "data/heldout_v2"}
    assert all({"path", "sha256", "size_bytes", "local_path"} <= set(f) for f in manifest["files"])

    heldout = next(f for f in manifest["files"] if "/heldout_v2/" in f["path"])
    raw = (workspace / "data" / "heldout_v2" / "repair_test_id.jsonl").read_bytes()
    assert heldout["sha256"] != hashlib.sha256(raw).hexdigest(), "manifest must hash the filtered bytes"
    assert heldout["sha256"] == hashlib.sha256(
        sync.transform_bytes(workspace / "data" / "heldout_v2" / "repair_test_id.jsonl", raw)).hexdigest()
