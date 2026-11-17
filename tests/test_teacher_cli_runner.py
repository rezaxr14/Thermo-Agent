"""The CLI teacher must journal every attempt, resume by id and keep costs/quota explicit."""
import json
import pathlib
import types

from thermo_agent.teacher import cli_runner as cr


def test_dotenv_reads_env_file(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# comment\nKAGGLE_API_TOKEN=abc\nHF_TOKEN=\"xyz\"\n", encoding="utf-8")
    assert cr.dotenv(p) == {"KAGGLE_API_TOKEN": "abc", "HF_TOKEN": "xyz"}


def test_quota_parses_money(monkeypatch):
    out = types.SimpleNamespace(returncode=0, stderr="", stdout=json.dumps([
        {"period": "Daily", "used": "$1.25", "remaining": "$8.75", "total": "$10.00"},
        {"period": "Monthly", "used": "$1.25", "remaining": "$98.75", "total": "$100.00"}]))
    monkeypatch.setattr(cr.subprocess, "run", lambda *a, **k: out)
    q = cr.quota()
    assert q["daily"]["remaining"] == 8.75 and q["monthly"]["total"] == 100.0


def test_usage_extracts_cost_and_finish_reason():
    raw = {"choices": [{"finish_reason": "length", "message": {"content": "x"}}],
           "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                     "cost": {"input_tokens_cost_nanodollars": 100, "output_tokens_cost_nanodollars": 250}}}
    u = cr._usage(raw)
    assert u["input_tokens"] == 10 and u["total_cost_nanodollars"] == 350
    assert u["finish_reason"] == "length"


def test_journal_append_and_resume_filter(tmp_path, monkeypatch):
    out = tmp_path / "completions.jsonl"
    cr.append(out, [{"id": "a", "completion": "ok"}])
    cr.append(out, [{"id": "b", "error": "quota"}])
    assert len(cr.journal(out)) == 2
    done = {r["id"] for r in cr.journal(out) if r.get("completion") and not r.get("error")}
    assert done == {"a"}


def test_load_rows_reads_all_tasks(tmp_path):
    for task in ("repair", "audit"):
        (tmp_path / f"{task}_train.jsonl").write_text(
            json.dumps({"id": f"{task}-1", "prompt": [{"role": "user", "content": "hi"}]}) + "\n", encoding="utf-8")
    rows = cr.load_rows(tmp_path, ("repair", "audit"))
    assert {r["task"] for r in rows} == {"repair", "audit"}


def test_one_uses_the_model_and_records_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "chat", lambda *a, **k: {"completion": "answer", "model": a[0], "usage": {}})
    r = cr._one({"id": "x", "task": "repair", "prompt": []}, "m", 100)
    assert r["id"] == "x" and r["model"] == "m" and r["completion"] == "answer" and r["ts"]
