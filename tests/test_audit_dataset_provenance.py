"""Tests for the per-row provenance/tier auditor."""
import json

import scripts.audit_dataset_provenance as audit


def _row(path, source, model=None, tier="easy", task="repair"):
    row = {"id": "x", "task": task, "source": source, "tier": tier}
    if source == "frontier_tools":
        row["provenance"] = {"generator": "frontier_tools", "model": model}
        if model:
            row["model"] = model
    elif source == "deterministic_tools":
        row["provenance"] = {"generator": "deterministic_tools", "variant": "direct"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_accepts_labelled_rows(tmp_path):
    d = tmp_path / "corpus"
    _row(d / "repair_train.jsonl", "deterministic_tools")
    _row(d / "repair_frontier.jsonl", "frontier_tools", model="gpt-6-luna", tier="hard")
    report = audit.audit_dirs([str(d)])
    assert report["passed"] is True
    assert report["by_source"] == {"deterministic_tools": 1, "frontier_tools": 1}
    assert report["by_model"]["gpt-6-luna"] == 1
    assert report["by_model"]["none (deterministic)"] == 1
    assert report["by_task_tier"]["repair"] == {"easy": 1, "hard": 1}


def test_rejects_ai_row_without_model(tmp_path):
    d = tmp_path / "corpus"
    _row(d / "repair_train.jsonl", "frontier_tools", model=None)
    report = audit.audit_dirs([str(d)])
    assert report["passed"] is False
    assert any("model id" in p["error"] for p in report["problems"])


def test_rejects_missing_tier_and_unknown_source(tmp_path):
    d = tmp_path / "corpus"
    _row(d / "repair_train.jsonl", "deterministic_tools", tier=None)
    _row(d / "repair_train2.jsonl", "magic")
    report = audit.audit_dirs([str(d)])
    assert report["passed"] is False
    errors = " ".join(p["error"] for p in report["problems"])
    assert "tier" in errors and "source" in errors


def test_audits_nested_directories(tmp_path):
    d = tmp_path / "corpus"
    _row(d / "sub" / "repair_train.jsonl", "ground_truth", tier="medium")
    report = audit.audit_dirs([str(d)])
    assert report["passed"] is True
    assert report["rows"] == 1
