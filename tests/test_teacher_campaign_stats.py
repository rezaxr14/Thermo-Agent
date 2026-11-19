"""Counters, ingest grading, and append-only history for teacher_campaign_stats.

Does not touch the real journals or logs: everything lives in ``tmp_path``.  A
completion without a verdict is ungraded; an ingest verdict grades deferred rows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import teacher_campaign_stats as stats  # noqa: E402


AUTH_ERROR = (
    "ValueError: Missing environment variables for Kaggle authentication: "
    "MODEL_PROXY_URL / MODEL_PROXY_API_KEY"
)

DEFAULT_ROWS = [
    {  # (a) hosted task: explicitly deferred to ingest -> ungraded
        "id": "u1", "task": "audit", "model": "m2", "harness": "tools",
        "completion": "<answer>pending</answer>",
        "verdict": {"accepted": None, "graded": False, "reason": "deferred_to_ingest"},
        "ts": "2026-01-01T00:00:00Z",
    },
    {  # (b) no verdict and no explicit flag -> ungraded (never assume success)
        "id": "u2", "task": "repair", "model": "m1", "harness": "tools",
        "completion": "<answer>42</answer>",
        "ts": "2026-01-01T00:00:01Z",
    },
    {  # (c) explicitly graded + accepted, with usage and tool calls
        "id": "a1", "task": "repair", "model": "m1", "harness": "tools",
        "completion": "<edits>...</edits>",
        "verdict": {"accepted": True, "graded": True, "reasons": [], "failed": []},
        "sft_completion": "<trajectory>...</trajectory>",
        "reasoning": "let me check with a tool",
        "tools": ["solve_plant", "inspect_plant"],
        "tool_calls": [{"name": "solve_plant"}, {"name": "solve_plant"},
                       {"name": "inspect_plant"}],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_cost_nanodollars": 300,
                  "total_backend_latency_ms": 100},
        "ts": "2026-01-01T00:00:02Z",
    },
    {  # (d) rejected: an error always wins over the verdict
        "id": "r1", "task": "audit", "model": "m2", "harness": "tools",
        "error": AUTH_ERROR,
        "verdict": {"accepted": False, "graded": True, "reasons": ["verifier: bad plant"],
                    "failed": ["E_PROP_RANGE"]},
        "usage": {"input_tokens": 20, "output_tokens": 7, "total_cost_nanodollars": 700},
        "ts": "2026-01-01T00:00:03Z",
    },
]


def deferred_rows(count: int, *, model: str = "m1", task: str = "repair") -> list[dict]:
    return [
        {"id": f"g{i}", "task": task, "model": model, "completion": "<answer>pending</answer>",
         "verdict": {"accepted": None, "graded": False, "reason": "deferred_to_ingest"}}
        for i in range(1, count + 1)
    ]


def _seed(tmp_path: Path, *, rows: list[dict] | None = None,
          reports: dict[str, str] | None = None, log: str | None = None,
          generated_at: str = "2026-01-01T00:00:00Z"):
    runs = tmp_path / "data" / "teacher_runs"
    memory = tmp_path / "data" / "memory"
    logs = tmp_path / "logs"
    runs.mkdir(parents=True, exist_ok=True)
    memory.mkdir(parents=True, exist_ok=True)
    rows = DEFAULT_ROWS if rows is None else rows
    (runs / "tiny.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows) + "\n{bad json\n", encoding="utf-8")
    (memory / "failure_reasons.json").write_text(json.dumps({
        "n": 2,
        "top_lessons": [{"lesson": "Attempt rejected (format: unknown component ids ['DC']).", "count": 2}],
    }), encoding="utf-8")
    if reports:
        logs.mkdir(parents=True, exist_ok=True)
        for name, content in reports.items():
            (logs / name).write_text(content, encoding="utf-8")
    log_path = tmp_path / "docs" / "TEACHER_CAMPAIGN_LOG.md"
    if log is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(log, encoding="utf-8")

    rc = stats.main([
        "--root", str(tmp_path),
        "--runs-dir", str(runs),
        "--memory-file", str(memory / "failure_reasons.json"),
        "--stats-out", str(runs / "CAMPAIGN_STATS.json"),
        "--log-out", str(log_path),
        "--generated-at", generated_at,
    ])
    assert rc == 0
    data = json.loads((runs / "CAMPAIGN_STATS.json").read_text(encoding="utf-8"))
    return data, log_path.read_text(encoding="utf-8")


def test_three_buckets_graded_rate_and_histogram(tmp_path):
    data, md = _seed(tmp_path)

    t = data["totals"]
    assert (t["rows"], t["accepted"], t["rejected"], t["ungraded"]) == (4, 1, 1, 2)
    assert t["graded_rows"] == 2
    assert t["acceptance_rate"] == 0.5
    assert data["graded_rows"] == 2
    assert data["acceptance_rate"] == 0.5
    assert "ungraded" in data["acceptance_rate_denominator"]
    assert data["source"]["malformed_lines"] == 1

    assert data["per_model"]["m1"]["rows"] == 2
    assert data["per_model"]["m1"]["accepted"] == 1
    assert data["per_model"]["m1"]["ungraded"] == 1
    assert data["per_model"]["m1"]["acceptance_rate"] == 1.0
    assert data["per_model"]["m2"]["rejected"] == 1
    assert data["per_model"]["m2"]["acceptance_rate"] == 0.0
    assert data["per_task"]["repair"]["rows"] == 2
    assert data["per_task"]["audit"]["ungraded"] == 1
    assert data["per_model_task"]["m1"]["repair"]["rows"] == 2

    assert data["failure_reasons"] == {"auth:missing_proxy": 1}
    assert data["failure_reasons_raw"]["auth:missing_proxy"]["count"] == 1
    assert "MODEL_PROXY_URL" in json.dumps(data["failure_reasons_raw"])
    assert data["memory_failure_reasons"] == {"parse:unknown_id": 2}

    assert data["tool_call_counts"] == {"solve_plant": 2, "inspect_plant": 1}
    assert data["rows_with_tool_calls"] == 1
    assert data["rows_with_thinking"] == 1

    assert data["usage"]["input_tokens"] == 30
    assert data["usage"]["output_tokens"] == 12
    assert data["usage"]["total_tokens"] == 42
    assert data["usage"]["cost_nanodollars"] == 1000
    assert data["usage"]["cost_usd"] == 0.000001

    # MD: explicit denominator label + Ungraded / ingest columns, short classes only.
    assert "## Latest snapshot" in md
    assert "## History" in md
    assert "Ungraded" in md
    assert "Accept % (graded)" in md
    assert "Accepted (ingest)" in md
    assert "Rejected (ingest)" in md
    assert "accepted_total / (accepted_total + rejected_total)" in md
    assert "Ungraded rows today: **yes**" in md
    assert "| Ungraded | 2 |" in md
    assert "`auth:missing_proxy`" in md
    assert AUTH_ERROR not in md


def test_classify_row_precedence():
    c = stats.classify_row
    assert c({"error": "boom", "verdict": {"accepted": True, "graded": True}}) == stats.REJECTED
    assert c({"verdict": {"accepted": None, "graded": False}}) == stats.UNGRADED
    assert c({"verdict": {"accepted": None, "graded": True}}) == stats.UNGRADED
    assert c({"verdict": {"accepted": True, "graded": False}}) == stats.UNGRADED
    assert c({"verdict": {"accepted": True, "graded": True}}) == stats.ACCEPTED
    assert c({"verdict": {"accepted": False, "graded": True}}) == stats.REJECTED
    assert c({"accepted": True}) == stats.ACCEPTED
    assert c({"accepted": False}) == stats.REJECTED
    assert c({"accepted": None}) == stats.UNGRADED
    assert c({"completion": "looks fine"}) == stats.UNGRADED
    assert c({}) == stats.UNGRADED


def test_no_graded_rows_gives_null_rate(tmp_path):
    data = stats.build_stats(
        [{"id": "u1", "verdict": {"accepted": None, "graded": False}}],
        ["journal.jsonl"], 0, tmp_path / "missing.json", "2026-01-01T00:00:00Z", tmp_path,
    )
    assert (data["totals"]["rows"], data["totals"]["graded_rows"]) == (1, 0)
    assert data["totals"]["acceptance_rate"] is None
    assert data["acceptance_rate"] is None
    assert "n/a" in stats.snapshot_body(data, "2026-01-01T00:00:00Z")


def test_ingest_by_record_id_grades_deferred_rows(tmp_path):
    report = {
        "model": "m1", "attempts": 3, "accepted": 2, "rejected": 1,
        "stats": {"attempts": 3, "accepted:repair": 2, "rejected:repair": 1,
                  "reason:format": 1},
        "failure_reasons": {"n": 1, "by_task": {"repair": 1}, "by_fault_type": {},
                            "top_lessons": [{"lesson": "Attempt rejected (format: no XML block).", "count": 1}]},
        "accepted_ids": ["g1", "g2"],
        "rejected_ids": [{"id": "g3", "reason": "format: no XML block"}],
    }
    data, md = _seed(tmp_path, rows=deferred_rows(3),
                     reports={"ingest_campaign_m1_all.json": json.dumps(report)})

    t = data["totals"]
    assert (t["rows"], t["accepted"], t["rejected"], t["ungraded"]) == (3, 2, 1, 0)
    assert (t["graded_by_ingest"], t["accepted_ingest"], t["rejected_ingest"]) == (3, 2, 1)
    assert (data["graded_rows"], data["accepted_total"], data["rejected_total"]) == (3, 2, 1)
    assert data["acceptance_rate"] == round(2 / 3, 4)
    assert data["per_model"]["m1"]["acceptance_rate"] == round(2 / 3, 4)
    assert data["per_task"]["repair"]["acceptance_rate"] == round(2 / 3, 4)
    assert data["failure_reasons"] == {"parse:no_xml": 1}
    assert data["failure_reasons_raw"]["parse:no_xml"]["count"] == 1

    ingest = data["ingest_reports"]
    assert ingest["files_read"] == ["logs/ingest_campaign_m1_all.json"]
    assert ingest["files_skipped"] == []
    assert ingest["graded_by_ingest"] == 3
    assert ingest["matched_ids"] == 3
    assert ingest["model_task_counts"] == {"m1": {"repair": {"accepted": 2, "rejected": 1}}}
    assert "Accepted (ingest)" in md and "Rejected (ingest)" in md
    assert "| Accepted (ingest) | 2 |" in md
    assert "| Rejected (ingest) | 1 |" in md


def test_aggregate_ingest_report_attributes_by_model_and_task(tmp_path):
    # The canonical transfer.py::ingest schema is aggregate: no record ids.
    report = {
        "model": "m1", "attempts": 3, "accepted": 2, "rejected": 1,
        "stats": {"attempts": 3, "accepted:repair": 2, "rejected:repair": 1,
                  "reason:format": 1},
        "failure_reasons": {"n": 1, "by_task": {"repair": 1}, "top_lessons": []},
        "usage": {"models": {"m1": 3}},
    }
    data, _ = _seed(tmp_path, rows=deferred_rows(3),
                    reports={"ingest_aggregate.json": json.dumps(report)})
    t = data["totals"]
    assert (t["accepted"], t["rejected"], t["ungraded"], t["graded_by_ingest"]) == (2, 1, 0, 3)
    assert data["failure_reasons"] == {"parse:no_xml": 1}
    assert data["ingest_reports"]["per_id_decisions"] == 0


def test_malformed_report_is_skipped_without_crashing(tmp_path):
    good = {"model": "m1", "accepted": 1, "rejected": 0,
            "accepted_ids": ["g1"]}
    data, _ = _seed(tmp_path, rows=deferred_rows(2), reports={
        "ingest_bad.json": "{not valid json",
        "ingest_good.json": json.dumps(good),
    })
    assert (data["totals"]["accepted"], data["totals"]["ungraded"]) == (1, 1)
    assert data["ingest_reports"]["files_read"] == ["logs/ingest_good.json"]
    assert data["ingest_reports"]["files_skipped"] == ["logs/ingest_bad.json"]


def test_graded_true_journal_verdict_beats_conflicting_ingest(tmp_path):
    rows = [
        {"id": "g1", "task": "repair", "model": "m1", "completion": "ok",
         "verdict": {"accepted": True, "graded": True}},
        {"id": "g2", "task": "repair", "model": "m1", "completion": "pending",
         "verdict": {"accepted": None, "graded": False, "reason": "deferred_to_ingest"}},
    ]
    report = {"model": "m1", "accepted": 0, "rejected": 2,
              "rejected_ids": [{"id": "g1", "reason": "format: no XML block"},
                               {"id": "g2", "reason": "format: no XML block"}]}
    data, _ = _seed(tmp_path, rows=rows,
                    reports={"ingest_conflict.json": json.dumps(report)})
    t = data["totals"]
    assert (t["accepted"], t["rejected"], t["ungraded"]) == (1, 1, 0)
    assert (t["graded_by_ingest"], t["accepted_ingest"], t["rejected_ingest"]) == (1, 0, 1)
    assert data["failure_reasons"] == {"parse:no_xml": 1}


def test_previous_snapshot_is_moved_into_history_not_lost(tmp_path):
    previous = (
        "# Teacher campaign — live stats\n\n"
        "## Latest snapshot\n\n"
        "_Generated 2020-01-01T00:00:00Z from 1 journals (1 rows)._\n\n"
        "SENTINEL-PREVIOUS-SNAPSHOT\n\n"
        "## History\n\n"
        "SENTINEL-OLDER-HISTORY\n"
    )
    data, md = _seed(tmp_path, log=previous, generated_at="2026-02-02T00:00:00Z")
    latest, history = stats.split_sections(md)

    assert "2026-02-02T00:00:00Z" in latest
    assert "SENTINEL-PREVIOUS-SNAPSHOT" not in latest
    assert "SENTINEL-PREVIOUS-SNAPSHOT" in history
    assert "SENTINEL-OLDER-HISTORY" in history
    assert "2020-01-01T00:00:00Z" in history
    assert data["generated_at"] == "2026-02-02T00:00:00Z"
    assert md.count("## Latest snapshot") == 1
    assert md.count("## History") == 1


def test_repeated_runs_keep_each_snapshot(tmp_path):
    _, md1 = _seed(tmp_path, generated_at="2026-03-01T00:00:00Z")
    _, md2 = _seed(tmp_path, log=md1, generated_at="2026-03-02T00:00:00Z")

    latest, history = stats.split_sections(md2)
    assert "2026-03-02T00:00:00Z" in latest
    assert "2026-03-01T00:00:00Z" in history
    assert "2026-03-01T00:00:00Z" not in latest
    assert latest == stats.snapshot_body(
        json.loads((tmp_path / "data/teacher_runs/CAMPAIGN_STATS.json").read_text(encoding="utf-8")),
        "2026-03-02T00:00:00Z",
    )
