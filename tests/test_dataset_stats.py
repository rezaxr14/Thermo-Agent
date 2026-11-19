"""Streaming dataset statistics: counts, tool tags, determinism, no text leakage.

Everything lives in ``tmp_path``; the real corpora and the real tokenizer are
never touched.  Three tiny jsonl directories stand in for the six datasets the
production command reads.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import dataset_stats as ds  # noqa: E402


ALPHA_SECRET = "SECRET_GROUND_TRUTH_ALPHA_123"
BETA_SECRET = "SECRET_GROUND_TRUTH_BETA_456"
GAMMA_SECRET = "SECRET_GROUND_TRUTH_GAMMA_789"
ENV_SECRET = "SUPER_SECRET_ENV_TOKEN_999"


def _messages(text: str) -> list[dict]:
    return [{"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user prompt"},
            {"role": "assistant", "content": text}]


def _seed(tmp_path: Path) -> tuple[Path, Path]:
    """Create three tiny dirs, including a .env decoy that must never be read."""
    base = tmp_path / "data"
    alpha = base / "alpha"
    beta = base / "beta"
    gamma = base / "gamma"
    for d in (alpha, beta, gamma):
        d.mkdir(parents=True, exist_ok=True)

    alpha_rows = [
        {
            "id": "a1", "task": "repair", "split": "train", "source": "deterministic_tools",
            "harness": "none", "tier": "easy", "topology": "cgam_orc",
            "prompt": _messages("repair prompt one"),
            "completion": _messages(
                "<think>look</think><tool_call name=\"inspect_plant\">{}</tool_call>"
                "<tool_result>ok</tool_result><diagnosis>" + ALPHA_SECRET + "</diagnosis>"),
        },
        {
            "id": "a2", "task": "repair", "split": "val", "source": "deterministic_tools",
            "harness": "skills", "topology": "cgam_orc",
            "prompt": _messages("repair prompt two"),
            "completion": _messages("<diagnosis>plain answer</diagnosis>"),
        },
    ]
    beta_rows = [
        {  # tool tags live in the trajectory, not in the stored completion
            "id": "b1", "task": "design", "split": "test_id", "source": "frontier_tools",
            "harness": "agents+skills",
            "prompt": _messages("design prompt"),
            "completion": _messages("<design>" + BETA_SECRET + "</design>"),
            "trajectory": {
                "final": "<design>done</design>",
                "sft_completion": "<tool_call name=\"solve_plant\">{}</tool_call>"
                                  "<tool_result>valid</tool_result>",
                "turns": [{"assistant": "", "tool_calls": [{"name": "solve_plant"}]}],
            },
        },
    ]
    gamma_rows = [
        {  # held-out style: no task/source/harness; task comes from the filename
            "id": "g1", "split": "test_id", "fault_types": ["missing_ech"],
            "prompt": _messages("ledger prompt"),
            "completion": _messages("<diagnosis>" + GAMMA_SECRET + "</diagnosis>"),
        },
    ]

    (alpha / "repair_train.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in alpha_rows), encoding="utf-8")
    (beta / "frontier_design_train.jsonl").write_text(
        json.dumps(beta_rows[0]) + "\n", encoding="utf-8")
    (gamma / "ledger_test_id.jsonl").write_text(
        json.dumps(gamma_rows[0]) + "\n", encoding="utf-8")
    (alpha / ".env").write_text(f"API_KEY={ENV_SECRET}\n", encoding="utf-8")
    (alpha / "notes.txt").write_text(ALPHA_SECRET + "\n", encoding="utf-8")

    out = tmp_path / "out" / "dataset_stats.json"
    md = tmp_path / "out" / "dataset_stats.md"
    return out, md


def _run(tmp_path: Path, out: Path, md: Path) -> int:
    return ds.main([
        "--root", str(tmp_path),
        "--dirs", "data/alpha", "data/beta", "data/gamma",
        "--out", str(out), "--markdown", str(md),
        "--no-tokens",
    ])


def _outputs(out: Path, md: Path) -> tuple[str, str, dict]:
    json_text = out.read_text(encoding="utf-8")
    md_text = md.read_text(encoding="utf-8")
    return json_text, md_text, json.loads(json_text)


def test_counts_tasks_splits_and_field_values(tmp_path):
    out, md = _seed(tmp_path)
    assert _run(tmp_path, out, md) == 0
    _, _, data = _outputs(out, md)

    assert data["totals"]["rows"] == 4
    assert data["totals"]["malformed_lines"] == 0
    assert sorted(data["dirs"]) == ["data/alpha", "data/beta", "data/gamma"]

    alpha = data["dirs"]["data/alpha"]
    assert alpha["rows"] == 2 and alpha["files"] == 1
    assert alpha["per_task"]["repair"]["rows"] == 2
    assert alpha["per_split"] == {"train": 1, "val": 1}
    assert alpha["per_tier"] == {"easy": 1, "<missing>": 1}
    assert alpha["harness_values"] == {"none": 1, "skills": 1}
    assert alpha["source_values"] == {"deterministic_tools": 2}
    assert data["dirs"]["data/beta"]["per_task"]["design"]["rows"] == 1
    assert data["dirs"]["data/beta"]["per_split"] == {"test_id": 1}
    assert data["dirs"]["data/beta"]["source_values"] == {"frontier_tools": 1}

    # No task key on held-out rows: inferred from ledger_test_id.jsonl.
    gamma = data["dirs"]["data/gamma"]
    assert gamma["per_task"]["ledger"]["rows"] == 1
    assert gamma["source_values"] == {"<missing>": 1}
    assert gamma["harness_values"] == {"<missing>": 1}

    for key in ("generated_from", "dirs", "totals", "tokenizer"):
        assert key in data
    assert set(data["generated_from"]) == {
        "data/alpha/repair_train.jsonl",
        "data/beta/frontier_design_train.jsonl",
        "data/gamma/ledger_test_id.jsonl",
    }
    assert all(len(v["sha256"]) == 64 for v in data["generated_from"].values())
    assert data["tokenizer"]["status"] == "not measured"
    assert data["totals"]["chars"]["prompt"]["n"] == 4
    assert data["totals"]["tokens"]["prompt"] is None


def test_tool_tags_completion_and_trajectory(tmp_path):
    out, md = _seed(tmp_path)
    assert _run(tmp_path, out, md) == 0
    _, _, data = _outputs(out, md)

    a = data["dirs"]["data/alpha"]["per_task"]["repair"]["flags"]
    assert a == {"with_stored_completion": 2, "with_trajectory": 0,
                 "with_tool_call": 1, "with_tool_result": 1}

    b = data["dirs"]["data/beta"]["per_task"]["design"]["flags"]
    assert b == {"with_stored_completion": 1, "with_trajectory": 1,
                 "with_tool_call": 1, "with_tool_result": 1}

    g = data["dirs"]["data/gamma"]["per_task"]["ledger"]["flags"]
    assert g["with_tool_call"] == 0 and g["with_tool_result"] == 0
    assert g["with_trajectory"] == 0 and g["with_stored_completion"] == 1


def test_deterministic_output_and_no_text_or_secrets_leak(tmp_path):
    out, md = _seed(tmp_path)
    assert _run(tmp_path, out, md) == 0
    json1, md1, _ = _outputs(out, md)
    assert _run(tmp_path, out, md) == 0
    json2, md2, _ = _outputs(out, md)

    assert json1 == json2
    assert md1 == md2
    assert json1 == json.dumps(json.loads(json1), indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n"

    combined = json1 + md1
    for secret in (ALPHA_SECRET, BETA_SECRET, GAMMA_SECRET, ENV_SECRET):
        assert secret not in combined
    assert "<diagnosis>" not in combined and "plain answer" not in combined

    # Markdown table is sorted by dir then task, with an ALL row per dir first.
    rows = [line for line in md1.splitlines() if line.startswith("| data/")]
    dirs = [line.split("|")[1].strip() for line in rows]
    assert dirs == sorted(dirs)


def test_quantiles_are_order_independent():
    values = [1, 2, 3, 4, 100]
    stats = ds.length_stats(values)
    assert stats == {"n": 5, "min": 1, "median": 3, "p95": 81, "max": 100}
    assert ds.length_stats(list(reversed(values))) == stats
    assert ds.length_stats([]) == {"n": 0, "min": None, "median": None,
                                   "p95": None, "max": None}


def test_discover_files_skips_env_and_non_jsonl(tmp_path):
    base = tmp_path / "data" / "alpha"
    base.mkdir(parents=True)
    (base / "repair_train.jsonl").write_text("", encoding="utf-8")
    (base / ".env").write_text("API_KEY=x\n", encoding="utf-8")
    (base / "notes.txt").write_text("x", encoding="utf-8")
    files = ds.discover_files(tmp_path, ["data/alpha"])
    assert [p.name for _, p in files] == ["repair_train.jsonl"]
