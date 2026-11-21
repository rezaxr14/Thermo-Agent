"""Regression tests for the hosted teacher prompt shards.

The hosted campaign must receive TOOL-READY inputs (plant/ledger/site), not just
prompt text. Before the fix the shards lacked those columns, so ``make_tools``
had nothing to inspect and every hosted row made zero tool calls. These tests
build the dataset into a temp dir and prove a real ``inspect_plant("")`` call
works for every task, while keeping all answers out of the upload.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_hosted_prompts.py"
EXPORT = ROOT / "data" / "frontier_export"

pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(
        not all((EXPORT / f"{t}_train.jsonl").exists()
                for t in ("repair", "audit", "ledger", "design")),
        reason="data/frontier_export is not present in this clone",
    ),
]


def _load_builder():
    spec = importlib.util.spec_from_file_location("_make_hosted_prompts", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Run the real builder into a temp directory once for this module."""
    builder = _load_builder()
    out = tmp_path_factory.mktemp("hosted_prompts")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "answer_leak_check: PASS" in proc.stdout
    return builder, out, proc.stdout


def _all_rows(out: pathlib.Path) -> list[dict]:
    rows = []
    for i in range(5):
        rows.extend(_read_jsonl(out / f"prompts_s{i:02d}.jsonl"))
    return rows


def _expected_counts(builder) -> dict:
    """The builder projects every export row; derive the target from the source."""
    return {t: len(_read_jsonl(EXPORT / f"{t}_train.jsonl")) for t in builder.TASK_ORDER}


def test_builder_counts_allowlist_and_no_answers(built):
    builder, out, _ = built
    rows = _all_rows(out)
    expected = _expected_counts(builder)
    assert len(rows) == sum(expected.values())

    counts = {t: 0 for t in builder.TASK_ORDER}
    for row in rows:
        task = row["task"]
        assert task in builder.FIELD_ALLOWLIST
        counts[task] += 1
        assert set(row) == set(builder.FIELD_ALLOWLIST[task])
        assert not (set(row) & set(builder.FORBIDDEN_KEYS))
        assert isinstance(row["prompt"], list) and row["prompt"]
        assert all(isinstance(m, dict) and "role" in m and "content" in m
                   for m in row["prompt"])
    assert counts == expected

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_rows"] == len(rows)
    assert manifest["per_task_counts"] == counts
    assert manifest["field_allowlist"] == builder.FIELD_ALLOWLIST
    assert manifest["leak_check"]["result"] == "PASS"
    assert manifest["source_hashes"] and all(
        len(v["sha256"]) == 64 for v in manifest["source_hashes"].values())
    assert "inputs only" in manifest["note"]
    assert "frontier_export" in manifest["note"]

    metadata = json.loads((out / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert all(f["description"] == "inputs only (no answers)" for f in metadata["files"])


def test_shards_balanced_within_one_row(built):
    builder, out, _ = built
    sizes = [len(_read_jsonl(out / f"prompts_s{i:02d}.jsonl")) for i in range(builder.N_SHARDS)]
    assert len(sizes) == builder.N_SHARDS
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == sum(_expected_counts(builder).values())


def test_tool_readiness_for_every_task(built):
    """The regression that would have caught the zero-tool-call hosted bug."""
    from thermo_agent.teacher.agent_graph import build_context
    from thermo_agent.teacher.toolkit import make_tools, tool_names

    builder, out, _ = built
    rows = _all_rows(out)
    for task in builder.TASK_ORDER:
        row = next(r for r in rows if r["task"] == task)
        # The input columns the tools actually read must be present.
        required = {"repair": ("plant_broken",), "audit": ("plant", "site"),
                    "ledger": ("ledger",), "design": ("site",)}[task]
        assert all(k in row and row[k] for k in required), (task, required)

        ctx = build_context({"row": row, "task": task, "notes": [],
                             "mode": "agents+skills"})
        tools = make_tools(task, row)
        assert [t.__name__ for t in tools] == tool_names()
        # Hosted opt-in adds nine component tools + two on-demand skill tools.
        all_tools = make_tools(task, row, include_dynamic=True)
        assert [t.__name__ for t in all_tools] == tool_names(include_dynamic=True)
        assert len(all_tools) == 16
        inspect = tools[0]
        assert inspect.__name__ == tool_names()[0] == "inspect_plant"

        result = inspect("")
        payload = json.loads(result)
        assert "ok" in payload, (task, result)
        assert isinstance(payload["ok"], bool)
        # A missing input column would surface as `KeyError: 'plant_broken'` /
        # 'plant' / 'ledger' / 'site' in the toolkit's uniform error payload.
        assert "KeyError" not in str(payload.get("error", "")), (task, result)
        if task == "design":
            # By design there is no as-given plant to inspect, so inspect_plant
            # returns ok=false. The site is read by every candidate-based tool.
            candidate_payload = json.loads(tools[1](""))  # first_law_check
            assert "ok" in candidate_payload
            assert "KeyError" not in str(candidate_payload.get("error", ""))
