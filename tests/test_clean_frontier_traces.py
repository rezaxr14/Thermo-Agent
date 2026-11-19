"""Defect 2: scrub random call_id blobs out of frontier tool_result blocks.

Additive-only script: reads a frontier dir, writes a NEW dir, rewrites only the
completion text. These tests cover the four required behaviours: long call_id is
rewritten with payload intact, a block without call_id is untouched, ids and the
final answer survive, and an existing non-empty output dir is refused.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.clean_frontier_traces as cft


BLOB = "call_" + "A" * 400 + "__mpthought__" + "b64+" * 60
OUTPUT_JSON = '{"ok": false, "solver_status": "E_FLUID_MISMATCH", "failed_checks": ["unit"]}'


def _block(name: str = "inspect_plant", call_id: str = BLOB, output: str = OUTPUT_JSON) -> str:
    cid = f", call_id='{call_id}'" if call_id is not None else ""
    return (f"<tool_result>ToolInvocationResult(name='{name}', arguments={{}}{cid}, "
            f"output='{output}', error=None)</tool_result>")


def test_long_call_id_block_is_rewritten_and_payload_survives():
    text = "before\n" + _block() + "\nafter <diagnosis>final answer</diagnosis>"
    new, count = cft.rewrite_completion(text)
    assert count == 1
    assert "call_id" not in new
    assert "ToolInvocationResult" not in new
    assert '{"tool": "inspect_plant"' in new
    assert "E_FLUID_MISMATCH" in new and "unit" in new
    # Everything outside the block is untouched.
    assert new.startswith("before\n")
    assert new.endswith("\nafter <diagnosis>final answer</diagnosis>")


def test_block_without_call_id_is_unchanged():
    clean = '<tool_result>{"tool": "inspect_plant", "ok": true}</tool_result>'
    text = "x\n" + clean + "\ny"
    new, count = cft.rewrite_completion(text)
    assert new == text
    assert count == 0


def test_python_repr_output_with_escaped_quotes_is_decoded():
    """Real-shaped block: output= is a Python repr, not literal JSON.

    The payload contains ``\\'`` (apostrophe in "don't") and ``\\\\"`` (quotes in
    a PropsSI call). ``json.loads`` on the raw body fails; the block must still be
    rewritten with its dict payload intact and no call_id left behind.
    """
    js = '{"ok": false, "solver_message": "Water don\'t support PropsSI(\\"H\\",\\"T\\")"}'
    body = repr(js)[1:-1]  # inside of the single-quoted Python repr used in the data
    block = (f"<tool_result>ToolInvocationResult(name='inspect_plant', arguments={{}}, "
             f"call_id='{BLOB}', output='{body}', error=None)</tool_result>")

    new, count = cft.rewrite_completion(block)
    assert count == 1
    assert "call_id" not in new
    assert "ToolInvocationResult" not in new
    assert '"tool": "inspect_plant"' in new
    obj = json.loads(new[len("<tool_result>"):-len("</tool_result>")])
    assert obj["tool"] == "inspect_plant"
    assert "don't" in obj["solver_message"]
    assert 'PropsSI("H","T")' in obj["solver_message"]


def _leftover_call_ids(directory: Path) -> int:
    """Whole-file assertion helper: no cleaned artifact may contain a call_id blob."""
    return sum((p.read_text(encoding="utf-8").count("call_id='")
                for p in directory.glob("*.jsonl")), 0)


def test_rewrite_is_additive_and_refuses_to_overwrite(tmp_path: Path):
    src = tmp_path / "frontier_tools"
    src.mkdir()
    rid = "rep-99-00044-t-google-gemini-3-8-flash"
    row = {
        "id": rid,
        "record_id": "rep-99-00044",
        "task": "repair",
        "model": "google-gemini-3-8-flash",
        "prompt": [{"role": "user", "content": "fix it"}],
        "completion": [{"role": "assistant", "content":
                        _block() + "\n<diagnosis>THE FINAL ANSWER</diagnosis>"}],
    }
    (src / "frontier_repair_train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    out = tmp_path / "frontier_tools_clean"
    manifest = cft.clean_dir(src, out)
    assert manifest["rewritten_blocks"] == 1
    assert manifest["files"] == 1
    assert "chars" in manifest["before"] and "chars" in manifest["after"]
    assert "tokens" in manifest["before"] and "tokens" in manifest["after"]
    assert manifest["after"]["chars"]["max"] < manifest["before"]["chars"]["max"]
    assert manifest["leftover_call_ids"] == 0
    assert _leftover_call_ids(out) == 0

    out_row = json.loads((out / "frontier_repair_train.jsonl").read_text(encoding="utf-8").strip())
    assert out_row["id"] == rid
    assert out_row["record_id"] == row["record_id"]
    assert out_row["model"] == row["model"]
    assert out_row["prompt"] == row["prompt"]
    assert "call_id" not in out_row["completion"][0]["content"]
    assert "<diagnosis>THE FINAL ANSWER</diagnosis>" in out_row["completion"][0]["content"]
    assert '{"tool": "inspect_plant"' in out_row["completion"][0]["content"]

    # Never overwrite an existing non-empty output dir.
    with pytest.raises(SystemExit):
        cft.clean_dir(src, out)


def test_trajectory_blobs_are_scrubbed_and_whole_dir_has_zero_leftovers(tmp_path: Path):
    src = tmp_path / "frontier_tools"
    src.mkdir()
    row = {
        "id": "r-traj",
        "record_id": "rec-1",
        "prompt": [{"role": "user", "content": "fix it"}],
        "completion": [{"content": _block() + "\n<diagnosis>FINAL</diagnosis>"}],
        "trajectory": {"turns": [{"tool_results": [_block(name="solve_plant")]}]},
    }
    (src / "frontier_repair_train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    out = tmp_path / "clean"
    manifest = cft.clean_dir(src, out)
    # one wrapped block in completion + one bare blob in trajectory
    assert manifest["rewritten_blocks"] == 2
    assert manifest["leftover_call_ids"] == 0
    assert _leftover_call_ids(out) == 0

    cleaned = json.loads((out / "frontier_repair_train.jsonl").read_text(encoding="utf-8").strip())
    assert cleaned["id"] == "r-traj" and cleaned["record_id"] == "rec-1"
    assert "call_id" not in json.dumps(cleaned["completion"])
    assert "call_id" not in json.dumps(cleaned["trajectory"])
    assert cleaned["trajectory"]["turns"][0]["tool_results"][0].startswith('{"tool": "solve_plant"')
    assert "<diagnosis>FINAL</diagnosis>" in cleaned["completion"][0]["content"]


def test_dry_run_writes_nothing(tmp_path: Path):
    src = tmp_path / "frontier_tools"
    src.mkdir()
    row = {"id": "r1", "completion": [{"content": _block()}]}
    (src / "frontier_repair_train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    out = tmp_path / "clean"
    manifest = cft.clean_dir(src, out, dry_run=True)
    assert manifest["rewritten_blocks"] == 1
    assert manifest["leftover_call_ids"] == 0
    assert not out.exists()
