"""The trace viewer must render a real accepted row without inventing captures.

These tests build a tiny fake cohort on disk, so they assert the reader's contract
(independently of whichever traces happen to be accepted today):

  * a tool call and its result are rendered,
  * served-memory ids are resolved back to note text,
  * the skill block is labelled reconstructed, never claimed as captured,
  * an empty `reasoning` field is reported as "not captured".
"""
import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("show_trace", ROOT / "scripts" / "show_trace.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def _write_row(base: pathlib.Path, model: str, row: dict) -> pathlib.Path:
    path = base / model / "frontier_design_train.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def _row(trace_id="des-1"):
    return {
        "id": trace_id, "record_id": trace_id, "task": "design",
        "model": "claude-haiku-5-5", "provider": "anthropic", "run_id": "run-1",
        "source": "frontier:claude-haiku-5-5",
        "provenance": {"source_family": "ai_generated", "generator": "frontier_teacher",
                       "model": "claude-haiku-5-5", "provider": "anthropic", "run_id": "run-1"},
        "prompt": [{"role": "system", "content": "You design plants."},
                   {"role": "user", "content": "SITE it_load_w=1"}],
        "trajectory": {
            "final": "<design>PR = 20</design>",
            "notes_used": ["design-note-1"],
            "tools_available": ["inspect_plant", "validity_check"],
            "turns": [
                {"assistant": "", "reasoning": "",
                 "tool_calls": [{"name": "validity_check", "arguments": {"candidate": "<design>PR = 20</design>"}}],
                 "tool_results": [json.dumps({"tool": "validity_check", "ok": True, "valid": True})]},
                {"assistant": "<design>PR = 20</design>", "reasoning": "",
                 "tool_calls": [], "tool_results": []},
            ],
        },
        "validation": {"reward": 1.0, "reasons": ["design accepted"]},
    }


def _notes(base: pathlib.Path) -> pathlib.Path:
    base.mkdir(parents=True, exist_ok=True)
    path = base / "frontier_notes.jsonl"
    path.write_text(json.dumps({
        "id": "design-note-1", "task": "design", "fault_type": "format",
        "architecture": "x", "symptoms": "keys wrong", "fix": "",
        "lesson": "Attempt rejected (format). Use exactly the catalog keys.",
        "source": "frontier:someone", "accepted": False, "record_id": "des-9",
    }) + "\n", encoding="utf-8")
    return path


def test_find_row_by_id_and_record_id(tmp_path):
    _write_row(tmp_path, "claude-haiku-5-5", _row("des-1-t-claude-haiku-5-5"))
    assert mod.find_row("des-1-t-claude-haiku-5-5", tmp_path)["id"] == "des-1-t-claude-haiku-5-5"
    # record_id lookup strips the #vN suffix, matching the gate's dedup rule.
    row = _row("des-1-t-claude-haiku-5-5")
    row["record_id"] = "des-1#v3"
    _write_row(tmp_path, "claude-haiku-5-5", row)
    assert mod.find_row("des-1", tmp_path)["record_id"] == "des-1#v3"


def test_render_shows_tool_call_result_memory_and_final(tmp_path):
    _write_row(tmp_path, "claude-haiku-5-5", _row())
    _notes(tmp_path)
    notes = mod.memory_index(tmp_path)
    text = mod.render(mod.find_row("des-1", tmp_path), notes)
    assert "call 1: validity_check" in text
    assert '"valid": true' in text
    assert "[design-note-1]" in text
    assert "Use exactly the catalog keys" in text
    assert "<design>PR = 20</design>" in text
    assert "reward : 1.0" in text


def test_render_labels_skill_as_reconstructed_and_reasoning_as_missing(tmp_path):
    _write_row(tmp_path, "claude-haiku-5-5", _row())
    text = mod.render(mod.find_row("des-1", tmp_path), {})
    assert "reconstructed from TASK_SKILLS" in text
    assert "reasoning: (not captured)" in text
    assert "skill id persisted      : no" in text


def test_persisted_skill_and_memory_block_take_precedence(tmp_path):
    row = _row()
    row["trajectory"]["served_skill"] = "design-catalog"
    row["trajectory"]["system_prompt_sha256"] = "abc123"
    row["trajectory"]["memory_block"] = "FAILURE MEMORY (compact block)"
    text = mod.render(row, {})
    assert "skill: design-catalog  (persisted in the trace)" in text
    assert "system prompt sha256: abc123" in text
    assert "FAILURE MEMORY (compact block)" in text
    assert "reconstructed" not in text


def test_missing_memory_id_is_reported_not_dropped(tmp_path):
    row = _row()
    row["trajectory"]["notes_used"] = ["does-not-exist"]
    text = mod.render(row, {})
    assert "id not found in data/memory" in text


def test_best_prefers_rewarded_richer_trace(tmp_path):
    weak = _row("des-weak")
    weak["validation"] = {"reward": 0.0, "reasons": ["rejected"]}
    strong = _row("des-strong")
    _write_row(tmp_path, "claude-haiku-5-5", weak)
    path = tmp_path / "claude-haiku-5-5" / "frontier_design_train.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(strong) + "\n")
    assert mod.best_row("design", tmp_path)["id"] == "des-strong"


def test_toolinvocationresult_repr_is_unwrapped():
    raw = ("ToolInvocationResult(name='validity_check', arguments={}, call_id='c1', "
           "output='{\"ok\": true, \"valid\": true}')")
    rendered = mod.pretty_result(raw)
    assert '"valid": true' in rendered
