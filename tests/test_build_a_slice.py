"""E2 A-slice: one plain row per ground-truth plant, task-balanced, gated.

Tiny temp fixtures only; the real corpora are never written.  Run with::

    .venv/Scripts/python.exe -m pytest tests/test_build_a_slice.py -q
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from scripts import build_a_slice as a_slice
from scripts import mix_tool_corpora as mix

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("repair", "design", "audit", "ledger")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")
    return path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _row(task: str, rid: str, *, tokens: int = 100, harness: str = "none",
         answer: str = "<diagnosis>ok</diagnosis>") -> dict:
    return {
        "id": rid, "split": "train", "task": task, "harness": harness,
        "prompt": [{"role": "system", "content": f"plain {task} task"},
                   {"role": "user", "content": "solve this plant"}],
        "completion": [{"role": "assistant", "content": answer}],
        "tokens": tokens,
    }


def _sft_all(tmp_path: Path, by_task: dict[str, list[dict]]) -> Path:
    root = tmp_path / "sft_all"
    for task in TASKS:
        _write_jsonl(root / f"{task}_train.jsonl", by_task.get(task, []))
    return root


def _plan(root: Path, *, tasks, tool_rows: int = 4, ratio: float = 0.25,
          heldout=None, seed: int = 0, tok=None) -> dict:
    return a_slice.plan_slice(root, tool_rows=tool_rows, ratio=ratio, seed=seed,
                              tasks=tasks, heldout=heldout or tmp_heldout(root),
                              tok=tok)


def tmp_heldout(root: Path) -> Path:
    """An empty heldout dir: no fixture id is a dev-set id by accident."""
    return root.parent / "heldout_empty"


def _first_id(path: Path) -> str:
    return _rows(path)[0]["id"]


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_one_ground_truth_plant_contributes_exactly_one_plain_row(tmp_path):
    # 20-style paraphrase family: the bare id and #v1..#vN share one plant.
    repair = ([_row("repair", "rep-0-00000" + s, answer=f"answer {s}")
               for s in ("", "#v1", "#v2", "#v3")]
              + [_row("repair", "rep-0-00001#v1", answer="other 1"),
                 _row("repair", "rep-0-00001#v2", answer="other 2"),
                 _row("repair", "rep-0-00002", answer="third")])
    root = _sft_all(tmp_path, {"repair": repair})

    plan = _plan(root, tasks=("repair",), tool_rows=8, ratio=0.25, tok=None)
    chosen = plan["rows"]["repair"]

    # target = round(8*0.25) = 2 -> two distinct plants, never two paraphrases.
    assert plan["target"] == 2 and len(chosen) == 2 and plan["met_target"]
    keys = [a_slice.plant_key(r) for r in chosen]
    assert len(set(keys)) == 2
    assert set(keys) <= {"rep-0-00000", "rep-0-00001", "rep-0-00002"}
    answers = {r["id"]: r["completion"][0]["content"] for r in repair}
    for row in chosen:
        assert row["task"] == "repair"
        assert row["completion"][0]["content"] == answers[row["id"]]
        assert row["prompt_variant"] == "plain" and row["harness"] == "none"

    # Seed makes the choice reproducible.
    again = _plan(root, tasks=("repair",), tool_rows=8, ratio=0.25, tok=None, seed=0)
    assert [r["id"] for r in again["rows"]["repair"]] == [r["id"] for r in chosen]


def test_over_gate_rows_are_dropped_and_backfilled(tmp_path, monkeypatch):
    # Force the recorded token column so the gate is testable without a 4B
    # tokenizer on a tiny fixture. build_a_slice still measures via the same
    # measure_tokens() path; nothing is truncated.
    monkeypatch.setattr(a_slice, "_tokenizer", lambda: None)
    repair = ([_row("repair", "rep-0-00000" + s, tokens=a_slice.GATE + 1)
               for s in ("", "#v1", "#v2")]
              + [_row("repair", "rep-0-00001" + s, tokens=120)
                 for s in ("", "#v1", "#v2")])
    root = _sft_all(tmp_path, {"repair": repair})

    plan = _plan(root, tasks=("repair",), tool_rows=4, ratio=0.25, tok=None)

    assert plan["target"] == 1 and plan["met_target"]
    assert plan["dropped_over_gate"] == 3
    assert plan["dropped_over_gate_by_task"]["repair"] == 3
    chosen = plan["rows"]["repair"]
    assert len(chosen) == 1
    assert a_slice.plant_key(chosen[0]) == "rep-0-00001"
    assert plan["token_length"]["max"] <= a_slice.GATE
    assert all(r["id"].startswith("rep-0-00001") for r in chosen)


def test_no_heldout_id_is_selected(tmp_path):
    heldout = ROOT / "data" / "heldout"
    leaked = _first_id(heldout / "repair_test_id.jsonl")
    repair = ([_row("repair", leaked, answer="LEAK")]
              + [_row("repair", f"rep-0-0000{i}") for i in range(3)])
    root = _sft_all(tmp_path, {"repair": repair})

    plan = _plan(root, tasks=("repair",), tool_rows=8, ratio=0.25,
                 heldout=heldout, tok=None)
    selected_ids = {r["id"] for r in plan["rows"]["repair"]}
    selected_plants = set(plan["plants_represented"])

    assert leaked not in selected_ids and a_slice.plant_key({"id": leaked}) not in selected_plants
    assert plan["heldout_excluded"] >= 1
    assert plan["met_target"]


def test_manifest_has_counts_plants_tokens_drops_hashes_and_rule(tmp_path):
    root = _sft_all(tmp_path, {
        task: [_row(task, f"{task[:3]}-0-{i:05d}", tokens=200 + i)
               for i in range(3)] for task in TASKS})
    plan = _plan(root, tasks=TASKS, tool_rows=8, ratio=0.5, tok=None)
    out = tmp_path / "a_slice"

    manifest = a_slice.write_slice(plan, out)

    assert manifest["target"] == 4 and manifest["selected"] == 4
    assert all(manifest["counts_by_task"][t] == 1 for t in TASKS)
    assert all(manifest["plants_represented_by_task"][t] == 1 for t in TASKS)
    st = manifest["token_length"]
    assert set(("p50", "p95", "max")) <= set(st) and st["max"] <= a_slice.GATE
    assert manifest["dropped_over_gate"] == 0
    assert "#vN" in manifest["selection_rule"] and "8192" in manifest["selection_rule"]
    on_disk = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    for rel, digest in manifest["input_file_hashes"].items():
        path = Path(rel)
        assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    for task in TASKS:
        rows = _rows(out / f"{task}_train.jsonl")
        assert len(rows) == 1 and rows[0]["prompt_variant"] == "plain"


def test_refuses_nonempty_output_dir_and_unmet_target(tmp_path):
    root = _sft_all(tmp_path, {"repair": [_row("repair", "rep-0-00000")]})
    out = tmp_path / "busy_slice"
    _write_jsonl(out / "KEEP_ME.jsonl", [_row("repair", "keep")])
    with pytest.raises(SystemExit, match="non-empty output dir"):
        a_slice.write_slice(_plan(root, tasks=("repair",), tok=None), out)
    assert _rows(out / "KEEP_ME.jsonl")[0]["id"] == "keep"

    # Too few plants for the target: never commit a partial slice.
    with pytest.raises(SystemExit, match="target not met"):
        a_slice.main(["--sft-all", str(root), "--out", str(tmp_path / "partial"),
                      "--tasks", "repair", "--tool-rows", "8", "--ratio", "0.5",
                      "--heldout", str(tmp_heldout(root))])
    assert not (tmp_path / "partial").exists()


def test_default_output_is_allowed_but_a_protected_corpus_is_not(tmp_path, monkeypatch):
    """`data/a_slice` is the default output; it must not be in the protected list (it was)."""
    monkeypatch.setattr(a_slice, "ROOT", tmp_path)
    a_slice._check_destination(tmp_path / "data" / "a_slice", tmp_path / "data" / "a_slice" / "MANIFEST.json")
    with pytest.raises(SystemExit, match="protected corpus"):
        a_slice._check_destination(tmp_path / "data" / "heldout_v2", tmp_path / "out" / "MANIFEST.json")


def test_stats_flag_writes_nothing(tmp_path, capsys):
    root = _sft_all(tmp_path, {"repair": [_row("repair", "rep-0-00000")]})
    out = tmp_path / "dry_slice"
    code = a_slice.main(["--sft-all", str(root), "--out", str(out),
                         "--tasks", "repair", "--stats",
                         "--heldout", str(tmp_heldout(root))])
    assert code == 0
    printed = capsys.readouterr().out
    assert "quota" in printed and "repair" in printed and "TOTAL" in printed
    assert not out.exists()


# --------------------------------------------------------------------------- #
# mix wiring: harness tool rows + plain A rows
# --------------------------------------------------------------------------- #
def _harness_row(task: str, rid: str) -> dict:
    row = mix._sft_row(task, rid) if hasattr(mix, "_sft_row") else {
        "id": rid, "record_id": rid, "split": "train", "task": task,
        "source": "deterministic_tools",
        "prompt": [{"role": "user", "content": f"solve {task}"}],
        "completion": [{"role": "assistant", "content": "<edits>SET @x.y = 1</edits>"}],
    }
    row["prompt_variant"] = "harness_tools"
    row["harness"] = "agents+skills"
    row["prompt"] = [{"role": "system", "content": mix.HARNESS_MARKER},
                     {"role": "user", "content": f"solve {task}"}]
    return row


def _mix_inputs(tmp_path: Path) -> dict[str, Path]:
    det_sft, det_grpo, frontier, a_dir = (tmp_path / n for n in
                                          ("det_sft", "det_grpo", "frontier", "a_dir"))
    for task in TASKS:
        _write_jsonl(det_sft / f"{task}_train.jsonl", [_harness_row(task, f"{task}-tool")])
        _write_jsonl(det_grpo / f"{task}_train.jsonl", [_harness_row(task, f"{task}-tool-grpo")])
        _write_jsonl(frontier / f"frontier_{task}_train.jsonl", [])
        _write_jsonl(a_dir / f"{task}_train.jsonl", [_row(task, f"{task}-plain")])
    return {"sft": det_sft, "grpo": det_grpo, "frontier": frontier, "a": a_dir}


def test_mix_keeps_tool_and_plain_variants_distinct(tmp_path):
    inputs = _mix_inputs(tmp_path)
    plan = mix.build_mix(inputs["sft"], inputs["grpo"], inputs["frontier"],
                         harness_prompt=True, a_slice=inputs["a"], a_slice_ratio=0.0)

    tool = [r for r in plan["sft_rows"] if r["id"].endswith("-tool")]
    plain = [r for r in plan["sft_rows"] if r["id"].endswith("-plain")]
    assert len(tool) == len(plain) == len(TASKS)
    assert all(r["prompt_variant"] == "harness_tools" and mix.is_harness_prompt(r)
               for r in tool)
    assert all(r["prompt_variant"] == "plain" and r["harness"] == "none"
               and not mix.is_harness_prompt(r) for r in plain)
    # A rows are never re-prompted: their prompt is byte-identical to the slice.
    original = _rows(inputs["a"] / "repair_train.jsonl")[0]
    assert next(r for r in plain if r["task"] == "repair")["prompt"] == original["prompt"]

    assert all(plan["counts"]["sft"][t]["plain_a"] == 1 for t in TASKS)
    assert plan["counts"]["sft"]["repair"]["total"] == 2
    # GRPO is untouched by the A-slice.
    assert all(plan["counts"]["grpo"][t]["total"] == 1 for t in TASKS)

    out = io.StringIO()
    mix.print_stats(plan, out)
    printed = out.getvalue()
    assert "plain-A" in printed and "det-sft" in printed and "frontier" in printed
    for task in TASKS:
        line = next(l for l in printed.splitlines() if l.startswith(task))
        parts = line.split()
        assert parts[1:4] == ["1", "1", "0"]  # det-sft, det-grpo, frontier
        assert parts[4] == "1"                # plain-A


def test_mix_main_writes_both_variants_and_unchanged_without_a_slice(tmp_path):
    inputs = _mix_inputs(tmp_path)
    out_sft, out_grpo = tmp_path / "sft_mix_h", tmp_path / "grpo_mix_h"
    mix.main(["--deterministic-sft", str(inputs["sft"]),
              "--deterministic-grpo", str(inputs["grpo"]),
              "--frontier", str(inputs["frontier"]),
              "--a-slice", str(inputs["a"]), "--a-slice-ratio", "0",
              "--harness-prompt",
              "--out-sft", str(out_sft), "--out-grpo", str(out_grpo),
              "--manifest", str(out_sft / "MANIFEST.json")])
    rows = _rows(out_sft / "repair_train.jsonl")
    assert {r["prompt_variant"] for r in rows} == {"harness_tools", "plain"}
    manifest = json.loads((out_sft / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["a_slice"]["enabled"]
    assert all(n == 1 for n in manifest["a_slice"]["selected"].values())
    assert all(manifest["counts"]["sft"][t]["plain_a"] == 1 for t in TASKS)
    assert manifest["a_slice"]["selected_ids"] == [f"{t}-plain" for t in TASKS]

    # A positive ratio caps A rows at round(ratio * deterministic rows).
    capped = tmp_path / "capped"
    _write_jsonl(capped / "det_sft" / "repair_train.jsonl",
                 [_harness_row("repair", f"repair-tool-{i}") for i in range(3)])
    _write_jsonl(capped / "det_grpo" / "repair_train.jsonl", [])
    _write_jsonl(capped / "frontier" / "frontier_repair_train.jsonl", [])
    _write_jsonl(capped / "a_dir" / "repair_train.jsonl",
                 [_row("repair", f"repair-plain-{i}") for i in range(3)])
    cap_plan = mix.build_mix(capped / "det_sft", capped / "det_grpo",
                             capped / "frontier", tasks=("repair",),
                             a_slice=capped / "a_dir", a_slice_ratio=0.3333)
    assert cap_plan["counts"]["sft"]["repair"]["plain_a"] == 1
    assert cap_plan["a_slice"]["skipped_over_cap"]["repair"] == 2

    # Without --a-slice the original behaviour is unchanged (no plain-A key).
    out2, grpo2 = tmp_path / "sft_mix_plain", tmp_path / "grpo_mix_plain"
    mix.main(["--deterministic-sft", str(inputs["sft"]),
              "--deterministic-grpo", str(inputs["grpo"]),
              "--frontier", str(inputs["frontier"]),
              "--out-sft", str(out2), "--out-grpo", str(grpo2),
              "--manifest", str(out2 / "MANIFEST.json")])
    manifest2 = json.loads((out2 / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest2["a_slice"]["enabled"] is False
    assert set(manifest2["counts"]["sft"]["repair"]) == {"deterministic", "frontier", "total"}
