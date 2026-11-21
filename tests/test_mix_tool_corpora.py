"""Gate 4 mix: deterministic-only pass-through + accepted-frontier tool rows.

All tests use tiny temp fixtures; the real corpora are only hashed (read-only)
to prove the additive-only contract.  Run with::

    .venv/Scripts/python.exe -m pytest tests/test_mix_tool_corpora.py -q
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import pytest

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


def _sft_row(task: str, rid: str) -> dict:
    """A full tool-shaped SFT row; extra columns stay untouched by the mix."""
    return {
        "id": rid, "record_id": rid, "split": "train", "task": task,
        "source": "deterministic_tools",
        "prompt": [{"role": "user", "content": f"solve {task}"}],
        "completion": [{"role": "assistant", "content": "<edits>SET @x.y = 1</edits>"}],
        "provenance": {"generator": "deterministic_tools", "variant": "direct"},
        "char_length": 123, "topology": "simple",
    }


def _grpo_row(task: str, rid: str) -> dict:
    return {
        "id": rid + "-grpo", "record_id": rid, "split": "train", "task": task,
        "source": "deterministic_tools",
        "prompt": [{"role": "user", "content": f"solve {task}"}],
        "validation": {"reward": 1.0, "reasons": ["ok"]},
        "trajectory": {"final": "<edits>SET @x.y = 1</edits>", "tool_calls": 1,
                       "turns": [{"role": "assistant", "content": "thinking"}]},
    }


def _frontier_row(task: str, rid: str, *, accepted=True, tools=True) -> dict:
    content = ("I will check the plant.\n<tool_call name=\"calc\">1+1</tool_call>\n"
               "<tool_result>2</tool_result>\n<edits>SET @x.y = 1</edits>") if tools \
        else "<edits>SET @x.y = 1</edits>"
    return {
        "id": rid, "record_id": rid.replace("-ft", ""), "split": "train", "task": task,
        "source": "frontier", "accepted": accepted,
        "prompt": [{"role": "user", "content": f"frontier {task}"}],
        "completion": [{"role": "assistant", "content": content}],
        "validation": {"reward": 1.0, "reasons": ["restored a valid plant"]},
    }


def _inputs(tmp_path: Path, *, frontier_rows: dict[str, list[dict]] | None = None,
            sft_rows: dict[str, list[dict]] | None = None,
            grpo_rows: dict[str, list[dict]] | None = None) -> dict[str, Path]:
    """Four tiny deterministic rows (one per task) and an optional frontier set."""
    dsft, dgrpo, front = tmp_path / "in_sft", tmp_path / "in_grpo", tmp_path / "in_frontier"
    for task in TASKS:
        sft = (sft_rows or {}).get(task, [_sft_row(task, f"{task}-sft-1")])
        grpo = (grpo_rows or {}).get(task, [_grpo_row(task, f"{task}-sft-1")])
        _write_jsonl(dsft / f"{task}_train.jsonl", sft)
        _write_jsonl(dgrpo / f"{task}_train.jsonl", grpo)
        _write_jsonl(front / f"frontier_{task}_train.jsonl",
                     (frontier_rows or {}).get(task, []))
    return {"sft": dsft, "grpo": dgrpo, "frontier": front}


def _run(inputs: dict[str, Path], tmp_path: Path, **kwargs) -> tuple[Path, Path, dict]:
    out_sft = kwargs.pop("out_sft", tmp_path / "out_sft")
    out_grpo = kwargs.pop("out_grpo", tmp_path / "out_grpo")
    manifest = kwargs.pop("manifest", out_sft / "MANIFEST.json")
    argv = [
        "--deterministic-sft", str(inputs["sft"]),
        "--deterministic-grpo", str(inputs["grpo"]),
        "--frontier", str(inputs["frontier"]),
        "--out-sft", str(out_sft), "--out-grpo", str(out_grpo),
        "--manifest", str(manifest),
    ]
    for key, value in kwargs.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    mix.main(argv)
    return out_sft, out_grpo, json.loads(manifest.read_text(encoding="utf-8"))


def _tree_hash(path: Path) -> dict[str, str]:
    """Content hash of every file under a dir (names + bytes), for immutability checks."""
    out: dict[str, str] = {}
    if not path.is_dir():
        return out
    for file in sorted(path.rglob("*")):
        if file.is_file():
            out[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    return out


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_deterministic_only_mix_passes_rows_through_verbatim(tmp_path):
    inputs = _inputs(tmp_path)
    out_sft, out_grpo, manifest = _run(inputs, tmp_path)

    assert sorted(p.name for p in out_sft.glob("*.jsonl")) == [
        "audit_train.jsonl", "design_train.jsonl", "ledger_train.jsonl", "repair_train.jsonl"]
    for task in TASKS:
        assert _rows(out_sft / f"{task}_train.jsonl") == [_sft_row(task, f"{task}-sft-1")]
        assert _rows(out_grpo / f"{task}_train.jsonl") == [_grpo_row(task, f"{task}-sft-1")]

    assert manifest["totals"] == {"sft": 4, "grpo": 4}
    assert manifest["frontier"]["selected_count"] == 0
    assert manifest["frontier"]["selected_ids"] == []
    assert manifest["seed"] == 0
    assert len(manifest["input_file_hashes"]) == 12
    assert manifest["counts"]["sft"]["repair"] == {
        "deterministic": 1, "frontier": 0, "total": 1}
    # No claim that any frontier row made it in.
    assert all(v["frontier"] == 0 for v in manifest["counts"]["sft"].values())
    assert all(v["frontier"] == 0 for v in manifest["counts"]["grpo"].values())


def test_stats_flag_prints_table_without_writing(tmp_path, capsys):
    inputs = _inputs(tmp_path)
    out_sft, out_grpo = tmp_path / "sft_mix", tmp_path / "grpo_mix"
    code = mix.main([
        "--deterministic-sft", str(inputs["sft"]),
        "--deterministic-grpo", str(inputs["grpo"]),
        "--frontier", str(inputs["frontier"]),
        "--out-sft", str(out_sft), "--out-grpo", str(out_grpo),
        "--manifest", str(out_sft / "MANIFEST.json"),
        "--stats",
    ])
    assert code == 0
    printed = capsys.readouterr().out
    assert "det-sft" in printed and "frontier" in printed
    for task in TASKS:
        assert task in printed
    assert not out_sft.exists() and not out_grpo.exists()


def test_accepted_frontier_row_with_tool_tags_is_selected(tmp_path):
    fr = _frontier_row("repair", "repair-ft-1")
    inputs = _inputs(tmp_path, frontier_rows={"repair": [fr]})
    out_sft, out_grpo, manifest = _run(inputs, tmp_path)

    assert _rows(out_sft / "repair_train.jsonl") == [_sft_row("repair", "repair-sft-1"), fr]
    assert _rows(out_grpo / "repair_train.jsonl") == [
        _grpo_row("repair", "repair-sft-1"), fr]
    assert manifest["frontier"]["selected_count"] == 1
    assert manifest["frontier"]["selected_ids"] == ["repair-ft-1"]
    assert manifest["frontier"]["accepted"] == 1
    assert manifest["frontier"]["with_tool_turns"] == 1
    assert manifest["per_source"]["frontier"]["repair"] == 1
    assert manifest["counts"]["sft"]["repair"]["frontier"] == 1
    assert manifest["totals"]["sft"] == 5


def test_frontier_row_without_tool_tags_is_rejected(tmp_path):
    fr = _frontier_row("design", "design-ft-1", tools=False)
    inputs = _inputs(tmp_path, frontier_rows={"design": [fr]})
    out_sft, _out_grpo, manifest = _run(inputs, tmp_path)

    assert _rows(out_sft / "design_train.jsonl") == [_sft_row("design", "design-sft-1")]
    assert manifest["frontier"]["scanned"] == 1
    assert manifest["frontier"]["accepted"] == 1
    assert manifest["frontier"]["with_tool_turns"] == 0
    assert manifest["frontier"]["selected_count"] == 0
    assert manifest["frontier"]["selected_ids"] == []


def test_unaccepted_frontier_row_is_rejected_even_with_tool_tags(tmp_path):
    fr = _frontier_row("audit", "audit-ft-1", accepted=False)
    inputs = _inputs(tmp_path, frontier_rows={"audit": [fr]})
    _out_sft, _out_grpo, manifest = _run(inputs, tmp_path)
    assert manifest["frontier"]["accepted"] == 0
    assert manifest["frontier"]["selected_count"] == 0


def test_id_collision_between_sources_fails(tmp_path):
    # Same id in deterministic SFT and an accepted tool-using frontier row.
    fr = _frontier_row("ledger", "ledger-sft-1")
    inputs = _inputs(tmp_path, frontier_rows={"ledger": [fr]})
    out_sft, out_grpo = tmp_path / "out_sft", tmp_path / "out_grpo"
    with pytest.raises(SystemExit, match="id collision"):
        mix.main([
            "--deterministic-sft", str(inputs["sft"]),
            "--deterministic-grpo", str(inputs["grpo"]),
            "--frontier", str(inputs["frontier"]),
            "--out-sft", str(out_sft), "--out-grpo", str(out_grpo),
            "--manifest", str(out_sft / "MANIFEST.json"),
        ])
    assert not out_sft.exists() and not out_grpo.exists()


def test_per_task_cap_takes_precedence_over_ratio_and_is_seeded(tmp_path):
    rows = [_frontier_row("design", f"design-ft-{i}") for i in range(3)]
    inputs = _inputs(tmp_path, frontier_rows={"design": rows})

    selected = []
    for i, seed in enumerate((0, 0, 7)):
        root = tmp_path / f"run-{i}"
        _out_sft, _out_grpo, manifest = _run(
            inputs, root, out_sft=root / "out_sft", out_grpo=root / "out_grpo",
            seed=seed, frontier_per_task=2, frontier_ratio=1.0)
        selected.append(manifest["frontier"]["selected_ids"])
        assert len(manifest["frontier"]["selected_ids"]) == 2
        assert manifest["frontier"]["cap_mode"] == "per_task"
    assert selected[0] == selected[1]          # same seed -> same ids


def test_force_refuses_foreign_files_and_only_clears_owned_on_second_run(tmp_path):
    inputs = _inputs(tmp_path)
    out_sft, out_grpo, manifest = _run(inputs, tmp_path)
    owned_names = {p.name for p in out_sft.glob("*")}

    # A foreign file in an existing output dir is never overwritten, even with --force.
    (out_sft / "KEEP_ME.txt").write_text("foreign", encoding="utf-8")
    with pytest.raises(SystemExit, match="unrecognized file"):
        _run(inputs, tmp_path, force=True, manifest=out_sft / "MANIFEST.json")
    assert (out_sft / "KEEP_ME.txt").read_text(encoding="utf-8") == "foreign"

    # Without --force the existing dir is likewise refused.
    with pytest.raises(SystemExit, match="already exists"):
        _run(inputs, tmp_path)

    # Remove the foreign file, then --force must succeed by clearing owned files only.
    (out_sft / "KEEP_ME.txt").unlink()
    out_sft2, out_grpo2, manifest2 = _run(
        inputs, tmp_path, force=True, manifest=out_sft / "MANIFEST.json")
    assert out_sft2 == out_sft and out_grpo2 == out_grpo
    assert manifest2["generator"] == mix.SCRIPT_TAG
    assert {p.name for p in out_sft.glob("*")} == owned_names


def test_existing_protected_corpora_are_unchanged(tmp_path):
    """Additive-only contract: hashing before/after must be identical."""
    protected = [ROOT / "data" / name for name in (
        "sft_tools", "grpo_tools", "sft_all", "grpo_all", "frontier")]
    if not all(p.exists() for p in protected):
        pytest.skip("needs the local corpora (pre-CoT tool corpora are archived under data/archive/)")
    before = {str(p): _tree_hash(p) for p in protected}

    inputs = {
        "sft": protected[0], "grpo": protected[1], "frontier": protected[4]
    }
    _run(inputs, tmp_path)

    after = {str(p): _tree_hash(p) for p in protected}
    assert before == after
    # A real deterministic corpus has rows; the mix must not be empty.
    out_sft = tmp_path / "out_sft"
    assert sum(len(_rows(out_sft / f"{t}_train.jsonl")) for t in TASKS) > 0


# --------------------------------------------------------------------------- #
# harness-prompt mix (ours_harness prompt contract for re-prompted frontier rows)
# --------------------------------------------------------------------------- #
def _harness_det_row(task: str, rid: str, base) -> dict:
    """A deterministic row already carrying the harness prompt contract."""
    row = base(task, rid)
    row["prompt_variant"] = "harness_tools"
    row["harness"] = "agents+skills"
    row["prompt"] = [
        {"role": "system", "content": mix.HARNESS_MARKER},
        {"role": "user", "content": f"solve {task}"},
    ]
    return row


def _harness_inputs(tmp_path: Path, frontier_rows=None) -> dict[str, Path]:
    sft_rows = {t: [_harness_det_row(t, f"{t}-sft-1", _sft_row)] for t in TASKS}
    grpo_rows = {t: [_harness_det_row(t, f"{t}-sft-1", _grpo_row)] for t in TASKS}
    return _inputs(tmp_path, frontier_rows=frontier_rows,
                   sft_rows=sft_rows, grpo_rows=grpo_rows)


def test_to_harness_prompt_keeps_completion_and_sets_harness_fields():
    row = _frontier_row("repair", "repair-ft-h1")
    completion_before = json.dumps(row["completion"], ensure_ascii=False)
    prompt_before = json.dumps(row["prompt"], ensure_ascii=False)

    out = mix.to_harness_prompt(row, "repair")

    # The tool trajectory is byte-identical and the source row is not mutated.
    assert out is not row
    assert json.dumps(out["completion"], ensure_ascii=False) == completion_before
    assert out["completion"] == row["completion"]
    assert json.dumps(row["prompt"], ensure_ascii=False) == prompt_before

    assert out["prompt_variant"] == "harness_tools"
    assert out["harness"] == "agents+skills"
    assert out["prompt"] != row["prompt"]

    system = [m for m in out["prompt"] if m.get("role") == "system"]
    assert system, "expected a system message in the re-prompted prompt"
    assert mix.HARNESS_MARKER in system[0]["content"]
    assert mix.is_harness_prompt(out) is True


def test_is_harness_prompt_detects_variant_and_marker():
    plain = _sft_row("repair", "repair-sft-1")
    assert mix.is_harness_prompt(plain) is False

    by_variant = dict(plain)
    by_variant["prompt_variant"] = "harness_tools"
    assert mix.is_harness_prompt(by_variant) is True

    by_marker = dict(plain)
    by_marker["prompt"] = [
        {"role": "system", "content": mix.HARNESS_MARKER},
        {"role": "user", "content": "solve repair"},
    ]
    assert mix.is_harness_prompt(by_marker) is True


def test_build_mix_harness_prompt_rejects_plain_deterministic(tmp_path):
    inputs = _inputs(tmp_path)  # plain (non-harness) deterministic rows
    with pytest.raises(SystemExit, match="harness-prompt needs the"):
        mix.build_mix(inputs["sft"], inputs["grpo"], inputs["frontier"],
                      harness_prompt=True)


def test_build_mix_harness_prompt_reprompts_selected_frontier(tmp_path):
    fr = _frontier_row("repair", "repair-ft-1")
    inputs = _harness_inputs(tmp_path, frontier_rows={"repair": [fr]})

    plan = mix.build_mix(inputs["sft"], inputs["grpo"], inputs["frontier"],
                         harness_prompt=True)

    assert plan["frontier"]["reprompted"] == 1
    assert plan["frontier"]["selected_count"] == 1
    assert plan["harness_prompt"] is True

    selected = [r for r in plan["sft_rows"] if r.get("id") == "repair-ft-1"]
    assert len(selected) == 1
    out = selected[0]
    assert mix.is_harness_prompt(out)
    assert out["completion"] == fr["completion"]
    assert out["prompt"] != fr["prompt"]
    system = [m for m in out["prompt"] if m.get("role") == "system"]
    assert mix.HARNESS_MARKER in system[0]["content"]


def test_main_wires_harness_prompt_flag(tmp_path):
    fr = _frontier_row("repair", "repair-ft-1")
    inputs = _harness_inputs(tmp_path, frontier_rows={"repair": [fr]})

    out_sft, _out_grpo, manifest = _run(inputs, tmp_path, harness_prompt=True)

    assert manifest["config"]["prompt_variant"] == "harness_tools"
    assert manifest["frontier_reprompted"] == 1
    selected = [r for r in _rows(out_sft / "repair_train.jsonl")
                if r.get("id") == "repair-ft-1"]
    assert len(selected) == 1
    assert mix.is_harness_prompt(selected[0])
    assert selected[0]["completion"] == fr["completion"]


# --------------------------------------------------------------------------- #
# exact row-level thirds in the FINAL mix
# --------------------------------------------------------------------------- #
def test_final_mix_tiers_are_exact_thirds_per_row_not_per_plant(tmp_path):
    """All nine rows share one record_id, yet every row must get a real tier and
    the per-task counts must be 3/3/3 (the trainer consumes rows, not plants)."""
    dsft = tmp_path / "in_sft"
    sft_rows = []
    for i in range(9):
        row = _sft_row("repair", f"repair-row-{i}")
        row["record_id"] = "one-shared-plant"  # would collapse under grouped tiering
        row["plant_broken"] = json.dumps({
            "components": [{"id": f"c{j}"} for j in range(4 + i % 3)]})
        sft_rows.append(row)
    _write_jsonl(dsft / "repair_train.jsonl", sft_rows)
    dgrpo = tmp_path / "in_grpo"
    grpo_rows = []
    for i in range(6):
        row = _grpo_row("repair", f"repair-row-{i}")
        row["record_id"] = "one-shared-plant"
        row["plant_broken"] = json.dumps({
            "components": [{"id": f"c{j}"} for j in range(4 + i % 3)]})
        grpo_rows.append(row)
    _write_jsonl(dgrpo / "repair_train.jsonl", grpo_rows)
    front = tmp_path / "in_frontier"

    plan = mix.build_mix(dsft, dgrpo, front, tasks=("repair",))

    for key, rows, expected in (("sft", plan["sft_rows"], {"easy": 3, "medium": 3, "hard": 3}),
                                ("grpo", plan["grpo_rows"], {"easy": 2, "medium": 2, "hard": 2})):
        counts = collections.Counter(r.get("tier") for r in rows)
        assert counts == expected, (key, counts)
        assert all(r.get("tier") in ("easy", "medium", "hard") for r in rows)
        assert plan["tiers"]["repair"][f"counts_{key}"] == expected
    assert plan["tiers"]["repair"]["counts"] == {"easy": 3, "medium": 3, "hard": 3}
    assert plan["tiers"]["repair"]["unscored"] == 0
