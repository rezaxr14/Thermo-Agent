"""Gate 3a/4: deterministic tool traces and the tool-only corpora.

Uses one real ground-truth row per task and checks that every trace embeds *verbatim*
real tool JSON, ends with the accepted answer, and never leaks a ground-truth answer
block into a tool result.  All tests that need the untracked corpora are marked
``corpus`` so a fresh clone skips them cleanly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from thermo_agent.synth.tool_traces import (
    build_trace, ground_truth, leaky_tool_results,
)
from thermo_agent.teacher.toolkit import make_tools
from thermo_agent.teacher.validate import adversarial_candidates, validate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TASKS = ("repair", "audit", "ledger", "design")


@pytest.fixture(scope="module")
def rows() -> dict[str, dict]:
    return {
        task: json.loads((DATA / "processed" / f"{task}_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
        for task in TASKS
    }


def _results(trace: dict) -> list[str]:
    return [r for turn in trace["turns"] for r in turn["tool_results"]]


@pytest.mark.corpus
@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("variant", ["direct", "recover"])
def test_trace_has_real_tools_and_ends_with_accepted_answer(rows, task, variant):
    row = rows[task]
    trace = build_trace(row, task, variant=variant)
    final = ground_truth(row)

    assert trace["accepted"] is True
    assert trace["source"] == "deterministic_tools"
    assert trace["tool_calls"] >= 2
    assert "<tool_call" in trace["sft_completion"]
    assert "<tool_result" in trace["sft_completion"]
    assert trace["sft_completion"].rstrip().endswith(final.rstrip())
    assert validate(task, final, row)["accepted"] is True
    assert trace["char_length"] == len(trace["sft_completion"])

    results = _results(trace)
    assert results
    assert leaky_tool_results(trace) == [], "a ground-truth answer block leaked into a tool result"
    for raw in results:
        assert final not in raw, "the full ground-truth answer was pasted into a tool result"
        assert json.loads(raw)  # verbatim tool JSON, not prose

    # Every recorded result must be exactly what the real toolkit returned for the
    # recorded candidate (no edited numbers, no invented JSON).
    tools = {fn.__name__: fn for fn in make_tools(task, row)}
    for turn in trace["turns"]:
        for call, raw in zip(turn["tool_calls"], turn["tool_results"]):
            candidate = call["arguments"]["candidate"]
            assert raw == tools[call["name"]](candidate)


@pytest.mark.corpus
def test_recover_exposes_failure_then_confirms_the_fix(rows):
    trace = build_trace(rows["repair"], "repair", variant="recover")
    results = [json.loads(raw) for raw in _results(trace)]
    assert trace["recover_candidate"] in {"no_edits", "echo_broken", "wrong_scale",
                                          "third_fluid", "bad_port"}
    bad = results[1]  # the adversarial candidate's check
    assert bad["ok"] is False and bad["failed_checks"], bad
    good = results[2]  # the corrected edits' check
    assert good["ok"] is True and good["failed_checks"] == [], good


@pytest.mark.corpus
def test_distinct_recover_variants_have_distinct_tool_results(rows):
    names = [name for name, _ in adversarial_candidates(rows["repair"])]
    assert len(names) >= 2, "need at least two adversarial candidates in this row"
    seen = {}
    for name in names:
        trace = build_trace(rows["repair"], "repair", variant=f"recover:{name}")
        assert trace["recover_candidate"] == name
        seen[name] = _results(trace)[1]  # the failing candidate's tool result
    assert len(set(seen.values())) == len(seen), "different candidates produced identical tool results"


@pytest.mark.corpus
def test_builder_writes_manifest_and_guard_rejects_planted_heldout_row(rows, tmp_path: Path):
    from scripts.build_tool_corpus import (build, contamination_guard,
                                            scan_rows, build_heldout_signatures)

    out = tmp_path / "sft_tools"
    grpo = tmp_path / "grpo_tools"
    manifest = build(
        ["data/processed"], str(out), str(grpo),
        tasks=("repair",), split="train", limit=1, include_frontier=False,
        heldout_dir="data/heldout")
    assert (out / "MANIFEST.json").is_file()
    assert manifest["totals"]["sft"] == 2  # direct + recover
    assert manifest["totals"]["grpo"] == 2
    assert manifest["char_length"]["max"] >= manifest["char_length"]["p50"]
    assert manifest["contamination_guard"]["passed"] is True
    assert manifest["contamination_guard"]["overlap"] == 0
    written = [json.loads(line) for line in (out / "repair_train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["provenance"]["variant"] for r in written} == {"direct", "recover"}
    assert all("<tool_call" in r["completion"][0]["content"] for r in written)

    # Self-check: a held-out row planted into a training file MUST be rejected.
    index, files = build_heldout_signatures(DATA / "heldout")
    assert files
    source = DATA / "heldout" / "repair_test_id.jsonl"
    planted = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
    assert scan_rows([planted], index), "the guard did not detect the planted held-out row"
    guarded = contamination_guard([planted], heldout_dir=DATA / "heldout")
    assert guarded["passed"] is False and guarded["overlap"] >= 1
    assert any("plant_broken" in leak["kinds"] or "id" in leak["kinds"]
               for leak in guarded["leaks"])
