"""The tool-using student must train on the same system prompt it sees in the H eval arm.

All builds use one tiny committed fixture row and temp dirs, so this is fast and
independent of the untracked full corpora.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from scripts.build_tool_corpus import build
from thermo_agent.agent.tool_prompt import build_harness_tool_messages
from thermo_agent.eval import model_eval as ME

TASK = "design"  # no plant needed; skills_for(None, "design") is always ["design-catalog"]
AGENTS_MARKER = "# AGENTS.md: energy-plant tasks"
CONTRACT_HEADER = "MULTI-TURN TOOL ENVIRONMENT"


def _src(tmp_path, corpus_fixture_rows):
    row = corpus_fixture_rows[TASK][0]
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    (src / f"{TASK}_train.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return src, row


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _build(src, tmp_path, variant, **kw):
    out = kw.get("out", tmp_path / "sft_tools")
    grpo = kw.get("grpo", tmp_path / "grpo_tools")
    manifest = build(
        [str(src)], str(out), str(grpo), tasks=(TASK,), split="train", limit=1,
        include_frontier=False, heldout_dir="tests/fixtures/heldout",
        prompt_variant=variant,
        out_harness=kw.get("out_harness"), grpo_out_harness=kw.get("grpo_out_harness"))
    if variant == "harness_tools":
        out = kw.get("out_harness") or pathlib.Path(str(out) + "_h")
        grpo = kw.get("grpo_out_harness") or pathlib.Path(str(grpo) + "_h")
    return manifest, pathlib.Path(out), pathlib.Path(grpo)


# ---------------------------------------------------------------------------
# 1. plain is today's behaviour: the source prompt verbatim, no harness text.
# ---------------------------------------------------------------------------
def test_plain_row_prompt_equals_source(tmp_path, corpus_fixture_rows):
    src, row = _src(tmp_path, corpus_fixture_rows)
    _, out, grpo = _build(src, tmp_path, "plain")
    sft = _rows(out / f"{TASK}_train.jsonl")
    grpo_rows = _rows(grpo / f"{TASK}_train.jsonl")
    assert sft and grpo_rows

    source_system = row["prompt"][0]["content"]
    for built in (sft[0], grpo_rows[0]):
        system = built["prompt"][0]["content"]
        assert system == source_system
        assert AGENTS_MARKER not in system
        assert "SKILL design-catalog" not in system
        assert CONTRACT_HEADER not in system
    assert sft[0]["harness"] == "none"
    assert "prompt_variant" not in sft[0] and "prompt_variant" not in grpo_rows[0]


# ---------------------------------------------------------------------------
# 2. harness_tools: AGENTS.md + a SKILL line + the exact shared contract header.
# ---------------------------------------------------------------------------
def test_harness_row_has_agents_skills_and_contract(tmp_path, corpus_fixture_rows):
    src, _ = _src(tmp_path, corpus_fixture_rows)
    _, plain_out, plain_grpo = _build(src, tmp_path, "plain")
    plain_sft = _rows(plain_out / f"{TASK}_train.jsonl")
    plain_grpo = _rows(plain_grpo / f"{TASK}_train.jsonl")

    _, h_out, h_grpo = _build(src, tmp_path, "harness_tools")
    assert h_out.name.endswith("_h") and h_grpo.name.endswith("_h")
    sft = _rows(h_out / f"{TASK}_train.jsonl")
    grpo_rows = _rows(h_grpo / f"{TASK}_train.jsonl")
    assert len(sft) == 2 and len(grpo_rows) == 2  # direct + recover

    plain_sft_by_variant = {r["provenance"]["variant"]: r for r in plain_sft}
    plain_grpo_by_variant = {r["provenance"]["variant"]: r for r in plain_grpo}

    for built, plain in ((sft, plain_sft_by_variant), (grpo_rows, plain_grpo_by_variant)):
        for row in built:
            system = row["prompt"][0]["content"]
            assert AGENTS_MARKER in system
            assert any(line.startswith("SKILL ") for line in system.splitlines())
            assert "SKILL design-catalog" in system
            assert CONTRACT_HEADER in system
            assert '<tool_call name="TOOL_NAME">' in system
            # Only the prompt/metadata change: completions and trajectories stay identical.
            twin = plain[row["provenance"]["variant"]]
            assert row["completion"] == twin["completion"]
            if "trajectory" in row:
                assert row["trajectory"] == twin["trajectory"]
            assert row["harness"] == "agents+skills"
            assert row["prompt_variant"] == "harness_tools"


# ---------------------------------------------------------------------------
# 3. regression against drift: shared builder == ToolLoop._base_messages.
# ---------------------------------------------------------------------------
def test_shared_builder_matches_toolloop_base_messages(corpus_fixture_rows):
    row = corpus_fixture_rows[TASK][0]
    loop = ME.ToolLoop(lambda messages: "", row, TASK, mode="agents+skills",
                       max_turns=4, max_tool_calls=6)
    expected, _ = loop._base_messages()
    got = build_harness_tool_messages(row["prompt"], TASK, None,
                                      max_turns=4, max_tool_calls=6)  # match the loop's explicit limits
    assert got == expected


# ---------------------------------------------------------------------------
# 4. additive-only: a harness run never touches existing plain corpora.
# ---------------------------------------------------------------------------
def test_harness_run_leaves_plain_corpora_untouched(tmp_path, corpus_fixture_rows):
    src, _ = _src(tmp_path, corpus_fixture_rows)
    _, out, grpo = _build(src, tmp_path, "plain")
    before = {str(p): p.read_bytes() for d in (out, grpo) for p in d.iterdir()}
    assert before

    _build(src, tmp_path, "harness_tools")
    after = {str(p): p.read_bytes() for d in (out, grpo) for p in d.iterdir()}
    assert after == before
    assert (tmp_path / "sft_tools_h" / f"{TASK}_train.jsonl").is_file()
    assert (tmp_path / "grpo_tools_h" / f"{TASK}_train.jsonl").is_file()


def test_harness_guard_rejects_non_h_dir_and_nonempty_dir(tmp_path, corpus_fixture_rows):
    src, _ = _src(tmp_path, corpus_fixture_rows)
    common = dict(tasks=(TASK,), include_frontier=False, heldout_dir="tests/fixtures/heldout")

    # Explicit harness destination must be a new *_h dir.
    with pytest.raises(SystemExit):
        build([str(src)], str(tmp_path / "sft_tools"), str(tmp_path / "grpo_tools"),
              prompt_variant="harness_tools", out_harness=str(tmp_path / "bad_out"), **common)

    # A pre-existing non-empty *_h dir is never overwritten.
    nonempty = tmp_path / "sft_tools_h"
    nonempty.mkdir()
    (nonempty / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(SystemExit):
        build([str(src)], str(tmp_path / "sft_tools"), str(tmp_path / "grpo_tools"),
              prompt_variant="harness_tools", **common)
    assert (nonempty / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "grpo_tools_h").exists()
