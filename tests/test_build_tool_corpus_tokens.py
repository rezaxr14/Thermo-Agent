"""Defect 1: the 8k gate must measure the FINAL (harness) prompt, not the plain one.

Regression test for the bug where ``build_trace`` measured the plain prompt, ``_emit``
then swapped in the ~4.3k-token harness prompt, and nothing re-checked the gate, so
over-budget rows were silently emitted.

We use a deterministic word-count tokenizer and a controllable harness builder so the
test does not depend on the real tokenizer file or on the exact size of AGENTS.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.build_tool_corpus as btc
from thermo_agent.synth import combine, tool_traces
from _rows import make_row


RESERVE = combine.TOKEN_RESERVE


class _WordTok:
    """Tokenize by whitespace: one 'token' per word, fully deterministic."""

    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=text.split())


def _ntok(text: str) -> int:
    return len(text.split())


@pytest.fixture
def word_tokenizer(monkeypatch):
    # patch the function itself so both tool_traces and build_tool_corpus see it,
    # without touching module-level caches.
    monkeypatch.setattr(combine, "tokenizer", lambda: _WordTok())


@pytest.fixture
def harness_builder(monkeypatch):
    """Small harness for rows marked HARNESS_SMALL, huge one for HARNESS_BIG."""

    def fake(messages, task, plant, **kw):
        text = " ".join(str(m.get("content", "")) for m in messages)
        big = "HARNESS_BIG" in text
        n = 20000 if big else 10
        prefix = [{"role": "system", "content": " ".join(["h"] * n)}]
        return prefix + [dict(m) for m in messages]

    monkeypatch.setattr(btc, "build_harness_tool_messages", fake)


def _row(marker: str, seed: int) -> dict:
    row = make_row("wrong_fluid", seed=seed)
    row["prompt"] = [
        {"role": "system", "content": f"sys {marker}"},
        {"role": "user", "content": "fix the plant"},
    ]
    return row


def _plain_tokens(row: dict, trace: dict) -> int:
    n_prompt = sum(_ntok(m.get("content", "")) for m in row["prompt"])
    return n_prompt + trace["completion_tokens"] + RESERVE


@pytest.mark.corpus
def test_gate_measures_final_harness_prompt_not_plain(tmp_path: Path, word_tokenizer, harness_builder):
    small = _row("HARNESS_SMALL", seed=0)
    big = _row("HARNESS_BIG", seed=1)
    rows = [small, big]

    # Threshold: above every plain-prompt row, and above the small harness row,
    # but far below the huge harness row. All plain rows fit; only the big
    # harness row must be dropped.
    plain_max = 0
    for row in rows:
        for variant in ("direct", "recover"):
            tr = tool_traces.build_trace(row, "repair", variant=variant, strict=False)
            plain_max = max(plain_max, _plain_tokens(row, tr))
    max_tokens = plain_max + 500

    src = tmp_path / "processed"
    src.mkdir()
    with (src / "repair_train.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    out = tmp_path / "sft_tools"
    grpo = tmp_path / "grpo_tools"
    manifest = btc.build(
        [str(src)], str(out), str(grpo),
        tasks=("repair",), split="train", include_frontier=False,
        max_tokens=max_tokens, prompt_variant="harness_tools",
        heldout_dir="data/heldout")

    written = [json.loads(line) for line in
               (tmp_path / "sft_tools_h" / "repair_train.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = {r["record_id"] for r in written}

    # The big-harness row is dropped and counted; the small-harness row is kept.
    assert str(big["id"]) not in ids
    assert str(small["id"]) in ids
    assert manifest["dropped_over_8k"] == 2, manifest["dropped_over_8k"]

    # Kept rows carry the re-measured value from the ACTUAL final prompt.
    for r in written:
        expected = (sum(_ntok(m.get("content", "")) for m in r["prompt"])
                    + _ntok(r["completion"][0]["content"]) + RESERVE)
        assert r["tokens"] == expected, (r["id"], r["tokens"], expected)
        assert r["tokens"] <= max_tokens
        assert r["prompt_variant"] == "harness_tools"

    # The manifest must expose BOTH measurements and they must differ.
    assert "token_length_plain" in manifest and "token_length" in manifest
    assert manifest["token_length_plain"]["max"] < manifest["token_length"]["max"]


@pytest.mark.corpus
def test_rows_with_frontier_style_id_suffix_still_get_a_tier(tmp_path: Path):
    """A `#vN-t-<model>` suffix in the source id must not cost the row its tier.

    Regression: the four accepted repair frontier traces carry ids like
    `rep-129-00072#v3-t-google-gemini-3-8-flash`; tier lookup used the unsplit id
    while the tier mapping is keyed by the ground-truth plant, so the whole build
    aborted with "no measurable artefact size".
    """
    row = make_row("wrong_fluid", seed=0)
    base_id = str(row["id"])
    row["id"] = f"{base_id}#v3-t-google-gemini-3-8-flash"

    src = tmp_path / "processed"
    src.mkdir()
    (src / "repair_train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    out, grpo = tmp_path / "sft_tools", tmp_path / "grpo_tools"
    manifest = btc.build([str(src)], str(out), str(grpo),
                         tasks=("repair",), split="train", include_frontier=False,
                         heldout_dir="data/heldout")

    for directory in (out, grpo):
        written = [json.loads(line) for line in
                   (directory / "repair_train.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(written) == 2  # direct + recover
        assert all(r["tier"] in ("easy", "medium", "hard") for r in written)
    assert sum(manifest["tiers"]["repair"]["counts"].values()) == 2


@pytest.mark.corpus
def test_frontier_grpo_row_carries_real_tool_turns(tmp_path: Path, monkeypatch):
    """An accepted frontier trace must not emit an empty `trajectory.turns`.

    The cleaned frontier export already contains the real tool turns; an earlier
    build path hardcoded `turns: []` for every frontier row.
    """
    # The pre-CoT frontier corpora were archived on 2026-11-20 and must never be
    # read back into a live path. Build a synthetic frontier row from a real
    # processed repair row instead: a tool call + result followed by the valid
    # ground-truth answer, with a non-empty trajectory.
    root = Path(__file__).resolve().parents[1]
    source = root / "data/processed/repair_train.jsonl"
    if not source.is_file():
        pytest.skip(f"no real processed repair row at {source}")
    base = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
    answer = base["completion"][0]["content"]
    turns = [{"role": "assistant", "content": "Checking the plant as given before proposing edits."},
             {"role": "tool", "content": "<tool_result>residuals ~0</tool_result>"}]
    text = ('<tool_call name="inspect_plant">{"candidate": ""}</tool_call>\n'
            '<tool_result>{"ok": true, "max_mass_residual_kg_s": 0.0}</tool_result>\n'
            + answer)
    frontier_row = dict(base)
    frontier_row["id"] = "synthetic-frontier-toolrow"
    frontier_row["record_id"] = "synthetic-frontier-toolrow"
    frontier_row["completion"] = [{"role": "assistant", "content": text}]
    frontier_row["trajectory"] = {"turns": turns}
    fdir = tmp_path / "frontier"
    fdir.mkdir()
    (fdir / "frontier_repair_train.jsonl").write_text(
        json.dumps(frontier_row) + "\n", encoding="utf-8")
    monkeypatch.setattr(btc, "FRONTIER_DIRS", (str(fdir),))

    src = tmp_path / "processed"
    src.mkdir()
    out, grpo = tmp_path / "sft_tools", tmp_path / "grpo_tools"
    btc.build([str(src)], str(out), str(grpo), tasks=("repair",), split="train",
              include_frontier=True, heldout_dir="data/heldout")

    written = [json.loads(line) for line in
               (grpo / "repair_train.jsonl").read_text(encoding="utf-8").splitlines()]
    frontier_rows = [r for r in written if r.get("source") == "frontier_tools"]
    assert len(frontier_rows) == 1
    assert frontier_rows[0]["trajectory"]["turns"] == frontier_row["trajectory"]["turns"]
    assert frontier_rows[0]["trajectory"]["turns"]


@pytest.mark.corpus
def test_build_trace_prompt_override_changes_tokens_and_over_budget(tmp_path: Path, word_tokenizer):
    row = _row("HARNESS_SMALL", seed=0)
    plain = tool_traces.build_trace(row, "repair", variant="direct", strict=False)
    big_prompt = [{"role": "system", "content": " ".join(["h"] * 20000)}] + row["prompt"]
    override = tool_traces.build_trace(row, "repair", variant="direct", strict=False,
                                       prompt_override=big_prompt, max_tokens=1000)
    assert override["tokens"] > plain["tokens"]
    assert override["prompt_tokens"] > plain["prompt_tokens"]
    assert override["over_budget"] is True
    # original measurement is still reported for comparison
    assert override["plain_tokens"] == plain["tokens"]
