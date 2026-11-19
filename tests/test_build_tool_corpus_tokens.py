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
