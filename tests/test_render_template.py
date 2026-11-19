"""Real-tokenizer check of the measured prefix property that TRAIN_ON_REASONING relies on.

Uses the locally cached Qwen3.5-4B tokenizer only (HF_HUB_OFFLINE=1,
local_files_only=True). Skips cleanly when the tokenizer is not cached.
"""
from __future__ import annotations

import os

import pytest

from thermo_agent.train import render

MODEL = "Qwen/Qwen3.5-4B"

# A stored one-message assistant turn exactly as the teacher harness writes it.
TRAJECTORY = (
    "<think>\nInventory the plant, then check the hot-side fluid.\n</think>\n"
    '<tool_call name="inspect_plant">{"candidate": ""}</tool_call>\n'
    '<tool_result>{"ok": true, "solved": true, "failed_checks": ["fluid mismatch"]}</tool_result>\n'
    "<diagnosis>The hot-side fluid is wrong.</diagnosis>\n"
    "<edits>\nFLUID CH.hot_in = water\n</edits>"
)

STORED = [
    {"role": "system", "content": "You repair plants."},
    {"role": "user", "content": "Repair this plant."},
    {"role": "assistant", "content": TRAJECTORY},
]


@pytest.fixture(scope="module")
def tok():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"transformers not installed: {exc}")
    try:
        return AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except Exception as exc:  # pragma: no cover - depends on the local cache
        pytest.skip(f"{MODEL} tokenizer is not cached locally: {exc}")


def test_chat_kw_uses_reasoning():
    assert render.TRAIN_ON_REASONING is True
    assert render.CHAT_KW["enable_thinking"] is True


def test_full_render_starts_with_generation_prompt(tok):
    full = tok.apply_chat_template(STORED, tokenize=False, **render.CHAT_KW)
    prompt = render.generation_prompt(tok, STORED[:2])
    assert prompt, "generation prompt must not be empty"
    assert full.startswith(prompt), (
        "enable_thinking=True must keep the generation prompt a literal prefix; "
        f"prompt tail {prompt[-60:]!r}, render at boundary {full[len(prompt)-20:len(prompt)+40]!r}"
    )


def test_tool_and_think_tags_survive_verbatim(tok):
    full = tok.apply_chat_template(STORED, tokenize=False, **render.CHAT_KW)
    for tag in ("<think>", "</think>", '<tool_call name="inspect_plant">', "</tool_call>",
                "<tool_result>", "</tool_result>"):
        assert tag in full, f"{tag} was altered by the chat template"
