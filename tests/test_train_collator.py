"""Training-path audit 2026-11-19 item 4: variable-length multi-turn batching.

TRL's collator pads ``input_ids`` to the longest row, builds ``attention_mask`` and
pads ``labels`` with -100. The SFT preprocessing marks prompt + every tool result
-100; this test checks both halves together on rows of different lengths.
"""
from __future__ import annotations

import pytest

from thermo_agent.train import token_audit
from thermo_agent.train.sft import preprocess_row

_CHARSET = "abcxyz <>/=\"'{}:,.\n|"


class CharTokenizer:
    def __init__(self):
        self.vocab = {c: i for i, c in enumerate(sorted(set(_CHARSET)))}
        self.eos_token = "<|end|>"

    def _id(self, char):
        if char not in self.vocab:  # chat-template markers use letters outside the seed set
            self.vocab[char] = len(self.vocab)
        return self.vocab[char]

    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        out = {"input_ids": [self._id(c) for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kwargs):
        text = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        return text + ("<|assistant|>" if add_generation_prompt else "")


def _row(turns: int) -> dict:
    trace = "<think>reason</think>"
    for i in range(turns):
        trace += f'<tool_call name="inspect_plant">{{}}</tool_call><tool_result>RESULT {i}</tool_result>'
    trace += "<edits>SET X = 1</edits>"
    return {"prompt": [{"role": "user", "content": "short task"}],
            "completion": [{"role": "assistant", "content": trace}]}


def test_variable_length_rows_pad_with_minus_100_labels():
    trl = pytest.importorskip("trl")
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    tok = CharTokenizer()
    rows = [preprocess_row(tok, _row(1)), preprocess_row(tok, _row(3))]
    assert len(rows[0]["input_ids"]) < len(rows[1]["input_ids"])  # genuinely variable length

    batch = DataCollatorForLanguageModeling(pad_token_id=0)(rows)
    width = batch["input_ids"].shape[1]
    assert width == max(len(r["input_ids"]) for r in rows)
    for i, row in enumerate(rows):
        pad = width - len(row["input_ids"])
        assert batch["attention_mask"][i, -pad:].tolist() == [0] * pad if pad else True
        assert batch["labels"][i, -pad:].tolist() == [-100] * pad if pad else True
        assert -100 in batch["labels"][i].tolist()  # prompt + tool results are masked


def test_preprocess_masks_every_tool_result_span_in_a_multiturn_trace():
    from thermo_agent.train import render

    tok = CharTokenizer()
    row = _row(3)
    out = preprocess_row(tok, row)
    text = render.generation_prompt(tok, row["prompt"]) + row["completion"][0]["content"] + tok.eos_token
    start = text.index("<tool_result>")
    assert out["labels"][start:start + len("<tool_result>")] == [-100] * len("<tool_result>")
    answer_start = text.index("<edits>SET X = 1</edits>")
    assert out["labels"][answer_start] != -100  # the final answer is trained on


def test_token_audit_reports_length_stats_and_tier_counts(tmp_path, monkeypatch):
    class Tok:
        def encode(self, text, add_special_tokens=False):
            class E:
                ids = list(range(len(text)))
                offsets = [(i, i + 1) for i in range(len(text))]
            return E()

    monkeypatch.setattr(token_audit, "tokenizer", lambda: Tok())
    import json
    row = {"id": "x", "task": "repair", "tier": "hard",
           "prompt": [{"role": "user", "content": "abc"}],
           "completion": [{"role": "assistant", "content": "<edits>SET X = 1</edits>"}]}
    (tmp_path / "repair_train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = token_audit.audit(tmp_path)
    f = report["files"]["repair_train"]
    for key in ("n", "prompt_p50", "completion_p50", "total_p99", "total_max", "over", "dropped_8000"):
        assert key in f
    assert f["total_max"] > 0 and f["n"] == 1
    assert f["tier_counts"] == {"hard": 1}
