"""Render training pairs exactly as GRPO and inference will see the prompt.

GRPO and the evaluator build the prompt with apply_chat_template(prompt, add_generation_prompt=True, enable_thinking=False).
Rendering a whole conversation for SFT can differ from that in the assistant prefix (for example an empty thinking block),
which would train the model on a prefix it never sees later. So SFT uses plain text: prompt = the generation prompt string,
completion = the assistant text."""
from __future__ import annotations

CHAT_KW = {"enable_thinking": False}


def generation_prompt(tok, messages: list[dict]) -> str:
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **CHAT_KW)


def render_pair(tok, row: dict) -> dict:
    return {"prompt": generation_prompt(tok, row["prompt"]), "completion": row["completion"][0]["content"]}


def prefix_report(tok, row: dict) -> dict:
    """Does the whole-conversation render start with the generation prompt? False means conversational SFT would mismatch GRPO."""
    full = tok.apply_chat_template(row["prompt"] + row["completion"], tokenize=False, **CHAT_KW)
    gp = generation_prompt(tok, row["prompt"])
    return {"prefix_matches": full.startswith(gp), "generation_prompt_tail": gp[-80:], "full_render_at_that_point": full[len(gp) - 80: len(gp) + 40] if len(full) >= len(gp) else full[-120:]}
