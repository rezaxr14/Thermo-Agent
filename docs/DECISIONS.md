# DECISIONS - binding decisions

## Current truth (2026-11-22)

All binding decisions are listed below. Superseded options were deleted, not marked.
Newest entry wins where two entries touch the same knob. `STATUS.md` holds state; this
file holds rules.

## Protocol and training data

1. One native Qwen3.5 structured tool protocol everywhere. Assistant tool calls are
   `tool_calls`; results are `role="tool"` messages; rendering uses
   `apply_chat_template(tools=...)`; parsing uses `tokenizer.parse_response` with the
   shared response template. Legacy `<tool_call name=...>` is rejected.
2. Prompt contract text contains limits and the final-answer format only. The chat
   template owns the call syntax and the 16 tool schemas; no prompt duplicates them.
3. Tool results are masked in SFT loss; tool calls stay trainable.
4. The 8,192-token gate measures exactly the text training tokenizes:
   `prompt + completion`, where `prompt` is rendered through
   `apply_chat_template(tools=<the 16 shared schemas>, add_generation_prompt=True)` and
   `completion` is the template suffix (see `train.render.render_pair`), plus a
   96-token chat reserve. That is one full render of `prompt + completion` with the same
   schemas; tokenizing prompt and completion separately, or omitting the schemas, is a
   different and shorter sequence and must not be used. Over-gate rows are dropped,
   never truncated or re-shortened.
5. All 16 tools are on by default in GRPO, eval and the hosted runner. `include_dynamic`
   exists only as an explicit test opt-out.
6. Every corpus rebuild starts from `data/processed` (and explicitly accepted frontier
   rows). `data/archive/` is never mixed in. Ground-truth answers are never pasted into
   tool results. A stored tool result is the served `output` JSON; cleaning returns an
   unchanged payload verbatim, never `None` or an empty string, and a trace with an
   empty result is a bug, not training data.

## Teacher and reasoning

7. The hosted TEACHER tool schema adds a required `reason` string to each tool; the
   student schema stays candidate-only. At ingest the reason becomes `<think>` text and
   is stripped from stored call arguments. Provider/visible reasoning takes precedence.
8. The reasoning gate requires >=75% reasoned tool turns for an accepted trace; luna's
   >=90% target is a target, not a hard reject.
9. No accepted frontier row may enter a corpus without a tool-using trajectory.

## Task-specific correctness

10. Audit prompts state the rankable ids explicitly: exactly the `ed_json` keys, i.e.
    the plant components minus the combustor, the gas engine and the IT load
    (`dc_load`). Ids outside that set, e.g. `DC`, are rejected by the verifier.
11. A design final answer is the last candidate whose `validity_check` returned
    `ok: true`, enforced in the eval harness and at ingest.
12. Ledger reasoning renders a compact `(line, value, recomputed, residual)` table and
    never points at a "faulty line" directly. A ledger answer must contain at least one
    tool call: a zero-tool ledger row is rejected at ingest even when its text passes the
    ledger verifier, with or without `--require-tools`.

## Operating rules

13. Budget models only: `gpt-6-luna`, `claude-haiku-5-5`. No premium or bulk model run.
14. No training, SFT, GRPO or full eval until explicitly authorised. The only permitted
    hosted activity is the 40-row luna/haiku pilot.
15. Credentials come only from repo-root `.env`; tokens are never printed.
16. After each numbered step: update the owning doc, regenerate dataset stats, one
    commit, run tests, and report anything not runnable. Stop after step 6 and wait.
