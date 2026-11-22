# EVAL_PROTOCOL - prompt arms, tool loop and parsing

## Current truth (2026-11-22)

Evaluation and training share one prompt contract and one native tool protocol. No
legacy XML call format exists on any path.

## Prompt arms

- `ours_base` / `ours_base_h`: plain vs harness prompt, no tools in the Z arm.
- `ours_sft` / `ours_sft_h`: SFT-shaped rows with the same compact AGENTS.md block.
- `ours_harness`: the H arm with the shared compact harness and tools enabled.
- The compact harness is built only by
  `thermo_agent.agent.tool_prompt.build_harness_tool_messages(compact=True)`. It
  contains the task answer format, the on-demand skill catalog and the multi-turn
  limits. It does not contain the tool list or call syntax.

## Tool loop

`thermo_agent.eval.model_eval.ToolLoop` runs a bounded propose / call / observe loop.
It exposes all 16 tools through `teacher.toolkit.make_tools` and hands the matching
OpenAI-style schemas to the model renderer. Tool results are appended as native
`role="tool"` messages, followed by a short revision nudge. The episode records tool
names, results and the final text; scoring is only via `teacher.validate.validate`.

`HFGenerator.parse_tool_calls` uses `tool_protocol.parse_generated`, i.e.
`tokenizer.parse_response` with the shared response template. A legacy
`<tool_call name=...>` string parses to zero calls and is therefore never executed.

## Generation and prefix alignment

`train/render.py` is the single renderer:

- `model_messages` keeps only chat fields (`role`, `content`, optional `tool_calls`,
  `name`, `tool_call_id`); row bookkeeping never reaches the tokenizer.
- `completion_messages` accepts structured completions and defaults a missing role to
  `assistant`.
- `row_tools` derives the 16 tool schemas from the row's task.
- `render_pair` renders the generation prompt and the full conversation with
  `tools=...` and requires the former to be a literal prefix of the latter.
- `enable_thinking=True` is mandatory; with `False` the template injects an empty
  closed think block that breaks prefix alignment.

## Masks and rewards

SFT masks the prompt and every `<tool_response>` / `<tool_result>` span by token offset;
tool-call tokens remain trainable. Rewards read the final assistant message only and
reject truncated rollouts; the last message must carry the task's XML tags.

## Task-specific evaluation rules

- Audit: rankable ids are taken from `ed_json`, excluding combustor/gas engine; other
  ids are rejected.
- Design: the final answer is the last candidate whose `validity_check` returned ok.
- Ledger: reasoning is judged against a compact `(line, value, recomputed, residual)`
  table; there is no faulty-line pointer in the prompt.
