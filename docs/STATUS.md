# STATUS - current state (2026-11-22)

One-page truth. This file describes the working tree now, not its history. Binding rules
are in `DECISIONS.md`; evidence is in `AUDITS.md`.

## Repository

- Checkout: `C:\Users\rezax\Home\Code\thermo_agent\Repo` (git `main`).
- Python: `.venv/Scripts/python.exe`; Kaggle CLI: `.venv/Scripts/kaggle.exe`.
- Credentials only from repo-root `.env`; tokens are never printed.
- Proxy for hosted calls: `socks5h://127.0.0.1:10808`.
- kbench test collection needs `LLM_DEFAULT=gpt-6-luna` (import sentinel only, no call).
- Docs: exactly 10 files, 31 KB total. `data/archive/` is never mixed into a corpus.

## Protocol

One native Qwen3.5 structured tool protocol everywhere: assistant `tool_calls` and
`role="tool"` results, rendered with `apply_chat_template(tools=...)` and parsed with
`tokenizer.parse_response` via `thermo_agent/agent/tool_protocol.py`. The legacy
`<tool_call name=...>` form is rejected. Contract text contains no call syntax or tool
list. All 16 tools are on by default. The student schema is candidate-only; the teacher
schema requires `reason`, which ingest moves into `<think>` and strips from arguments.

## Corpora

The 8,192-token gate is measured on the exact tools-rendered conversation plus a
96-token chat reserve. Over-gate rows are dropped, never truncated.

| Corpus | Rows |
| --- | ---: |
| `data/sft_tools_v3_h` | 2,198 |
| `data/grpo_tools_v3_h` | 2,198 |
| `data/sft_mix_v3` | 3,804 |
| `data/grpo_mix_v3` | 2,204 |

`reports/harness_corpus_audit_native.json`: 10,404 audited rows, **0 over 8,192**,
**0 anomalies**. `reports/dataset_stats.json` / `.md` are regenerated.
`data/processed` is unchanged.

## Verification

- Full suite: **1,172 passed, 9 skipped** (last run used a benign import sentinel;
  the same 7 kbench tests were separately verified with `LLM_DEFAULT=gpt-6-luna`).
- `scripts/audit_provenance_chain.py --strict`: **PROVENANCE OK, 8/8 checks**.
- 3-step GRPO-style parse smoke in `tests/test_tool_protocol.py` (no model run).

## Work state

- Steps 1-3 complete and committed: `2906d2b`, `9307a46`, `046bcfe`.
- Step 4: `native_pilot` 40-row shards built (10/task, 14/13/13 tiers), local preflight
  40/40 per model, rehearsal 39/39 per model, HF sync done. Kaggle is reachable again;
  bridge kernel pushed (version 16, COMPLETE), but the Kaggle dataset copy is stale and
  dataset preflight returns 404 for the new shards. No hosted rows, no reasoning-share
  measurement, and no bulk run.
- Step 5: 27 one-row repair/design reviews in `reports/trace_review.jsonl`; the four old
  markdown reviews deleted.
- Step 6: docs curated to this 10-file/<=150 KB set with current-truth headers.

## Still not done

- Refresh the Kaggle dataset copy, run only luna/haiku on the existing 40-row shards,
  apply the >=0.85 reasoning-share gate and per-task accepted-row rule, then ingest.
- No training, SFT, GRPO or full eval has been launched.
