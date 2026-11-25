# STATUS - current state (2026-11-25)

One-page truth. This file describes the working tree now, not its history. Binding rules
are in `DECISIONS.md`; evidence is in `AUDITS.md`.

## Repository

- Checkout: `C:\Users\rezax\Home\Code\thermo_agent\Repo` (git `main`).
- Python: `.venv/Scripts/python.exe`; Kaggle CLI: `.venv/Scripts/kaggle.exe`.
- Credentials only from repo-root `.env`; tokens are never printed.
- Proxy for hosted calls: `socks5h://127.0.0.1:10808`.
- kbench test collection needs `LLM_DEFAULT=gpt-6-luna` (import sentinel only, no call).
- Docs: exactly 10 files, under 150 KB. `data/archive/` is never mixed into a corpus.

## Protocol

One native Qwen3.5 structured tool protocol everywhere: assistant `tool_calls` and
`role="tool"` results, rendered with `apply_chat_template(tools=...)` and parsed with
`tokenizer.parse_response` via `thermo_agent/agent/tool_protocol.py`. The legacy
`<tool_call name=...>` form is rejected, and archived rows are never migrated. All 16
tools are on by default; the serving schema is the short contract (1,247 tokens). The
teacher add-on is a required `reason`, which ingest moves into `<think>` and strips from
arguments.

## Corpora

The 8,192 gate measures the exact training text: harness prompt + completion, one
template render with the 16 serving schemas, plus the 96-token reserve. Over-gate rows
are dropped, never truncated.

| Corpus | Rows |
| --- | ---: |
| `data/sft_tools_v3_h` | 4,780 |
| `data/grpo_tools_v3_h` | 4,780 |
| `data/sft_mix_v3` | 6,410 |
| `data/grpo_mix_v3` | 4,810 |
| `data/frontier_tools_accepted` | 30 |

Deterministic retention: 4,780 / 4,800 = 99.58% (repair 100%, design 100%, audit
99.08%, ledger 99.25%; worst tier audit/hard 97.25%). Accepted frontier rows: 30 / 30
under the gate, max 7,369 tokens. Evidence: `reports/step3_row_retention.md`,
`reports/gate_retention_step3.json`, `reports/harness_corpus_audit_step3_after.json`.

Tool results are compact: `solve_plant` returns `columns` + `rows`; ledger tools return
`line_table` + `line_errors`; skill bodies are budgeted; stored frontier results are
re-served through the same compaction.

## Verification

- Full suite green after step 3: **1,183 passed, 9 skipped** (1,178.7 s). The 9 skips are
  data-gated, not failures: 8 in `tests/test_final_mix_tiers_provenance.py` ("v2 mix not
  built yet") and 1 in `tests/test_mix_tool_corpora.py` ("pre-CoT tool corpora are archived
  under `data/archive/`"). No test failed to run.
- `scripts/audit_provenance_chain.py --strict`: **PROVENANCE OK (9/9)** after the final
  step-3 rebuild - 30,410 rows scanned for protocol markers, 11,220 mix rows inspected,
  0 leaks.

## Work state

- Hard-fix steps 1-3 committed: `d81ca1f` (native protocol), `bda80ab` (native pilot),
  plus the step-3 retention commit.
- Skill docs: `SKILL.md` is the repo-agnostic Kaggle/HF guide; `KAGGLE_SKILL.md` and
  `HF_KAGGLE_SKILL.md` hold the project-specific commands and repo ids.
- Step 4 is next: audit final-answer format, haiku ledger zero tool calls.

## Still not done

- Step 4 audit/ledger fixes; steps 5-8 of the plan.
- No training, SFT, GRPO or full eval has been launched.
