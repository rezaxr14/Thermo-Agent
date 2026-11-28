# STATUS - current state (2026-11-27)

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
teacher add-on is a required `reason`, which ingest moves into `</think>` and strips from
arguments.

Task contracts now match their validators: an audit ranking may use exactly the
`ed_json` ids (plant components minus combustor, gas engine and the IT load, so never
`DC`), and a ledger answer must contain at least one `inspect_plant` call. Cleaned
traces keep the served tool output: `compact_result` returns an unchanged payload
verbatim, never `None` or a bare hint string.

## Corpora

The 8,192 gate measures the exact training text: harness prompt + completion, one
template render with the 16 serving schemas, plus the 96-token reserve. Over-gate rows
are dropped, never truncated.

| Corpus | Rows |
| --- | ---: |
| `data/sft_tools_v3` | 4,800 |
| `data/grpo_tools_v3` | 4,800 |
| `data/sft_tools_v3_h` | 4,778 |
| `data/grpo_tools_v3_h` | 4,778 |
| `data/sft_mix_v3` | 6,410 |
| `data/grpo_mix_v3` | 4,810 |
| `data/frontier_tools_accepted` | 32 |

Deterministic retention after the step-4 prompt fix: 4,778 / 4,800 = 99.54% (repair
100%, design 100%, audit 99.08%, ledger 99.08%; worst tiers audit/hard and ledger/hard
97.25%). All 22 drops are hard-tier. The two extra ledger drops versus step 3 are the
cost of the explicit "ledger needs a tool call" sentence. Accepted frontier rows: 32 / 32
under the gate, max 7,306 tokens. Evidence: `reports/step4_audit_format.md`,
`reports/gate_retention_step4.json`, `reports/harness_corpus_audit_step4.json`.

Tool results are compact: `solve_plant` returns `columns` + `rows`; ledger tools return
`line_table` + `line_errors`; skill bodies are budgeted; stored frontier results are
re-served through the same compaction.

## Verification

- Full suite after step 4: **1,188 passed, 9 skipped** (1,072.0 s,
  `$env:LLM_DEFAULT="gpt-6-luna"; .venv/Scripts/python.exe -m pytest -q`). The 9 skips
  are data-gated, not failures: 8 in `tests/test_final_mix_tiers_provenance.py` ("v2 mix
  not built yet") and 1 in `tests/test_mix_tool_corpora.py` ("needs the local corpora
  (pre-CoT tool corpora are archived under `data/archive/`)"). No test failed to run.
- `scripts/audit_harness_corpus.py` over `sft_tools_v3_h`, `grpo_tools_v3_h`,
  `sft_mix_v3`, `grpo_mix_v3`: **20,776 rows, 0 over 8,192 (rendered), 0 over 8,192
  (pipeline), 0 anomalies**.
- `scripts/audit_provenance_chain.py --strict`: **PROVENANCE OK (9/9)** - 30,408 rows
  scanned for protocol markers, 11,220 mix rows inspected, 0 leaks.

## Work state

- Hard-fix steps 1-4 committed: `d81ca1f` (native protocol), `bda80ab` (native pilot),
  `80f3c6a` (exact gate), and this step (audit ids, ledger tool flow, tool-result
  cleaning).
- Skill docs: `SKILL.md` is the repo-agnostic Kaggle/HF guide; `KAGGLE_SKILL.md` and
  `HF_KAGGLE_SKILL.md` hold the project-specific commands and repo ids.
- Step 5 is next: reasoning quality gate (>=40 chars, distinct, cite earlier numbers).

## Still not done

- Steps 5-8 of the plan: reasoning gate, whole-row reviews, docs/test hygiene, and the
  audit+ledger re-pilot.
- No training, SFT, GRPO or full eval has been launched.
