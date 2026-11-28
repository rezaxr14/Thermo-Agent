# Step 4 - audit answer format and the haiku ledger tool flow

Date: 2026-11-26. Source: the native pilot journal
`data/teacher_runs/kaggle_hosted/budget_batches/native_pilot/.../native_pilot_completions.jsonl`
(10 audit + 10 ledger attempts per model). No new model call was made to diagnose
these rows; the re-pilot is reported at the end.

## Expected audit contract

From `thermo_agent/audit/spec.py` and the compact harness:

```
<analysis>...only numbers the solved plant shows (MW, K)...</analysis>
<ranking>ID1 > ID2 > ID3</ranking>          # 1-5 ids, largest exergy destruction first
<change>SET <component>.<param> = <number></change>
<prediction>fuel_change_mw = <number></prediction>
```

`<ranking>` ids must be the keys of `ed_json` minus the combustor/gas engine. The
validator rejects any other id (`parse_audit(..., set(ed))`). `dc_load` is deliberately
absent from `ed_json`: `audit.truth.ranking` excludes it ("the IT load is not part of the
plant").

## Three raw audit final answers per model

### gpt-6-luna

`aud-0-00046`

```
<analysis>fuel 154.80 MW; net electricity 50.47 MW; cooling 38.24 MW; T0 301.75 K; CH rejection-water rise set to 10.580798805 K</analysis>
<ranking>DC > GT > CH > AC</ranking>
<change>SET CH.dT_rej = 10.580798805</change>
<prediction>fuel_change_mw = 0.4204235</prediction>
```

`aud-0-00057`

```
<analysis>Fuel 136.10 MW; net electricity 38.19 MW; cooling 28.01 MW. DC destruction 37.59 MW; AC 3.43 MW; CH 4.10 MW; REJ 1.76 MW.</analysis>
<ranking>DC > CH > AC > REJ</ranking>
<change>SET REJ.fan_frac = 0.012</change>
<prediction>fuel_change_mw = -0.22</prediction>
```

`aud-0-00059`

```
<analysis>Fuel input is 119.72 MW; CC operates from 304 K to 1496 K. DC cooling is 24.84 MW.</analysis>
<ranking>DC > GT > AC > CH</ranking>
<change>SET CC.q_loss = 0.008</change>
<prediction>fuel_change_mw = -0.24</prediction>
```

### claude-haiku-5-5

`aud-0-00046` (prose inside `<analysis>`; tags and blank lines are accepted by the parser)

```
<analysis>
Baseline solve converges with residuals at zero and all checks passing (V1 to V8c OK). ... Per-component exergy destruction is DC 49.96 MW, GT 9.50 MW, CH 5.64 MW and AC 3.93 MW. The combustor (CC, 50.27 MW) is excluded from ranking. DC is the largest rankable item because ...
</analysis>

<ranking>DC > GT > CH > AC</ranking>

<change>SET CH.dT_rej = 8.0</change>

<prediction>fuel_change_mw = -1.5</prediction>
```

`aud-0-00057`

```
<analysis>
The baseline solves with zero energy, mass and exergy-closure residuals, and all validity checks pass. ... Per-component destruction is DC 37.59 MW, GT 5.74 MW, CH 4.10 MW, AC 3.43 MW and REJ 1.76 MW. The combustor CC (42.97 MW) is the largest destroyer but is not ranked. DC dominates ...
</analysis>

<ranking>DC > GT > CH > AC > REJ</ranking>

<change>SET REJ.fan_frac = 0.012</change>

<prediction>fuel_change_mw = -0.310</prediction>
```

`aud-0-00059`

```
<analysis>
Baseline solve converged with all checks passing: ... Among the rankable components, DC destroys 33.97 MW, GT 5.80 MW, AC 3.56 MW and CH 2.53 MW. ...
</analysis>

<ranking>DC > GT > AC > CH</ranking>

<change>SET CH.T_chw_out = 295.0</change>

<prediction>fuel_change_mw = -2.9</prediction>
```

Both models produce all four tags in the right order. The format failure is not the
tags: every one of these rankings contains `DC`.

## Diagnosis: the prompt advertised an id the validator rejects

The harness built the rankable list from the plant alone:

```python
rankable = [c.id for c in plant.components if c.type not in ("combustor", "gas_engine")]
```

That includes `dc_load`, so the system prompt said:

> Rankable audit ids (the components in ed_json, excluding the combustor/gas engine):
> **AC, GT, DC, CH**. Never rank any other id.

but `ed_json` for the same row is `{"CC": ..., "GT": ..., "CH": ..., "AC": ...}` -
no `DC`. Running the validator on the 20 pilot audit answers gives **18 rows rejected
with `format: unknown component ids ['DC']`** (9 per model), plus 2 rows
(`aud-0-00078`) rejected for `change: change not allowed` and
`prediction: no usable prediction` - a change-selection issue, not the tag format.

The fix derives the advertised ids from the same rule the validator uses: exclude
`combustor`, `gas_engine` **and `dc_load`**. The line now reads:

> Rankable audit ids (the components in ed_json, excluding the combustor/gas engine and
> the IT load): AC, GT, CH. Never rank any other id.

`tests/test_audit_ranking_prompt.py` pins the contract: the advertised set must equal
`set(ed_json) - excluded` and `DC` must not be advertised. Before the fix the equality
fails because `DC` is advertised and absent from `ed_json`.

## Haiku ledger: ten answers, zero tool calls

All ten haiku ledger attempts have exactly one assistant turn and **zero tool calls**
(luna: 1-9 calls). They are not malformed: the ledger prompt carries the whole ledger
and the residual hint, so haiku wrote a full diagnosis directly from the prompt. The
ledger verifier accepted all ten texts (reward 1.0). They were kept out of the accepted
corpus only because `reasoning_report` has no tool turns to score and returned
`ok=False`; that is an accidental catch, not an explicit rule.

Example (`led-0-00152`), zero tool calls:

```
<diagnosis>
The closure residual is 26.41 MW, which equals |W[DC]| = 26.41 MW. The DC line uses
`- 2*W[DC]`, so ED[DC] = 52.85 MW is about twice the physical value. ...
</diagnosis>

<fix>
FIX ED[DC] = E[chw_sup] - E[chw_ret] - W[DC]
</fix>
```

Two fixes:

1. `transfer.ingest` now rejects a ledger row with no tool call **unconditionally**
   (`rejected_no_tool_use:ledger`, `reason:no_tool_use`), even when `--require-tools`
   is not passed and even when the text passes the verifier.
2. The ledger contract now says: "A ledger answer needs at least one inspect_plant
   call; a zero-tool answer is rejected even when the text looks right."

`tests/test_transfer.py::test_ingest_rejects_ledger_without_tool_calls` is the
failing-then-passing test: it feeds an accepted ledger answer with an empty trajectory
and asserts the row is rejected.

Re-ingesting the real native-pilot journals with `--tasks ledger` and **without**
`--require-tools` confirms both directions: luna's 10 ledger attempts are accepted
(`accepted:ledger=10`, `accepted_tool_using=10`), haiku's 10 are rejected
(`rejected_no_tool_use:ledger=10`, `rejected:ledger=10`, 0 accepted). No model call was
made; the journals are the stored pilot answers.

## Corpus rebuild after the prompt fix

The harness prompt text changed, so all four deterministic corpora were rebuilt from
`data/processed` and the mixes re-derived:

| Corpus | Rows |
| --- | ---: |
| `data/sft_tools_v3` | 4,800 |
| `data/grpo_tools_v3` | 4,800 |
| `data/sft_tools_v3_h` | 4,778 |
| `data/grpo_tools_v3_h` | 4,778 |
| `data/sft_mix_v3` | 6,410 |
| `data/grpo_mix_v3` | 4,810 |
| `data/frontier_tools_accepted` | 32 |

Retention at the exact 8,192 gate: **4,778 / 4,800 = 99.54%** (repair 100%, design
100%, audit 99.08%, ledger 99.08%; worst tiers audit/hard and ledger/hard 97.25%). All
22 drops are hard-tier. The ledger prompt gained the zero-tool sentence, which cost two
additional hard ledger rows (8,207 and 8,256 tokens); that is the deliberate price of
stating the rule the ingest now enforces.

`scripts/audit_harness_corpus.py` over the four harness/mix directories reports 20,776
rows, 0 over 8,192 rendered, 0 over 8,192 pipeline, 0 anomalies
(`reports/harness_corpus_audit_step4.json`). `scripts/audit_provenance_chain.py
--strict` reports PROVENANCE OK 9/9 over 30,408 rows and 11,220 mix rows with 0 leaks.
Dataset stats: 231,359 rows, 0 malformed.

## A third bug found while validating the tool flow: tool results were erased

Re-ingesting the journals exposed a data-destroying bug in the step-3 compaction. In the
30 accepted frontier rows, **114 of 158 stored tool messages were the literal string
`None`**; the harness mixes rendered the same results as empty strings. A student trained
on those rows would learn that tool calls return nothing.

Cause: `thermo_agent/teacher/toolkit.py::compact_result` reused its `text` parameter as
the loop variable for the solver-hint rewrite:

```python
for key in ("solver_message", "message"):
    text = payload.get(key)          # clobbers the caller's JSON
...
return json.dumps(payload) if changed else text   # None, or the bare hint
```

Every unchanged payload therefore came back as `None` (or as the bare `message` string),
and `normalize_tool_turns` stored that as the tool result.

Fixes and failing-then-passing tests:

- `hint = payload.get(key)`; the unchanged path returns the original JSON.
  `tests/test_toolkit.py::test_compact_result_returns_an_unchanged_payload_verbatim`.
- `tests/test_tool_protocol.py::test_normalize_tool_turns_keeps_the_served_output` pins
  the cleaned turn to the served output JSON.

Re-ingest (no model call): both journals were re-gated (luna 18 survived / 3 over gate,
haiku 15 / 0) and re-ingested with `--require-tools`. Result: **32 accepted rows** (luna
18: repair 9, design 2, ledger 7; haiku 14: repair 10, design 4), **0 `None` or empty
tool results**, max 7,306 tokens under the gate. `data/frontier_tools_cohort` and
`data/frontier_tools_accepted` were rebuilt from those rows, and both mixes re-derived.

## Re-pilot

Audit + ledger only, luna + haiku, 10 rows each (40 attempts). Results are appended
after the run; no other model or task is re-run.
