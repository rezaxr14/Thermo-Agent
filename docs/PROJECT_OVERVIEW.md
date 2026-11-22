# PROJECT_OVERVIEW - what this repository is

## Current truth (2026-11-22)

This repository builds verifier-grounded data and training corpora for an agent that
diagnoses, repairs and designs simulated energy plants for AI data centres. A solver
and physics verifiers judge every answer; there is no answer-oracle tool.

## Tasks

- **repair**: diagnose a broken plant and emit `<diagnosis>` + `<edits>`.
- **design**: choose a plant for a site and emit `<rationale>` + `<design>`.
- **audit**: rank component exergy destruction, propose a change and predict its fuel
  effect; emit `<analysis>`, `<ranking>`, `<change>`, `<prediction>`.
- **ledger**: repair a broken exergy ledger; emit `<diagnosis>` + `<fix>`.

## Data flow

1. `data/processed` holds ground-truth rows. It is never rewritten by the training
   pipeline and never mixed with `data/archive`.
2. `synth/tool_traces.py` builds deterministic traces by calling the real teacher
   toolkit and recording its verbatim JSON results.
3. Hosted frontier rows are produced by budget models through the kbench adapter, then
   validated and ingested by `teacher/transfer.py`.
4. `scripts/build_tool_corpus.py` writes SFT and GRPO tool corpora; the 8,192-token gate
   is measured on the exact tools-rendered conversation.
5. `scripts/mix_tool_corpora.py` combines deterministic rows, accepted frontier rows
   and the plain A-slice into `sft_mix_v3` / `grpo_mix_v3`.
6. `scripts/dataset_stats.py`, `audit_harness_corpus.py` and
   `audit_provenance_chain.py` verify the result.

## Physics and verification

Plants are graphs of components and streams. The verifier checks mass and energy
balance, exergy closure and non-negative destruction, temperature crosses, second-law
and Carnot bounds, units/ranges, grounded numbers and site requirements. All numbers in
an answer must be numbers the solver reproduces.

## Protocol

Training, eval and hosted traces all use the native Qwen3.5 structured tool protocol in
`thermo_agent/agent/tool_protocol.py`. See `DECISIONS.md` for the binding rules and
`EVAL_PROTOCOL.md` for the prompt and parsing details.
