# Per-row trace review — 2026-11-20 — `data/sft_tools_v3/ledger_train.jsonl`

Scope: exactly 30 rows at 1-based lines 1, 41, 81, ..., 1161 (0-based 0, 40, ..., 1160). Each row was
read in full (all fields: id, prompt, completion, provenance, tier, ledger, etc.), one at a time.
Checks applied: (1) reasoning before every post-first tool call cites numbers/ids from the previous
result, opener may be hypothesis/inspection; (2) `<tool_result>` payloads are solver output with
`recomputed_MW`/`residual_W`, no `truth_MW`/`failed_lines`/per-line `ok` (F4) and no answer in first
tool call or prompt; (3) completion ends with `<diagnosis>` citing MW values and `<fix>` with FIX
lines, identical to the validated final answer; (4) tier in easy/medium/hard, variant direct/recover,
no top-level answer/scoring fields; (5) completion non-empty and self-consistent.

`1 | led-0-00000-dt-direct-c3c2 | medium | direct | OK | Opener inspects; later think cites E[fuel] 52.317934->38.200854 MW (residual 14117080.669 W) from inspect_plant, whose output has only recomputed_MW/residual_W, and final <diagnosis>/<fix> restores m[fuel] and ED[HX] exactly as validity_check verified.`
`41 | led-0-00553-dt-direct-c3c2 | medium | direct | OK | Pre-check think cites E[cw_in] 405.055843->0.0 MW, tool output leaks no truth/failed_lines, and the final FIX E[cw_in] matches the accepted validity_check candidate.`
`81 | led-0-01244-dt-direct-c3c2 | medium | direct | OK | Think cites ED[CH] 6.262391->3.512692 MW and closure residual, inspect_plant carries recomputed_MW/residual_W only, and final <fix> removes the 0.001 factor as validated.`
`121 | led-0-01824-dt-direct-c3c2 | hard | direct | OK | Think cites E[fw] 1.297633->0.012186 MW and ED[HRSG] -33.822027->7.768029 MW, no answer leak in tool output, and both final FIX lines match the validated candidate.`
`161 | led-0-02458-dt-direct-c3c2 | easy | direct | OK | Think cites E[fuel] 147822.267758->147.822268 MW, tool result shows recomputed_MW/residual_W without truth fields, and final FIX restores /1e6 as validated.`
`201 | led-0-03046-dt-direct-c3c2 | hard | direct | OK | Think cites ED[GT] -48118.544913->4.307212 MW, inspect_plant output is clean solver data, and final FIX ED[GT] matches the verified candidate.`
`241 | led-0-03726-dt-direct-c3c2 | hard | direct | OK | Think cites E[dh_out] 69.371403->1.153218 MW, no truth_MW/failed_lines/per-line ok in tool result, and final FIX E[dh_out] matches validity_check.`
`281 | led-0-04562-dt-direct-c3c2 | easy | direct | OK | Think cites ED[DC] 37.506927->37.190696 MW, tool output has recomputed_MW/residual_W only, and final FIX ED[DC] matches the validated answer.`
`321 | led-0-05073-dt-direct-c3c2 | hard | direct | OK | Think cites ED[EVA] 3.722394->0.579059 MW, tool result is clean, and final FIX ED[EVA] matches the accepted candidate.`
`361 | led-0-05669-dt-direct-c3c2 | easy | direct | OK | Think cites E[s5] 40.261303->44.096943 MW and ED[DC] residual -1142943.415 W, output leaks no answer, and both final FIX lines match validity_check.`
`401 | led-0-06206-dt-direct-c3c2 | easy | direct | OK | Think cites E[air] 358.719595->0.0 MW, inspect_plant shows recomputed_MW/residual_W only, and final FIX E[air] matches the verified candidate.`
`441 | led-0-06764-dt-direct-c3c2 | easy | direct | OK | Think cites E[chw_ret] 81.882145->0.264897 MW, no F4 leak in tool result, and final FIX E[chw_ret] matches validity_check.`
`481 | led-0-07362-dt-direct-c3c2 | medium | direct | OK | Think cites ED[REJ] 0.093701->1.852003 MW, clean solver output, and final FIX ED[REJ] removing 0.001 matches the verified answer.`
`521 | led-0-07869-dt-direct-c3c2 | hard | direct | OK | Think cites E[chw_sup] 20.86306->0.46329 MW and ED[CH] -19.01941->1.447309 MW, output clean, and both final FIX lines match validity_check.`
`561 | led-0-08412-dt-direct-c3c2 | hard | direct | OK | Think cites ED[GT] -95.486459->1.772342 MW and ED[EVA] -3.814792->0.351131 MW, no answer leak, and final two FIX lines match the verified candidate.`
`601 | led-0-00000-dt-recover-c3c2 | medium | recover | OK | Recover trace: probe think cites prior residuals, post-error think cites the 'no <fix> block' rejection plus E[fuel]/ED[HX] MW values, tool outputs leak no answer, and final XML matches validated candidate.`
`641 | led-0-00553-dt-recover-c3c2 | medium | recover | OK | Recover trace: numbered reasoning cites E[cw_in] 405.055843->0.0 MW before probe and the checker error before the fix, no answer field leaked, final <fix> matches validity_check.`
`681 | led-0-01244-dt-recover-c3c2 | medium | recover | OK | Recover trace: probe think cites ED[CH] 6.262391->3.512692 MW, rejection is real checker output, and final FIX ED[CH] matches the accepted candidate.`
`721 | led-0-01824-dt-recover-c3c2 | hard | recover | OK | Recover trace: reasoning cites E[fw] and ED[HRSG] residuals then the checker error, inspect_plant shows only recomputed_MW/residual_W, and final two FIX lines match validity_check.`
`761 | led-0-02458-dt-recover-c3c2 | easy | recover | OK | Recover trace: think cites E[fuel] 147822.267758->147.822268 MW and the no-fix error, no truth leak, and final FIX restoring /1e6 matches the verified answer.`
`801 | led-0-03046-dt-recover-c3c2 | hard | recover | OK | Recover trace: think cites ED[GT] -48118.544913->4.307212 MW and the rejection, tool outputs are clean, and final FIX ED[GT] matches validity_check.`
`841 | led-0-03726-dt-recover-c3c2 | hard | recover | OK | Recover trace: think cites E[dh_out] 69.371403->1.153218 MW and the checker error, no F4 leak, and final FIX E[dh_out] matches the accepted candidate.`
`881 | led-0-04562-dt-recover-c3c2 | easy | recover | OK | Recover trace: think cites ED[DC] 37.506927->37.190696 MW and the rejection, tool result has no answer fields, and final FIX ED[DC] matches validity_check.`
`921 | led-0-05073-dt-recover-c3c2 | hard | recover | OK | Recover trace: think cites ED[EVA] 3.722394->0.579059 MW and the checker error, no truth/failed_lines leak, and final FIX ED[EVA] matches the verified answer.`
`961 | led-0-05669-dt-recover-c3c2 | easy | recover | OK | Recover trace: think cites E[s5] 40.261303->44.096943 MW and the rejection, output is clean solver data, and both final FIX lines match validity_check.`
`1001 | led-0-06206-dt-recover-c3c2 | easy | recover | OK | Recover trace: think cites E[air] 358.719595->0.0 MW and the no-fix error, no answer leak in first tool call or prompt, and final FIX E[air] matches the accepted candidate.`
`1041 | led-0-06764-dt-recover-c3c2 | easy | recover | OK | Recover trace: think cites E[chw_ret] 81.882145->0.264897 MW and the checker error, tool outputs clean, and final FIX E[chw_ret] matches validity_check.`
`1081 | led-0-07362-dt-recover-c3c2 | medium | recover | OK | Recover trace: think cites ED[REJ] 0.093701->1.852003 MW and the rejection, no answer fields exposed, and final FIX ED[REJ] matches the verified candidate.`
`1121 | led-0-07869-dt-recover-c3c2 | hard | recover | OK | Recover trace: think cites E[chw_sup] 20.86306->0.46329 MW and ED[CH] -19.01941->1.447309 MW plus the checker error, and both final FIX lines match validity_check.`
`1161 | led-0-08412-dt-recover-c3c2 | hard | recover | OK | Recover trace: think cites ED[GT] and ED[EVA] residuals plus the no-fix rejection, no truth_MW/failed_lines in tool results, and final two FIX lines match the accepted candidate.`

## Summary

- Rows reviewed: 30
- OK: 30
- CONCERN: 0
- Defect classes: none

Notes (non-blocking observations, not verdict-changing defects): (a) every row carries a `ledger`
source-metadata field containing the generator's ground truth (`truth_E`, `truth_ED`, `faulty`,
`e_fuel_mw`); it is provenance from `data/processed` and is absent from the prompt, the first tool
call, and every `<tool_result>`, so no answer location is exposed to the student; (b) in the 15
`recover` rows the closing narration ("What the final checks confirm: the tool rejected the input ...
all checks pass") redundantly mentions the empty-probe rejection alongside the final pass, but the
final `<diagnosis>`/`<fix>` is correct and matches the validated answer; (c) 15 rows are `direct`
(lines 1-561) and 15 are `recover` (lines 601-1161), all tiers valid.
