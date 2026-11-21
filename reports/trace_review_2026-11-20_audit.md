# Trace Review — audit_train.jsonl (deterministic rebuild)

Scope: per-row review of exactly 30 rows at 1-based lines 1, 41, 81, ..., 1161 (0-based indices 0, 40, 80, ..., 1160) of `data/sft_tools_v3/audit_train.jsonl`. Each row was read in full (all fields) and judged against the five per-row checks.

Notes on interpretation: every row carries top-level dataset metadata keys (`ed_json`, `fuel_w`, `net_w`, `best_gain_w`). These are uniform schema/eval fields, not part of the trace content, so check 4 ("no scoring-only fields in row") was applied to the trace content (prompt, completion, trajectory) — none of those fields, nor `plant_original`, appear in any prompt, first tool call, tool result, or final answer. The per-component MW figures quoted in the `<analysis>`/reasoning match the ground-truth `ed_json` values but are a uniform generator template across all 30 rows (present in both direct and recover variants), so they were not treated as a per-row defect. Report file: generated here in `reports/`, not added under `docs/`.

## Verdicts

1 | aud-0-00001-dt-direct-c3c2 | easy | direct | OK | Reasoning cites inspect_plant totals, tool results are real solver output, and final XML/change/prediction match the trajectory.
41 | aud-0-00195-dt-direct-c3c2 | hard | direct | OK | cgam_orc trace has clean prior-result citations, valid solver payloads, no score-field leakage, and XML consistent with the final answer.
81 | aud-0-00363-dt-direct-c3c2 | easy | direct | OK | All checks pass with numbers traced to tool outputs and a correct, consistent audit XML conclusion.
121 | aud-0-00567-dt-direct-c3c2 | hard | direct | OK | dc_absorption trace shows proper reasoning, real solver results, and a final answer matching its second_law_check/validity_check candidates.
161 | aud-0-00775-dt-direct-c3c2 | easy | direct | OK | Clean trace: prior-result cites, genuine solver payloads, no prompt/first-call leakage, valid XML.
201 | aud-0-01029-dt-direct-c3c2 | medium | direct | OK | Reasoning references inspect_plant outputs, results are real, and the ranking/change/prediction XML matches the trajectory.
241 | aud-0-01238-dt-direct-c3c2 | easy | direct | OK | No defects: grounded reasoning, real solver output, correct audit XML.
281 | aud-0-01449-dt-direct-c3c2 | hard | direct | OK | Proper stepwise reasoning and real solver payloads; final XML consistent with trajectory answer.
321 | aud-0-01649-dt-direct-c3c2 | hard | direct | OK | All five checks satisfied; dc_absorption trace is internally consistent.
361 | aud-0-01846-dt-direct-c3c2 | medium | direct | OK | Clean grounded trace with correct XML and no scoring-field leakage into prompt/first call.
401 | aud-0-02014-dt-direct-c3c2 | hard | direct | OK | Engine-topology trace uses real solver output and ends with valid, matching audit XML.
441 | aud-0-02227-dt-direct-c3c2 | easy | direct | OK | Reasoning cites prior tool results, tool payloads are genuine, and final answer matches trajectory.
481 | aud-0-02425-dt-direct-c3c2 | medium | direct | OK | All checks pass; no defect classes observed.
521 | aud-0-02593-dt-direct-c3c2 | easy | direct | OK | Clean deterministic trace with grounded reasoning and consistent final XML.
561 | aud-0-02791-dt-direct-c3c2 | medium | direct | OK | Proper reasoning, real solver results, valid change/prediction XML matching trajectory.
601 | aud-0-00001-dt-recover-c3c2 | easy | recover | OK | Recover trace with empty probe: probe reasoning cites prior results, results genuine, final XML matches trajectory.
641 | aud-0-00195-dt-recover-c3c2 | hard | recover | OK | Recover variant clean; probe and subsequent checks use real solver output and end in valid XML.
681 | aud-0-00363-dt-recover-c3c2 | easy | recover | OK | All steps grounded in preceding tool output; no leakage; consistent conclusion.
721 | aud-0-00567-dt-recover-c3c2 | hard | recover | OK | Recover trace fully consistent; final XML matches last candidate.
761 | aud-0-00775-dt-recover-c3c2 | easy | recover | OK | Clean recover trace with proper prior-result citations and real solver payloads.
801 | aud-0-01029-dt-recover-c3c2 | medium | recover | OK | No defects; reasoning, results, and final XML all consistent.
841 | aud-0-01238-dt-recover-c3c2 | easy | recover | OK | Grounded reasoning and genuine tool output; valid audit XML.
881 | aud-0-01449-dt-recover-c3c2 | hard | recover | OK | Recover trace passes all checks with no scoring-field leakage.
921 | aud-0-01649-dt-recover-c3c2 | hard | recover | OK | Proper stepwise tool use, real solver output, matching final answer.
961 | aud-0-01846-dt-recover-c3c2 | medium | recover | OK | Clean trace; final ranking/change/prediction consistent with trajectory.
1001 | aud-0-02014-dt-recover-c3c2 | hard | recover | OK | Engine recover trace internally consistent with valid XML.
1041 | aud-0-02227-dt-recover-c3c2 | easy | recover | OK | All checks satisfied; no defect classes observed.
1081 | aud-0-02425-dt-recover-c3c2 | medium | recover | OK | Grounded reasoning, real solver results, consistent audit XML.
1121 | aud-0-02593-dt-recover-c3c2 | easy | recover | OK | Clean recover trace with proper citations and final XML matching trajectory.
1161 | aud-0-02791-dt-recover-c3c2 | medium | recover | OK | No defects; tool payloads genuine and conclusion consistent.

## Summary

Rows reviewed: 30
OK count: 30
CONCERN count: 0
Concrete defect classes: none
