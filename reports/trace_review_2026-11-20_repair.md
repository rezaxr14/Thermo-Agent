# Trace Review — repair_train.jsonl (2026-11-20)

File reviewed: `data/sft_tools_v3/repair_train.jsonl`
Rows reviewed: 30, at 1-based lines 1, 41, 81, …, 1161 (indices 0, 40, 80, …, 1160). Each row was read in full (all fields) before assigning a verdict.

Verdict format: `<line_no> | <id> | <tier> | <variant> | OK|CONCERN | <one-sentence reason>`

1 | rep-1-00005-dt-direct-c3c2 | medium | direct | CONCERN | Contains answer/scoring-only fields `plant_original` (with the ground-truth fixed value) and `fault_type`; reasoning, tool outputs and final XML are otherwise sound.
41 | rep-10-00074-dt-direct-c3c2 | easy | direct | CONCERN | Contains answer/scoring-only fields `plant_original` and `fault_type` (`wrong_fluid`); trace and final XML otherwise sound.
81 | rep-19-00064-dt-direct-c3c2 | medium | direct | CONCERN | Contains answer/scoring-only fields `plant_original` and `fault_type` (`wrong_fluid`); trace and final XML otherwise sound.
121 | rep-28-00032-dt-direct-c3c2 | hard | direct | CONCERN | Contains answer/scoring-only fields `plant_original` and `fault_type`; trace and final XML otherwise sound.
161 | rep-38-00007-dt-direct-c3c2 | medium | direct | CONCERN | Contains `plant_original` and `fault_type` (`efficiency_gt_1` mismatches the actual AC.eta_s fix); trace and final XML otherwise sound.
201 | rep-47-00037-dt-direct-c3c2 | easy | direct | CONCERN | Contains `plant_original` and `fault_type` (`efficiency_gt_1` mismatches the actual ENG.eta_el fix); trace and final XML otherwise sound.
241 | rep-56-00044-dt-direct-c3c2 | easy | direct | CONCERN | Contains answer/scoring-only fields `plant_original` and `fault_type` (`wrong_fluid`); trace and final XML otherwise sound.
281 | rep-65-00017-dt-direct-c3c2 | medium | direct | CONCERN | Contains `plant_original` and `fault_type` (`efficiency_gt_1` mismatches the actual AC.eta_s fix); trace and final XML otherwise sound.
321 | rep-73-00002-dt-direct-c3c2 | hard | direct | CONCERN | Contains answer/scoring-only fields `plant_original` and `fault_type` (`efficiency_gt_1` for GT.eta_s); trace and final XML otherwise sound.
361 | rep-78-00017-dt-direct-c3c2 | easy | direct | CONCERN | Contains `plant_original` and `fault_type` (`efficiency_gt_1` label on a CH.eps percentage-slip fix); trace and final XML otherwise sound.
401 | rep-85-00089-dt-direct-c3c2 | easy | direct | CONCERN | Contains answer/scoring-only fields `plant_original` and `fault_type` (`wrong_fluid`); trace and final XML otherwise sound.
441 | rep-96-00082-dt-direct-c3c2 | easy | direct | CONCERN | Contains `plant_original` and `fault_type` (`efficiency_gt_1` label on a CH.eps percentage-slip fix); trace and final XML otherwise sound.
481 | rep-106-00007-dt-direct-c3c2 | medium | direct | CONCERN | Contains `plant_original` and `fault_type` (`efficiency_gt_1` label on an AC.eta_s percentage-slip fix); trace and final XML otherwise sound.
521 | rep-113-00052-dt-direct-c3c2 | medium | direct | CONCERN | Contains `plant_original` and `fault_type` (`efficiency_gt_1` label on a CH.eps fix); trace and final XML otherwise sound.
561 | rep-122-00077-dt-direct-c3c2 | hard | direct | CONCERN | Contains `plant_original` and `fault_type` (`efficiency_gt_1` for OT.eta_s); trace and final XML otherwise sound.
601 | rep-1-00005-dt-recover-c3c2 | medium | recover | CONCERN | Contains answer/scoring-only fields `plant_original` and `fault_type`; recover-style baseline probe and final XML are otherwise sound.
641 | rep-10-00074-dt-recover-c3c2 | easy | recover | CONCERN | Contains answer/scoring-only fields `plant_original` and `fault_type` (`wrong_fluid`); recover trace and final XML otherwise sound.
681 | rep-19-00064-dt-recover-c3c2 | medium | recover | CONCERN | Contains answer/scoring-only fields `plant_original` and `fault_type` (`wrong_fluid`); recover trace and final XML otherwise sound.
721 | rep-28-00032-dt-recover-c3c2 | hard | recover | CONCERN | Contains answer/scoring-only fields `plant_original` and `fault_type`; recover trace and final XML otherwise sound.
761 | rep-38-00007-dt-recover-c3c2 | medium | recover | CONCERN | Contains `plant_original` and `fault_type` (`efficiency_gt_1` mismatches the actual AC.eta_s fix); recover trace and final XML otherwise sound.
801 | rep-47-00037-dt-recover-c3c2 | easy | recover | CONCERN | Contains `plant_original` and `fault_type` (`efficiency_gt_1` mismatches the actual ENG.eta_el fix); recover trace and final XML otherwise sound.
841 | rep-56-00044-dt-recover-c3c2 | easy | recover | CONCERN | Contains answer/scoring-only fields `plant_original` and `fault_type` (`wrong_fluid`); recover trace and final XML otherwise sound.
881 | rep-65-00017-dt-recover-c3c2 | medium | recover | CONCERN | Contains `plant_original` and `fault_type` (`efficiency_gt_1` mismatches the actual AC.eta_s fix); recover trace and final XML otherwise sound.
921 | rep-73-00002-dt-recover-c3c2 | hard | recover | CONCERN | Contains answer/scoring-only fields `plant_original` and `fault_type` (`efficiency_gt_1` for GT.eta_s); recover trace and final XML otherwise sound.
961 | rep-78-00017-dt-recover-c3c2 | easy | recover | CONCERN | Contains `plant_original` and `fault_type` (`efficiency_gt_1` label on a CH.eps percentage-slip fix); recover trace and final XML otherwise sound.
1001 | rep-85-00089-dt-recover-c3c2 | easy | recover | CONCERN | Contains answer/scoring-only fields `plant_original` and `fault_type` (`wrong_fluid`); recover trace and final XML otherwise sound.
1041 | rep-96-00082-dt-recover-c3c2 | easy | recover | CONCERN | Contains `plant_original` and `fault_type` (`efficiency_gt_1` label on a CH.eps percentage-slip fix); recover trace and final XML otherwise sound.
1081 | rep-106-00007-dt-recover-c3c2 | medium | recover | CONCERN | Contains `plant_original` and `fault_type` (`efficiency_gt_1` label on an AC.eta_s percentage-slip fix); recover trace and final XML otherwise sound.
1121 | rep-113-00052-dt-recover-c3c2 | medium | recover | CONCERN | Contains `plant_original` and `fault_type` (`efficiency_gt_1` label on a CH.eps fix); recover trace and final XML otherwise sound.
1161 | rep-122-00077-dt-recover-c3c2 | hard | recover | CONCERN | Contains answer/scoring-only fields `plant_original` and `fault_type` (`efficiency_gt_1` for OT.eta_s); recover trace and final XML otherwise sound.

## Summary

- Rows reviewed: 30
- OK: 0
- CONCERN: 30

Checks 1–3 and 5 passed on every reviewed row: each assistant turn between tool calls cites concrete numbers/identifiers (status strings, MW values, check names) from the preceding `<tool_result>`; all `<tool_result>` payloads are real solver output (`ok`, residuals, check lists, peak temperatures) with the `recover` rows additionally showing a genuine failed baseline probe (`EditError: no <edits> block`); the first tool call (`inspect_plant` with an empty candidate, or the no-edit probe) never leaks the ground-truth answer; every completion is non-empty, consistent with its trajectory, and ends with the required `<diagnosis>` + `<edits>` XML (SET/FLUID lines) matching the last tool-call candidate. `tier` is always one of easy/medium/hard and `provenance.variant` is always direct or recover.

Concrete defect classes found:

1. Answer/scoring-only fields present in every row — `plant_original` (which embeds the ground-truth repaired value, e.g. the corrected `eta_s`/`eps`/fluid) and `fault_type` (the ground-truth fault label). Neither field should appear in the training row; the required checks explicitly forbid `ref_spec`, `ed_json`, `plant_original` and `fault_type`.
2. Within that same `fault_type` field, the label is frequently generic/inconsistent with the parameter actually edited: `efficiency_gt_1` is used for faults on AC.eta_s, ENG.eta_el, CH.eps and OT.eta_s (only the GT.eta_s rows are a literal match). This is a secondary consistency issue on top of defect class 1.

No placeholder tool outputs, no first-call answer leakage, no malformed/missing final XML, no empty completions, no invalid tier/variant values, and no trajectory/completion mismatch were observed in the 30 reviewed rows.
