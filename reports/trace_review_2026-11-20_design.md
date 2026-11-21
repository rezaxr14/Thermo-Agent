# Trace review — design_train.jsonl (deterministic rebuild)

File reviewed: `data/sft_tools_v3/design_train.jsonl`
Rows reviewed: 30, at 1-based lines 1, 41, 81, ..., 1161 (indices 0, 40, 80, ..., 1160).
Each row was read in full (all fields: id, prompt, completion/trajectory, provenance, tier, tokens, site, ref_*). No grep/bulk scan was used to form verdicts.

Verdicts (format: `<line_no> | <id> | <tier> | <variant> | OK|CONCERN | <reason>`):

1 | des-0-00000-dt-direct-c3c2 | medium | direct | CONCERN | Scoring-only ref_fuel_w/ref_spec fields appear at row top level (check 4); otherwise tool results are real solver output and the final design XML is correct and consistent.
41 | des-0-00304-dt-direct-c3c2 | hard | direct | CONCERN | Scoring-only ref_fuel_w/ref_spec fields appear at row top level (check 4); reasoning cites the prior check's min-approach/peak-T numbers and the final XML matches the trajectory answer.
81 | des-0-00576-dt-direct-c3c2 | medium | direct | CONCERN | Scoring-only ref_fuel_w/ref_spec fields appear at row top level (check 4); tool payloads are genuine ok/numbers/units output and the closing <rationale>/<design> is intact.
121 | des-0-00832-dt-direct-c3c2 | easy | direct | CONCERN | Scoring-only ref_fuel_w/ref_spec fields appear at row top level (check 4); the only other observations are clean reasoning and a valid final design block.
161 | des-0-01085-dt-direct-c3c2 | easy | direct | CONCERN | Scoring-only ref_fuel_w/ref_spec fields appear at row top level (check 4); tier/variant valid, stream output real, final XML correct.
201 | des-0-01341-dt-direct-c3c2 | hard | direct | CONCERN | Scoring-only ref_fuel_w/ref_spec fields appear at row top level (check 4); turbine design XML and stream solve are consistent with the trace.
241 | des-0-01607-dt-direct-c3c2 | easy | direct | CONCERN | Scoring-only ref_fuel_w/ref_spec fields appear at row top level (check 4); boilerplate reasoning cites prior-check numbers and the final design is correct.
281 | des-0-01926-dt-direct-c3c2 | medium | direct | CONCERN | Scoring-only ref_fuel_w/ref_spec fields appear at row top level (check 4); no prompt/first-call leak and final XML matches the solved design.
321 | des-0-02199-dt-direct-c3c2 | hard | direct | CONCERN | Scoring-only ref_fuel_w/ref_spec fields appear at row top level (check 4); reasoning and final heat-pump design XML are otherwise sound.
361 | des-0-02406-dt-direct-c3c2 | medium | direct | CONCERN | Scoring-only ref_fuel_w/ref_spec fields appear at row top level (check 4); water-limit check output is real and the closing design XML is correct.
401 | des-0-02656-dt-direct-c3c2 | hard | direct | CONCERN | Scoring-only ref_fuel_w/ref_spec fields appear at row top level (check 4); PR/T4/toluene design XML matches the trajectory's final answer.
441 | des-0-02899-dt-direct-c3c2 | easy | direct | CONCERN | Scoring-only ref_fuel_w/ref_spec fields appear at row top level (check 4); tool results are real and final XML is consistent.
481 | des-0-03179-dt-direct-c3c2 | medium | direct | CONCERN | Scoring-only ref_fuel_w/ref_spec fields appear at row top level (check 4); reasoning cites prior-check numbers and final cyclopentane design is correct.
521 | des-0-03432-dt-direct-c3c2 | medium | direct | CONCERN | Scoring-only ref_fuel_w/ref_spec fields appear at row top level (check 4); final design XML and stream solve agree.
561 | des-0-03751-dt-direct-c3c2 | easy | direct | CONCERN | Scoring-only ref_fuel_w/ref_spec fields appear at row top level (check 4); heat-pump/no-bottoming design XML is otherwise clean.
601 | des-0-00000-dt-recover-c3c2 | medium | recover | CONCERN | Scoring-only ref_fuel_w/ref_spec leakage plus the text before tool call #2 cites only site estimates, not numbers from the preceding "no <design> block" error result (check 1).
641 | des-0-00304-dt-recover-c3c2 | hard | recover | CONCERN | Scoring-only ref_fuel_w/ref_spec leakage plus the pre-second-call reasoning does not cite numbers/identifiers from the preceding probe error result (check 1).
681 | des-0-00576-dt-recover-c3c2 | medium | recover | CONCERN | Scoring-only ref_fuel_w/ref_spec leakage plus the transition text before the real validity_check cites no values from the preceding error tool result (check 1).
721 | des-0-00832-dt-recover-c3c2 | easy | recover | CONCERN | Scoring-only ref_fuel_w/ref_spec leakage plus the second-turn reasoning omits citation of numbers from the prior error result (check 1).
761 | des-0-01085-dt-recover-c3c2 | easy | recover | CONCERN | Scoring-only ref_fuel_w/ref_spec leakage plus pre-second-call text cites only hypothesis MW/K, not the preceding tool result (check 1).
801 | des-0-01341-dt-recover-c3c2 | hard | recover | CONCERN | Scoring-only ref_fuel_w/ref_spec leakage plus the reasoning before tool call #2 fails to cite numbers/identifiers from the prior error result (check 1).
841 | des-0-01607-dt-recover-c3c2 | easy | recover | CONCERN | Scoring-only ref_fuel_w/ref_spec leakage plus the second-turn reasoning does not reference the preceding probe error's content (check 1).
881 | des-0-01926-dt-recover-c3c2 | medium | recover | CONCERN | Scoring-only ref_fuel_w/ref_spec leakage plus the pre-second-call text lacks citation of the previous tool result's numbers (check 1).
921 | des-0-02199-dt-recover-c3c2 | hard | recover | CONCERN | Scoring-only ref_fuel_w/ref_spec leakage plus the transition before the main check cites no numbers from the error result (check 1).
961 | des-0-02406-dt-recover-c3c2 | medium | recover | CONCERN | Scoring-only ref_fuel_w/ref_spec leakage plus the second-turn reasoning omits citation of the prior tool result's identifiers (check 1).
1001 | des-0-02656-dt-recover-c3c2 | hard | recover | CONCERN | Scoring-only ref_fuel_w/ref_spec leakage plus the reasoning before tool call #2 cites site estimates rather than the preceding error result (check 1).
1041 | des-0-02899-dt-recover-c3c2 | easy | recover | CONCERN | Scoring-only ref_fuel_w/ref_spec leakage plus the pre-second-call text does not cite numbers from the preceding error tool result (check 1).
1081 | des-0-03179-dt-recover-c3c2 | medium | recover | CONCERN | Scoring-only ref_fuel_w/ref_spec leakage plus the second-turn reasoning fails to reference numbers/identifiers from the prior result (check 1).
1121 | des-0-03432-dt-recover-c3c2 | medium | recover | CONCERN | Scoring-only ref_fuel_w/ref_spec leakage plus the transition text omits citation of the preceding probe error's output (check 1).
1161 | des-0-03751-dt-recover-c3c2 | easy | recover | CONCERN | Scoring-only ref_fuel_w/ref_spec leakage plus the reasoning before the real check cites only site estimates, not the prior error result (check 1).

## Summary

- Rows reviewed: 30 (lines 1, 41, 81, ..., 1161).
- OK: 0
- CONCERN: 30
- Concrete defect classes:
  1. **Scoring-only reference fields present in the row (all 30 rows):** every reviewed row carries top-level `ref_fuel_w` and `ref_spec` fields, violating check 4 ("no scoring-only ref_* field appears in the row"). No `ref_evals` was observed, and no ref_* name is leaked into any prompt or tool-call candidate, but the fields themselves are present in all 30 rows.
  2. **Recover-variant reasoning gap (all 15 recover rows, lines 601–1161):** the assistant text before tool call #2 (after the empty-probe error result) gives a site-analysis hypothesis and the phrase "The probe taught me what the checker demands...", but cites no numbers or identifiers from the preceding `DesignError: no <design> block` tool result, which is the specific requirement of check 1. (The transition before tool call #3 does correctly cite min-approach/peak-T numbers.)
- Otherwise clean across all 30 rows: tier ∈ {easy, medium, hard}; provenance.variant ∈ {direct, recover}; tool results are genuine solver output (check arrays with ok/numbers/units and solved stream tables, plus a real error payload for the recover probe); no placeholder payloads; final completions end with a correct `<rationale>` followed by a `<design>` block of `key = value` lines and match the trajectory's final answer; completions are non-empty and consistent with their trajectories.
