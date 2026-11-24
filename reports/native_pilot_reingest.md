# Native pilot re-ingest (2026-11-24)

Both journals were re-processed from scratch after the legacy cohort was archived.
Gate counts are the exact native-template measurement of the cleaned trace.

## gpt-6-luna

- journal rows: 40; cost: $0.07797
- gate: 18 survived of 21 accepted tool rows (3 over gate); raw max 9347 tokens, surviving max 7635

| task | journal rows | tool rows | tool turns | reasoning share | cohort accepted |
|---|---|---|---|---|---|
| repair | 10 | 10 | 48 | 100.0% | 8 |
| design | 10 | 10 | 59 | 100.0% | 2 |
| audit | 10 | 10 | 49 | 100.0% | 0 |
| ledger | 10 | 10 | 41 | 100.0% | 7 |

## claude-haiku-5-5

- journal rows: 40; cost: $0.14529
- gate: 15 survived of 15 accepted tool rows (0 over gate); raw max 7224 tokens, surviving max 7224

| task | journal rows | tool rows | tool turns | reasoning share | cohort accepted |
|---|---|---|---|---|---|
| repair | 10 | 10 | 33 | 100.0% | 9 |
| design | 10 | 10 | 41 | 100.0% | 4 |
| audit | 10 | 10 | 36 | 100.0% | 0 |
| ledger | 10 | 0 | 0 | 0.0% | 0 |

## Mix selection

- accepted frontier rows: 30 (repair 17, design 6, ledger 7, audit 0)
- selected into both mixes: 17 (repair 9, design 6, ledger 2, audit 0)
- skipped over the exact 8,192-token gate: 13 (repair 8, ledger 5)

## Row loss found

- One gated row per model (`arch-107-00019#v16`, repair) has no export record in the
  ingest export dirs; it resolves only through `data/frontier_export`. Both were skipped
  as `skipped_unknown_or_empty`. This is the row-loss bug to fix in step 3.

## Audit yield

- 0 of 20 audit rows reached the cohort: every audit trace is rejected for format
  (9 per model) plus change/prediction. Outside step 2; step 4 owns this.
