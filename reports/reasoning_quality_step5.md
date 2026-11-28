### Native-pilot journals (frontier models)

| Model | Task | Rows | Gate OK | Gate fail | short | repeated | ungrounded | cited / non-first |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| claude-haiku-5-5 | audit | 10 | 6 | 4 | 0 | 0 | 4 | 12/26 |
| claude-haiku-5-5 | design | 10 | 4 | 6 | 0 | 0 | 6 | 5/31 |
| claude-haiku-5-5 | ledger | 10 | 0 | 10 | 10 | 0 | 0 | 0/0 |
| claude-haiku-5-5 | repair | 10 | 3 | 7 | 0 | 0 | 7 | 4/23 |
| gpt-6-luna | audit | 10 | 3 | 7 | 0 | 0 | 7 | 10/39 |
| gpt-6-luna | design | 10 | 0 | 10 | 0 | 0 | 10 | 6/49 |
| gpt-6-luna | ledger | 10 | 4 | 6 | 0 | 0 | 6 | 3/31 |
| gpt-6-luna | repair | 10 | 3 | 7 | 0 | 0 | 7 | 13/38 |

### Deterministic tool corpus (`data/sft_tools_v3`)

| Model | Task | Rows | Gate OK | Gate fail | short | repeated | ungrounded | cited / non-first |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| deterministic | audit | 1200 | 1200 | 0 | 0 | 0 | 0 | 3000/3000 |
| deterministic | design | 1200 | 1200 | 0 | 0 | 0 | 0 | 1200/1800 |
| deterministic | ledger | 1200 | 1200 | 0 | 0 | 0 | 0 | 1724/1800 |
| deterministic | repair | 1200 | 1200 | 0 | 0 | 0 | 0 | 2234/2400 |
