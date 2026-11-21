# Archived 2026-11-20 — pre-CoT traces (not training data)

Moved out of the live paths by the 2026-11-20 review (`docs/AUDITS.md` F1, F2; `docs/DECISIONS.md` 2026-11-20). Nothing was edited.

| dir | what it is | why it is archived |
|---|---|---|
| `frontier_tools_cohort`, `frontier_tools_accepted`, `frontier_tools_clean*`, `frontier_tools`, `frontier_tools_pilot` | 237 accepted AI tool traces (luna/haiku) | every non-final turn has no reasoning text: tool calls only, so they teach no chain of thought |
| `sft_tools*`, `grpo_tools*` | deterministic tool traces v1/v2 | reasoning is boilerplate (12.7% distinct `<think>`; 8 sentences = 45% of blocks) and the first tool call already carries the ground-truth answer |
| `sft_mix_v2`, `grpo_mix_v2` | mixes built from the above | inherit both defects |
| `frontier_eval_export` | replay prompts with pre-heldout_v2 ids | ids never matched the held-out set |

Live pipelines write fresh data to the canonical names (`data/frontier_tools_cohort`, `data/sft_tools_v3`, ...). Keep this folder for forensics; never mix it into a corpus.
