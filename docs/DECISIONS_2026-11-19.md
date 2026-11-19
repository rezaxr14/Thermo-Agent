# Decisions — training, decoding, corpus, evaluation (answers to `OPEN_PARAMETERS_FOR_REVIEW.md`)

Decided by Claude on 2026-11-19 at Reza's instruction ("make all of the training decisions and hyper-parameters").
These values are now authoritative. The local agent implements them and does not change them; a value changes only
by a new entry here that supersedes it, with the measurement that justified the change. Estimates are marked `est.`;
every est. has a **rule** that replaces it by a measured number.

Hard facts used: 2x T4 fp16 LoRA (no QLoRA/bf16/FSDP), 12 h sessions, 26.26 GPU-h left until the 2026-11-21 refill
and about 30 h per week after, 30 h/epoch measured for the 133k-row zero-shot corpus (about 1,850 tokens/s est.),
harness prompt 4.3k tokens vs plain 0.7k, 155/960 harness rows over the 8,192 gate, smoke v28 red, zero frontier tool traces.

## 1. Gates first (nothing below runs before these)

1. **G1 smoke.** Triage the v28 log, fix, re-run until green. Green means: 20 SFT steps on the compact-harness tool corpus at
   `max_length 8192`, `per_device_batch 2`; loss falls; the step-20 eval forward passes; one checkpoint saves and resumes;
   plus a 3-step multi-turn GRPO smoke (k=4) in which every rollout finishes and a reward is computed. Seconds per row is recorded.
2. **J2/J3 compact harness (before any trace is generated).** Shorten the harness block to **<= 2,500 tokens** measured with the
   real Qwen3.5 tokenizer (keep: tool contract, output tags, failure checklist, the one selected skill; drop prose examples and
   non-selected skills). One harness text for every arm (tracegen, training, eval, frontier H). Rebuild `*_h` corpora and the
   hosted prompt shards from it. `build_trace` must re-measure tokens on the rendered harness row and gate on that value.
   Gate stays **8,192**; over-gate rows are dropped and counted, never truncated. If audit still loses > 10% of rows, shorten the
   audit tool outputs (cap `solve_plant` rows), do not raise `max_length`.

## 2. Answers

### A. SFT
| ID | Value | Reason |
|---|---|---|
| A1 epochs | **3** (upper bound; A10 stops earlier) | small curated corpus; 1 epoch is too little for tool-format learning, 3 is the usual LoRA SFT range |
| A2 max_steps | off (-1) | length comes from epochs + early stopping |
| A3 lr | **1e-4** | LoRA all-linear, about 10x full-FT lr (Schulman et al. 2025); 2e-4 is too hot for 3 epochs on 6k rows in fp16 |
| A4/A5 batch | **2 x 8 x 2 GPUs = 32 sequences/step** | unchanged; fallback if OOM at 8k: 1 x 16 x 2 (same effective batch) |
| A6 grad clip | **1.0** | fp16 needs it |
| A7 save/eval | **50 / 50 steps** (about 1 h each, est.) | resume granularity well inside a 12 h session; must be equal for best-model loading |
| A8 corpus | **`sft_mix_h`** = A-slice + tool rows, see E1/E2 | |
| A9 tasks | **all four, one model** | one model is the paper's claim; 4 models = 4x runs |
| A10 schedule | cosine, warmup_ratio 0.05, weight_decay 0.0, AdamW (torch), seed 0, **early stopping patience 3 evals on val loss**, load best model at end | "patience" decision; patience 3 x 50 steps = 0.75 epoch |

SFT size (est.): 4,800 tool rows + 1,600 plain rows = 6,400 rows = 200 steps/epoch = **600 steps max**, about 4 h/epoch,
about 12 h total, i.e. 1-2 sessions with resume. **Rule:** after G1, `R_tool = floor(5.5 h / measured_seconds_per_tool_row)`,
clamped to [2,000, 8,000]. Reduce unique rows, never epochs.

### B. LoRA / precision
| ID | Value |
|---|---|
| B1 | `Qwen/Qwen3.5-4B` |
| B2/B3 | **r 32, alpha 64** (4 tasks + tool protocol; rank 16 risks under-capacity, 64 buys nothing at this data size) |
| B4 | dropout **0.05** |
| B5 | **all-linear** |
| B6 | **float16**, full-precision base weights, no QLoRA (not authorised) |
| B7 | GRPO adapter **r 8, alpha 16** on top of the merged SFT model (RL carries few bits per episode) |

### C. GRPO (multi-turn, tools, TRL `environment_factory`)
| ID | Value | Reason |
|---|---|---|
| C1 k | **4, 8, 16**, all three, same prompts, same step count | the sweep is a paper claim |
| C2 lr | **2e-5**, constant with 10 warmup steps | LoRA RL, small batches, noisy multi-turn reward |
| C3 steps | **S = the largest value <= 100 for which the k=16 run fits one 12 h session**; all three k use the same S | equal-steps comparison; S measured in the G1 GRPO smoke (time per step x S <= 11 h) |
| C4 G | **G = k**, `per_device_batch 2` | |
| C4a | **prompts per optimizer step = 4 for every k**, so `grad_accum = 4*k/(2*world) = k` on 2 GPUs (k=4: 16, k=8: 32, k=16: 64 sequences) | old setup gave 4/2/1 prompts for k=4/8/16, which confounded group size with diversity; now derived in `grpo.py` |
| C5 max_completion_length | **4,096** (whole multi-turn completion); `max_tool_calling_iterations` **6**; `mask_truncated` on, truncation rate logged | 320 cannot hold a tool trace. If dev truncation > 10%, raise to 6,144 |
| C6 rollout sampling | **temperature 1.0, top_p 1.0, top_k 0 (off), min_p 0, repetition_penalty 1.0** | RL must sample the distribution the loss differentiates; truncation/penalties bias importance ratios |
| C7 beta | **0.0** | no reference model in memory |
| C8 loss | **dapo**, `scale_rewards="group"` | unchanged |
| C9 tasks | **one joint run over all four tasks**, prompts sampled uniformly over task x tier; reward dispatched by the row's task | per-task runs would be 12 runs |
| C10 base | merged **best** SFT checkpoint (`runs/sft_merged`), never raw Qwen | |
| seeds | seed 0 for every arm; seeds 1,2 only for the k pair with the largest gap, after first results | quota |
| stopping | log zero-variance-group share; if > 50% for 10 consecutive logs, stop and report | |

### D. Trace campaign
| ID | Value |
|---|---|
| D1/D2/D6/D7/D9 | **no cap** on traces, per-model counts, cost or rows; report accepted per task x tier x model after every run |
| D3 | `limit 0` (every row of the shard) |
| D4 | **6** tool rounds (assistant turns) for every H arm, tracegen and eval (`DEFAULT_MAX_ROUNDS = 6`) |
| D5 | `k_memory` **3** |
| D8 acceptance | verifier **accepted** (not just reward > 0) AND >= 1 `<tool_call>` + `<tool_result>` AND final answer parses AND every called tool exists AND <= 6 rounds |
| order | Phase 0 pilot (40 rows) per model, record cost, then bulk. **J4:** `gemini-3.5-flash-lite` first (only payload that reached a model), `claude-haiku-5-5` second, `gpt-6-luna` after a 5-row pilot passes with its tools-no-reasoning setting (a documented deviation from "reasoning on": the API rejects tools + reasoning for luna), premium tier after bulk shows non-zero tool yield |

### E. Corpus composition
| ID | Value |
|---|---|
| E1 | tool rows: **all accepted frontier tool traces first**, deterministic traces fill to `R_tool` (default 4,800 = 600 plants x 2 variants (direct, recover) x 4 tasks), balanced over task, tiers equal thirds where a hard tier exists (repair, audit), else 50/50 easy/medium |
| E2 / J1 | **yes**: plain-prompt single-turn rows from stream A = **25% of the final mix** (R_tool/3), **one paraphrase per ground-truth plant**, task-balanced. They keep Z-mode competence and the answer format; the student learns two modes keyed by the prompt |
| E3 | drop over-gate rows, never truncate |
| E4 | compact harness variant is the training prompt for every tool row; plain prompt only for the A-slice |

### F. Evaluation
| ID | Value |
|---|---|
| F1 | **test = `data/heldout_v2`** (new seed, generated on CPU now, >= 85 rows per reported cell where a generator can reach it, merged cells documented); the existing 214-id `data/heldout` becomes the **dev set** (tuning only). Freeze v2's manifest before any model is scored on it. Reason: current cells have 4-16 rows and cannot support the paper's pairwise claims |
| F2 | `LIMIT 0` |
| F3 | `enable_thinking=True` in training, GRPO and eval |
| F4 | arms and modes: base 4B / base 9B / ours-SFT / ours-GRPO k4,k8,k16 each in **Z** (plain prompt, no tools) and **H** (compact harness, tools, memory); frontier Z (hosted zero-shot task) and H (tool harness); other LLMs placeholder. The old guard "sft none" is replaced by this table |
| F5 | extra arm **SFT tools-only vs SFT mix** (tests the A-slice) only after the main table exists |

**Decoding at evaluation (all local arms identical):** primary **greedy** (`do_sample=False`, repetition_penalty 1.0, no presence penalty).
Secondary pass@8: **temperature 0.6, top_p 0.95, top_k 20, min_p 0**, seeds 0-7, no repetition/presence penalty. These are Qwen's
published thinking-mode precise-task settings; Qwen's general-chat setting adds presence_penalty 1.5, which is rejected here because
structured output legitimately repeats unit tokens, ids and tool arguments. Frontier hosted models use provider defaults (not settable with reasoning), recorded per run.
Per-turn `max_new_tokens` **4,096**; `max_tool_calls` **8**; `max_turns` **6**; `max_length` 8,192 unchanged.

### G / H / I
| ID | Value |
|---|---|
| G1 | smoke gate defined in section 1 |
| G2 | `NGPU == 2` stays |
| G3 / G4 | private HF repo and 12 h sessions stay; relay through `thermo-agent-resume` after every session |
| H1-H3 | sync after the corpus is rebuilt on the compact harness and passes the audit; allow-list unchanged plus `heldout_v2` prompts only (no answers) in a separate private dataset |
| H4 | paper/provenance regeneration after the first full score |
| I1-I8 | superseded by C3 (S), C5 (4,096), C4 (G=k), A7 (50/50), A1 (3 + early stop), I6 = 8, I7 = 4,096, I8 = 3 |

## 3. Implemented in this commit

`thermo_agent/train/config.py` (all values above), `sft.py` (scheduler, warmup, weight decay, best-model loading, early stopping),
`grpo.py` (sampling fields, scheduler, derived `grad_accum`, 4,096 / 6 defaults), `model_eval.py` (8 / 4,096 / 4,096),
`agent_graph.py` (`DEFAULT_MAX_ROUNDS = 6`), `tests/test_decisions_applied.py` (fails if any value drifts).

## 4. Left for the agent (each needs a test; see `IMPLEMENTATION_PLAN.md`)

Compact harness text; `build_trace` re-measure; deterministic trace expansion to 600 plants/task; A-slice builder; joint-task GRPO reward dispatch;
sampled pass@8 mode in `HFGenerator` (temperature/top_p/top_k/seed); `heldout_v2` generation; eval arm table in `eval_runner.py`.
