# Open parameters — ANSWERED 2026-11-19 (see docs/DECISIONS_2026-11-19.md; that file wins on any difference)

Rules for this file:

1. Every value below is an **assistant placeholder**, including anything in `thermo_agent/train/config.py`.
   None of them is authorised by the user.
2. The reviewing model (Claude) fills the `ANSWER:` line for each item and nothing else. Do not infer an
   answer from a code default, from a paper, or from what looks reasonable.
3. Until an item is answered, no run may use it. Items marked **GATE** block the whole training campaign.
4. The assistant does not pick, recommend, or pre-fill. Where a tradeoff exists it is stated as a fact with
   a measured or documented cost, not as advice.
5. After answering, Claude updates `docs/IMPLEMENTATION_PLAN.md` so the plan references these IDs instead of
   numbers.

Quota snapshot used for the cost lines (2026-11-19): monthly `$7.27 / $100` used, daily `$1.23 / $10` used,
GPU `26.26 h` remaining until the 2026-11-21 refill. Kaggle sessions are capped at 12 h; the corpus is
~30 h/epoch on 2×T4 (measured estimate from a partial epoch, not a completed run).

---

## A. SFT run shape

| ID | Parameter | Location | Placeholder | What it controls | Options / measured facts | ANSWER |
|---|---|---|---|---|---|---|
| A1 **GATE** | `epochs` | `config.py:23` | `1` | passes over the SFT corpus | 1 ≈ 30 h ≈ 3 sessions; 2 does not fit before refill; step-based (`--max-steps`) gives a fixed cost | **3 (upper bound; early stopping A10)** |
| A2 | `max_steps` (alternative to A1) | `sft.py --max-steps` | `-1` (off) | hard stop independent of epochs | used only if A1 is answered as "step-based" | **off (-1)** |
| A3 | `lr` | `config.py:22` | `2e-4` | LoRA learning rate | no sweep has been run on this project; 2e-4 is the loftq/LoRA SFT convention | **1e-4** |
| A4 | `per_device_batch` | `config.py:25` | `2` | sequences per GPU per step | combined with A5 and 2 GPUs → 32 sequences/step today | **2** |
| A5 | `grad_accum` | `config.py:26` | `8` | accumulation | changes effective batch and step time linearly | **8 (32 seq/step on 2 GPUs)** |
| A6 | `max_grad_norm` | `config.py:27` | `1.0` | clipping before the fp16 optimizer step | fp16 needs clipping; 1.0 is the convention | **1.0** |
| A7 | `save_steps` / `eval_steps` | `config.py:30-31` | `200 / 200` | resume granularity inside a 12 h session | smaller = more HF uploads and checkpoint churn | **50 / 50** |
| A8 **GATE** | corpus for SFT | `SFT_DATA` env | `sft_mix_h` (intended) | which rows the model sees | candidates: `sft_tools_h` (960), `sft_mix_h` (960 + accepted frontier), `sft_all` (133,032 zero-shot), or a combination | **sft_mix_h (E1/E2)** |
| A9 | task subset | `sft.py --tasks` | all four | whether one model covers 4 tasks or 4 models | one model = 4× fewer runs, shared capacity | **all four tasks, one model** |

## B. LoRA and precision

| ID | Parameter | Location | Placeholder | Facts | ANSWER |
|---|---|---|---|---|---|
| B1 | base model | `config.py:9` | `Qwen/Qwen3.5-4B` | 4B fits 2×T4 in fp16 with LoRA; changing it changes the whole campaign | **Qwen/Qwen3.5-4B** |
| B2 | `r` (SFT) | `config.py:14` | `16` | rank; 16 is the common instruction-tuning default | **32** |
| B3 | `alpha` (SFT) | `config.py:15` | `32` | 2×r today | **64** |
| B4 | `dropout` | `config.py:16` | `0.05` | | **0.05** |
| B5 | `target_modules` | `config.py:17` | `all-linear` | all linear layers vs attention-only | **all-linear** |
| B6 | dtype | `sft.py --dtype` | `float16` | T4 has no bf16 kernels; **QLoRA/4-bit is currently not authorised by the user** | **float16, no QLoRA** |
| B7 | `r` / `alpha` (GRPO) | `config.py:50` | `8 / 16` | separate adapter rank for RL | **r 8 / alpha 16** |

## C. GRPO

| ID | Parameter | Location | Placeholder | Facts | ANSWER |
|---|---|---|---|---|---|
| C1 **GATE** | `k` values to run | `grpo.py` / stage names | `4, 8, 16` | each k = a separate multi-hour run and a separate checkpoint | **4, 8, 16** |
| C2 | `lr` | `config.py:37` | `5e-5` | | **2e-5, constant, 10 warmup** |
| C3 | `max_steps` | `config.py:43` | `300` | RL length dominates cost | **S = largest <=100 that lets k=16 finish in one 12 h session; same S for all k** |
| C4 | `num_generations` (G) | `config.py:38` | `4` | G=8 was previously rejected on T4 memory | **G = k; per_device 2; prompts/step 4 => grad_accum = k** |
| C5 | `max_completion_length` | `config.py:39` | `320` | must leave room for tool turns; 320 tokens is short for a multi-turn trace | **4096 whole multi-turn completion; 6 tool iterations** |
| C6 | `temperature` | `config.py:40` | `1.0` | | **T 1.0, top_p 1.0, top_k 0, min_p 0, rep 1.0** |
| C7 | `beta` (KL) | `config.py:47` | `0.0` | 0 = no reference model in memory | **0.0** |
| C8 | `loss_type` | `config.py:46` | `dapo` | `grpo` = original length normalisation | **dapo, group scaling** |
| C9 | which tasks get GRPO | `TASK` env | one | GRPO per task multiplies runs by 4 | **one joint run, all four tasks** |
| C10 | base for GRPO | `SFT_MERGED` | merged SFT adapter | must be the merged model, never raw Qwen | **merged best SFT checkpoint** |

## D. Trace campaign

| ID | Parameter | Location | Placeholder | Facts | ANSWER |
|---|---|---|---|---|---|
| D1 **GATE** | accepted-trace target | — | **none (cap removed)** | assistant had written 480 then 2,880; both unauthorised and deleted. Generation continues until the user stops it | **no cap** |
| D2 | per-model counts | — | **none (cap removed)** | same: no allocation invented | **no cap** |
| D3 | rows per hosted run | `kaggle/teacher_harness_task_s00.py` `SPEC["limit"]` | `0` = every row in the shard | assistant had set 40 (pilot) and then 120; both removed | **0 (all rows)** |
| D4 | max tool rounds | `SPEC["rounds"]` / `DEFAULT_MAX_ROUNDS` | `4` | more rounds = longer traces, more tokens, more cost | **6** |
| D5 | memory notes served per row | `SPEC["k_memory"]` | `3` | served block measured at p50 80 tokens | **3** |
| D6 | per-model cost cap | `SPEC["max_cost_usd"]` | `0` = no internal cap | account quotas are the only ceiling | **no cap** |
| D7 | premium-tier spend ceiling | — | **none** | user authorised these models; no ceiling set by the assistant | **no cap** |
| D8 | acceptance rule | `mix_tool_corpora.py` | verifier reward > 0 **and** ≥1 `<tool_call>` + `<tool_result>` | | **verifier accepted AND >=1 tool_call+tool_result AND final parses AND tools exist AND <=6 rounds** |
| D9 | trace cap enforcement | `mix_tool_corpora.py --frontier-per-task` | `0` (unlimited) | assistant-invented caps removed | **0 (unlimited)** |

## E. Corpus composition

| ID | Parameter | Location | Placeholder | Facts | ANSWER |
|---|---|---|---|---|---|
| E1 | deterministic : frontier ratio | mix script cap | 960 : accepted frontier | today the frontier side is 0 | **frontier tool traces first, deterministic fills to R_tool=4,800** |
| E2 | does the 133k zero-shot corpus join SFT | `SFT_DATA` | not joined | joining 20-variant zero-shot with tool traces mixes two prompt shapes | **yes: plain single-turn slice = 25% of mix, 1 paraphrase/plant** |
| E3 | over-8k handling | build scripts | drop, never truncate | 8,192 gate, 8,000 prompt/completion, 96 reserve | **drop, never truncate** |
| E4 | harness variant as the default training prompt | `*_h` dirs | built, not chosen | plain corpora remain untouched | **compact harness variant for tool rows** |

## F. Evaluation

| ID | Parameter | Location | Placeholder | Facts | ANSWER |
|---|---|---|---|---|---|
| F1 | scoring set | `data/heldout` | fixed | 214 test ids; only legal scoring set | **test = heldout_v2 (new seed, >=85/cell); current 214 ids become dev** |
| F2 | `LIMIT` | `kaggle/eval_runner.py` | `0` | 0 = all rows | **0** |
| F3 | `enable_thinking` at eval | `model_eval.py` | `True` (fixed today) | must match training render | **True** |
| F4 | tool access per arm | `eval_runner.py` SPECS | base none / sft none / harness tools / GRPO tools | guarded by tests | **Z and H for every model; table in DECISIONS_2026-11-19.md** |
| F5 | extra eval arms | — | none | e.g. SFT trained on mixed corpus vs tools-only | **SFT tools-only vs mix, after main table** |

## G. Run mechanics

| ID | Parameter | Location | Placeholder | Facts | ANSWER |
|---|---|---|---|---|---|
| G1 **GATE** | smoke gate | `kaggle/train_kernel_smoke` | v28 = `ERROR`, untriaged | no training run until a smoke kernel is green | **smoke defined in DECISIONS section 1** |
| G2 | GPU count assertion | `train_runner.py` | `== 2` | dual T4 required | **== 2** |
| G3 | resume strategy | `hf_sync.py` | private repo `thermo-agent-checkpoints` | must equal `<token owner>/thermo-agent-checkpoints` | **keep** |
| G4 | session length | Kaggle | 12 h | save cadence must beat it | **12 h** |

## H. Sync / release

| ID | Parameter | Location | Placeholder | Facts | ANSWER |
|---|---|---|---|---|---|
| H1 | HF corpus repo | `scripts/sync_corpora_hf.py` | `Reza-Nadimi/thermo-agent-corpora` (private) | never executed | **keep, after corpus rebuild passes audit** |
| H2 | which dirs sync | same | `sft_tools`, `grpo_tools`, `*_h`, `frontier`, `memory` | allow-list enforced | **keep allow-list + heldout_v2 prompts only** |
| H3 | Kaggle dataset version | `scripts/upload_corpus_dataset.ps1` | not executed | adds `_h` zips | **after rebuild** |
| H4 | paper/provenance regeneration | `docs/`, `paper/` | pending | only after held-out scores exist | **after first full score** |

---

## I. Assistant-invented caps still live in code (NOT changed, awaiting answers)

These are limits the assistant introduced without instruction. They were left untouched because changing them
changes training/eval behaviour, which the user forbade. They are listed here so none of them can be mistaken
for an agreed value.

| ID | Cap | Location | Placeholder | Consequence if left as-is | ANSWER |
|---|---|---|---|---|---|
| I1 | `max_steps = 300` | `config.py:43` | 300 | GRPO stops at 300 steps regardless of convergence | **see C3** |
| I2 | `max_completion_length = 320` | `config.py:39` | 320 | caps one GRPO rollout; a multi-turn trace is longer than this | **4096** |
| I3 | `num_generations = 4` | `config.py:38` | 4 | GRPO group size; G=8 was rejected on T4 memory earlier | **G = k** |
| I4 | `save_steps` / `eval_steps` | `config.py:30-31` | 200 / 200 | checkpoint granularity | **50 / 50** |
| I5 | `epochs = 1` | `config.py:23` | 1 | see A1 | **3 + early stopping** |
| I6 | eval `max_tool_calls = 6` | `model_eval.py:41` | 6 | a trace can be cut after 6 calls | **8** |
| I7 | eval `max_new_tokens` 2048 (tools) / 400 (text) | `model_eval.py:42-43` | 2048 / 400 | truncation changes the score | **4096 / 4096** |
| I8 | hosted `k_memory = 3` | `SPEC["k_memory"]` | 3 | memory notes served per row | **3** |

Removed on 2026-11-19 because they were pure assistant caps with no safety justification: hosted
`SPEC["limit"] = 40`, `max_cost_usd = 2.0`, `deadline_seconds = 7200` (now `0` = no stop for each), and the
480 / 2,880 accepted-trace targets.

## J. Decisions created by the assistant's own failures (reviewer decides)

| ID | Question | Why it exists | ANSWER |
|---|---|---|---|
| J1 | Does the 133k zero-shot corpus join SFT, or train on tool traces only? | the assistant claimed both were in use; only the tool corpus is wired | **yes, 25% plain slice** |
| J2 | 157 over-gate `_h` rows: raise `max_length`, shorten the harness block, or drop (audit loses 143/240)? | build gated tokens on the plain prompt, not the harness prompt | **shorten harness block to <=2,500 tokens; never raise max_length** |
| J3 | Is a 17.2k-char (~4.3k-token) harness prompt acceptable when the plain prompt is 2.4k? | same root cause as J2 | **no; target <=2,500 tokens** |
| J4 | Which hosted model is attempted first, given luna has never completed a run? | 4 days, 0 tool traces | **gemini-3.5-flash-lite, then haiku-5-5, luna after a 5-row pilot** |
| J5 | Is further Kaggle spend authorised ($92.73/$100 monthly, $8.77 daily remaining)? | every prior cap was invented by the assistant | **authorised within account quotas, effective only when Reza lifts the HARD RULE** |
| J6 | What defines a row's `tier`? C9 samples uniformly over task x tier and D1 reports per task x tier x model, but no corpus or generator emits a tier label. Measured on `data/processed/*_train.jsonl` (only tasks that carry a plant): repair components 4/5/6/7/9 (n=5,458; p33=6, p66=7), audit 4-13 (n=2,192; p33=6, p66=7); ledger and design have no plant column, so a component-count tier cannot be computed for them. `thermo_agent/synth/big_topologies.py` exists (expert-scale cycles) but no row in the training corpora uses it. Until defined, `sample_joint` puts every row in one `<missing>` bucket, i.e. uniform over task only. | assistant found it while verifying C9 on the real corpus, 2026-11-19 | **Reza 2026-11-19: tiers equal thirds for every task; a real `tier` column in training and test data, derived from a documented per-task size measure (`thermo_agent/synth/tiers.py`); remainder to the higher tier and recorded** |

---

## What stays frozen regardless of answers

- No model architecture change: full fp16 LoRA only. No QLoRA, no bf16 on T4, no FSDP/ZeRO.
- Held-out `data/heldout` is the only scoring set.
- Additive-only: no corpus, model, or teacher journal is deleted or overwritten.
- No token, `.env` value, or HF token in a notebook, dataset, or log.
- Kaggle calls only through `HTTP_PROXY/HTTPS_PROXY=socks5h://localhost:10808`.
- No premium-tier model run before its 40-row pilot cost is recorded.
