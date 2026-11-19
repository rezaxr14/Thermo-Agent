# User directives — verbatim intent, recorded 2026-11-18

This file exists so you can verify the instructions were heard. Every item below came directly
from the user in conversation; nothing here is my invention. Implementation locations are listed
so each directive can be checked against code.

## D1 — Real thermodynamics tools, no fake wrappers
> "make proper thermodynamics tools i dont want bullshit crap" / "dont make bullshit stuff"

I deleted the first attempt (`physics_tools.py`, which only filtered verifier reason strings by
keyword) and replaced it with **`thermo_agent/teacher/toolkit.py`**: tools that build the plant
from the model's own candidate, run the real sequential-modular solver (`core.solve.solve`), and
return numbers with units. Verified on a real row: mass residual 0.0 kg/s, energy residual 0.0 W,
exergy destruction 74.7 MW, eta_II 0.297 — produced by the solver, not hardcoded.

## D2 — The full harness loop: propose, use tools, self-correct, answer
> "the kaggle ai can propse shit and call the tools and iteratively fix its shit and give a response
> using skills and tools and memory which it should have access to it"

Implemented as a **LangGraph state machine** in `thermo_agent/teacher/agent_graph.py`:
`build_context -> propose -> {call_tools -> reflect -> propose}* -> finalize -> END`.
The model can call tools across up to 4 rounds, see numeric feedback, and revise before finalizing.

## D3 — Skills + memory exposed, model decides when to use them
> "expose the fucking tools and skill to the models" / "memory which it should have access to it,
> the memory gets updated preiodicly so its always fresh"

- `build_context` injects **AGENTS.md + component/task SKILL.md** (same skills the student trains on).
- Fresh failure memory is loaded from `data/memory/*.jsonl` every generator row (and reloaded every
  25 rows in the CLI driver) via `memory.retrieve`, and injected as a "KNOWN FAILURE MODES" block.
- The system prompt explicitly tells the model it has tools, skills and memory and should use them
  to self-correct. Recorded in `docs/TRAINING_DATA_PROTOCOL.md`.

## D4 — Record frontier traces and train our model to behave like them
> "the forniter models sit in harnes and do everything and we record them and train our model to
> behave like them"

`teacher_row()` (`thermo_agent/teacher/validate.py`) now persists the full trajectory — reasoning,
tool calls, numeric tool results, revisions — as the SFT completion when a trace is accepted
(`source=frontier_tools`). `transfer.ingest` passes `trajectory` through. So the student learns the
tool-using behavior, not just the final answer.

## D5 — LangGraph to organize it
> "use langGraph to make and orginize stuff, so it is not a shitshow"

`thermo_agent/teacher/agent_graph.py` is a `StateGraph(AgentState)` with explicit nodes and a
conditional edge; `GRAPH.invoke(state)` is the single entry point. Installed langgraph 1.2.22.

## D6 — Stop AI calls, don't waste quota, restart only after fixed
> "stop the faulty ai calls now and dont waste my quota, after you fixxed these proeprly start it again"

I killed the local `run_big_teacher_v2.ps1` processes. The server-side Kaggle task v3 has no CLI
cancel and self-limits via its `$1.6` cost cap / 2h deadline. I did not launch any new AI call
while fixing the harness.

## D7 — Model choices and reminder
> "use haudi 5.5 instead for claude, sonnet is expensive, we will do it later, (remind me to do that
> later on) ... gemini 3.5 flash lite ... gpt-6-luna ... reasoning being on is ok and we want
> reasoning if we can capture it"

Recorded in `docs/TEACHER_MODEL_CHOICES.md` and `docs/STATUS.md`:
- now: `claude-haiku-5-5`, `gemini-3.5-flash-lite`, `gpt-6-luna`, reasoning ON (`reasoning="low"`).
- **REMINDER: revisit `claude-sonnet-5-5` later** — the user explicitly deferred it as too expensive.
2026-11-19: budget tier is luna then haiku; expensive tier after; gemini-3.5-flash-lite is excluded and must not consume quota.

## D8 — Zero-shot baseline -> tool CoT -> train -> eval on same held-out set -> compare
Recorded in full in `docs/TRAINING_DATA_PROTOCOL.md` (section "The decision, in order").

The held-out test set is `data/heldout`, built from a fresh ground-truth seed **987654** and excluded from every
training and prompt corpus. **All of the following arms are scored on that one set**, so every score is directly
comparable:

1. frontier models zero-shot (the *frontier baseline*);
2. frontier models in the LangGraph harness with real physics tools + AGENTS.md skills + fresh failure memory
   (the *tool-CoT arm* that generates the SFT data);
3. other thermodynamics LLMs (domain-model arm);
4. our SFT 4B;
5. our SFT + GRPO with **k=4**;
6. our SFT + GRPO with **k=8**;
7. our SFT + GRPO with **k=16**.

Each checkpoint is written to a **distinct path** so runs never overwrite one another: e.g. `runs/sft`,
`runs/grpo_k4`, `runs/grpo_k8`, `runs/grpo_k16` (and separate paths for the frontier/other-model arm reports).
The tool-CoT traces accepted by the verifiers become SFT completions, so the 4B learns tool use rather than only
the final answer.

## D9 — Update notebooks and docs to the new objective, no going back
> "update our notebooks to this new objective and also the docs no turning back"

In progress: `scripts/kaggle_harness_cli.py` now uses the LangGraph harness; this file and
`docs/TRAINING_DATA_PROTOCOL.md` carry the objective. Notebook updates are being applied now.

## D10 — Agent behavior: look for skills/tools first, then examine the cycle against memory
> "add some common reasons of failure like unit mismatch and some good practices like first go and
> look for skills and tools available then examine the cycle to the memory"

The agent's system prompt (in `agent_graph.build_context`) must instruct the model to:
1. FIRST inventory the skills and tools available to it,
2. THEN examine the plant/cycle against its memory,
3. Then propose, call tools, self-correct, and answer.
Add common failure causes to the prompt/memory: unit mismatch (MW vs W, kJ vs J, fraction vs percent,
K vs C, bar vs Pa), sign flips, swapped ports, wrong fluid class, efficiency > 1, out-of-range params.

## D11 — Token-efficient memory for the AI: what failed and what worked, not model/time logs
> "make the memory that we feed to the agent token efficient, the ai needs to know what failed and
> what worked not which model failed and which succeeded and times ... we dynamically strip the things
> that dont effect the ai reasoning and feed the model what actually it is interested in"

Implement a `compact_memory(messages, notes)` that produces a SHORT memory block containing only
the failure pattern + the correct fix pattern (what failed / what worked). Strip model ids,
timestamps, token counts, cost, latency, run ids and verbose logs — keep those in the full memory
files and detailed reports for OUR use, but feed the AI only the reasoning-relevant kernel. Wire
it into `agent_graph.build_context` so the AI sees the compact version while the full ledger is
still recorded for provenance.

---

How to verify any of this yourself: open the linked file, or run
`python -m pytest tests/test_agent_graph.py -q` (checks the graph and real numeric tools).

---

# Addendum 2026-11-19 (same convention: user intent, verbatim where possible)

## D12 - Two data sources across the four task families - answer with yes/no
> "our training data is consisted of two things right? deterministic 20 varient stuff and frontier ai in
> harnes traces in 4 cohortes? give me a yes or no , do we train on both of thesee right?"

Honest answer recorded in `docs/DIRECTIVE_COMPLIANCE_AUDIT_2026-11-19.md`: there are **three** streams
(20-variant zero-shot, deterministic multi-turn tools, frontier tool CoT), the four "cohorts" are the four
task families, and frontier tool CoT is **0 rows today**, so training on it has not happened.

## D13 - Traces first, training later (binding order)
> "we will not train on this craps we will make that fuking ai shit frst then we will start traninng"

Order is binding: no SFT/GRPO run starts until verifier-accepted frontier tool traces exist and are mixed in.
Recorded as the gating condition in the audit doc and in `docs/STATUS.md`.

## D14 - Many more AI traces, multi-model
> "in crease ai traces we will make a lot of them with luna and hiki and some of them with sol and oppos and
> sonnet and geimini 4 stuff"

Bulk from `gpt-6-luna` and `claude-haiku-5-5`; a premium subset from sol / opus / sonnet / gemini-4.
Exact count plan and caps: "Trace campaign v2" in `docs/DIRECTIVE_COMPLIANCE_AUDIT_2026-11-19.md`.
Premium slugs are read from the hosted run menu, never invented.

## D15 - Prompt separation is mandatory in tracegen, training and evals
> "we need different prompts for hte ai mdeos (zero shot + no harnes , baseline + harness, traned model +
> harnes and stuff like that) in training and in tracegen and in evals ... so our traned model + harness does
> not get feed zero shot prompt, and our zeroshot runs dont get feed the multi turn harness usage prompts"

Implemented as a matrix in the audit doc. Two real leaks were found and fixed on 2026-11-19:
GRPO eval arms scored without tools, and `HFGenerator` rendering `enable_thinking=False` while training
rendered `True`. One open item remains: replay arms re-execute stored `<tool_call>` blocks.

## D16 - Smoke-test before any big run
> "actually smoke test cople of compilations before commiting to a big run so we dont get hit by this kind of problems"

Smoke kernel v28 was pushed and came back **ERROR**; its log has not been triaged yet, so the gate is
**closed** and no training was started. This is recorded as an open blocker, not as a pass.

## D17 - Sync artifacts to Hugging Face and Kaggle datasets
> "also sync our stuff to the huggingface and kaggle datasets"

Scripts exist (`scripts/sync_corpora_hf.py`, `scripts/upload_corpus_dataset.ps1` with `_h` corpora added).
They have **not been executed** yet; no upload is claimed.

## D18 - Free subagents for bulk work
> "use subagents you shit"

Bulk work runs on `opencode/hy4-flash-free` subagents. No DeepSeek subagents are ever launched.
Three were used on 2026-11-19 (corpus token audit, mix harness tests, eval-arm separation tests).

## D19 - Document every instruction so an independent model can audit the work
> "doucument all of my instruction so a clude can check them and find your cheats and dihinisty and your trash builds"

`docs/DIRECTIVE_COMPLIANCE_AUDIT_2026-11-19.md` is the audit target: directive-by-directive status,
evidence paths, reproducing commands, and a self-reported deviation list.

## D20 - Epoch count is the user's decision, not the assistant's
> "and who told you to do 1 epoch idiot"

`thermo_agent/train/config.py` had `epochs = 1` as an assistant placeholder sized against quota. It was
never authorized. It is now treated as **undecided** until the user sets it; the audit doc records the
time cost of each option.
