# Kaggle: the frontier-teacher pass

The second half of the training corpus comes from a frontier model whose answers must pass the same tool suite as
everything else. Kaggle supplies the model access and quota (Kaggle Benchmarks: `kaggle_benchmarks` with Kaggle's
own model proxy); the physics checks run locally, so nothing the teacher writes enters the corpus unverified.

The earlier version of this path was a self-hosted vLLM server behind a cloudflared tunnel. That is gone. What
remains is a hand-off in three steps, and **step 2 needs only a browser** — the local Kaggle API is not required
(and is currently returning 403 for this account until the Kaggle phone/identity verification is done).

## 1. Export the prompts (local)

```powershell
python -m thermo_agent.teacher.transfer export data/sft data/frontier_export --per-task 400
Compress-Archive -Path data/frontier_export -DestinationPath data/frontier_export.zip -Force
```

`data/frontier_export/` holds `{task}_train.jsonl` with **one prompt per ground-truth plant** (paraphrase variants
are a student-side device), the harness and the retrieved failure notes already inside the prompt, and **no
ground-truth completion**, so the notebook cannot leak the answer to the teacher. `export_manifest.json` records the
counts, the harness mix and the maximum prompt length. The zip is what Kaggle's dataset uploader wants.

The current export is 1,600 prompts (400 per task) over 36 to 44 diagrams per task, 1.8 MB zipped; the longest
prompt is 3,998 tokens, so every prompt fits the model's 4,096-token training window with room for the answer. The
export is deterministic: re-running the command reproduces the same ids and prompts from the same corpus.

## 2. Run the teacher (Kaggle, browser only)

1. <https://www.kaggle.com/datasets> -> **New dataset** -> drag `data/frontier_export.zip` -> name it
   `thermo-teacher-prompts` -> **Create**. (Kaggle unpacks the zip, so the notebook sees the `*.jsonl` files.)
2. <https://www.kaggle.com/notebooks> -> **File -> Import notebook** -> `kaggle/teacher.ipynb` (built by
   `python kaggle/make_notebook.py` from `kaggle/teacher_benchmark.py`).
3. In the notebook's right panel: **Add Input -> Your datasets -> thermo-teacher-prompts**, and set
   **Accelerator: None** (the model runs on Kaggle's side, not on the notebook's GPU).
4. Set `MODEL` to a key printed by the second cell (for example `anthropic/claude-sonnet-5-5@default`) and
   **Run all**. The notebook finds the dataset by searching `/kaggle/input` for `export_manifest.json`.
5. Download `/kaggle/working/completions.jsonl` from the notebook's output panel.

The loop writes one JSON line per attempt, sends each prompt as its own system instruction plus its user message,
and skips ids it has already answered, so a session that dies keeps its progress. Failures are recorded with their
error instead of aborting the run.

The alternative is to paste `kaggle/teacher_benchmark.py` into a notebook created from
<https://www.kaggle.com/benchmarks/tasks/new>, which already has `kaggle_benchmarks` installed. The script and the
notebook are the same code: the notebook is generated from the script on `# %%` markers, and a test fails if the
checked-in notebook is stale.

## 3. Validate and merge (local)

```powershell
python -m thermo_agent.teacher.transfer ingest data/frontier_export completions.jsonl data/frontier `
    --memory-dir data/memory --model anthropic/claude-sonnet-5-5@default
```

Every answer is parsed and run through the task's evaluator (repair: apply the patch, re-solve, first- and
second-law verifiers; ledger: energy closure and non-negative exergy destruction; audit: ranking, change and
prediction checks with grounding; design: build and verify the plant). Accepted answers are appended to
`data/frontier/frontier_{task}_train.jsonl` with `source: frontier`, their reward and the verifier's reasons;
rejections become hindsight-memory notes in `data/memory/frontier_notes.jsonl`.

Ingest is resumable and never duplicates a row. `combine` already reads `data/frontier/`, so the next
`pwsh scripts/build_corpus.ps1` folds the accepted traces into `data/sft/` as extra `source: frontier` rows.

## Training notebook (dual T4, resumable)

`kaggle/train.ipynb` is generated from `kaggle/train_runner.py` by `python kaggle/make_notebook.py`, exactly like the
teacher notebook. It expects two private Kaggle datasets:

- `thermo-agent-code`: upload `kaggle_bundle.zip`, built locally by `pwsh scripts/make_kaggle_bundle.ps1`.
- `thermo-agent-corpus`: the `data/sft` and `data/grpo` directories from the local corpus build.

Add a Kaggle Secret `HF_TOKEN` (the same value as `.env` locally). The setup cell creates/verifies the private repo
`Reza-Nadimi/thermo-agent-checkpoints`; the SFT, merge and GRPO cells each resume from its newest checkpoint, so
a cell can be re-run in another 12 h session without losing progress. Set `STEPS`/`ROWS` for a smoke run, and
`TASK=ledger` (or repair/design/audit) for the GRPO cell. The notebook never contains a token literal and never
writes to any Hugging Face repo other than that one.

## Frontier-in-the-harness notebook (tool-use comparison arm)

`kaggle/teacher_harness.ipynb` (source `teacher_harness.py`, rebuilt by `python kaggle/make_notebook.py`) is the third
frontier arm: the same prompts, but our physics tool suite is exposed to the model through
`kbench.llm.prompt(..., tools=[check_answer])`. It records the answer, tool calls, reasoning trace and Kaggle's
token/cost usage per row in `harness_completions.jsonl`, and mirrors progress to
`teacher/harness_progress.jsonl` in the same private repo. Attach `thermo-agent-code`, `thermo-teacher-prompts` and
the `HF_TOKEN` secret. Together with the zero-shot and prompt-only frontier arms and the trained 4B, this lets the
paper measure how much our harness and tools help a frontier model.

## Evaluation notebook and paper figures

`kaggle/eval.ipynb` (source `eval_runner.py`) runs every comparison arm through the same `model_eval` evaluator: our base 4B, the SFT checkpoint with and without the inference text, the frontier model zero-shot and with our tools (replayed from the Kaggle answers), and another model served behind an OpenAI-compatible endpoint. It writes one report JSON per arm, builds `paper/assets/model_results.json` with the family/inference-text/tool axes, aggregates `reports/tool_usage.json` (tool calls, cost, reasoning), and redraws the trace figure.

`paper/tools/collect_examples.py` selects deterministic examples from every stage (datagen, SFT, teacher-in-harness, GRPO rollouts, evaluation) into `paper/assets/examples.json` with `examples_schema.json` beside it. `paper/tools/figures_from_examples.py` turns them into `figures/svg|pdf/fig_trace_walkthrough.*`, a four-panel vector figure for the paper. No figure value is hand-written: every string and number is copied from a stage output.
