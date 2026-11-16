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
