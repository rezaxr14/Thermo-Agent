# Kaggle: the frontier-teacher pass

The second half of the training corpus comes from a frontier model whose answers must pass the same tool suite as
everything else. Kaggle is used for the model access (Kaggle Benchmarks, i.e. `kaggle_benchmarks` with Kaggle's
own model quota); the physics checks run locally, so nothing the teacher writes enters the corpus unverified.

The path was removed earlier in this project as a self-hosted vLLM server behind a cloudflared tunnel. That is
gone; what remains is a three-step hand-off between this machine and a Kaggle notebook.

## 1. Export the prompts (local)

```powershell
python -m thermo_agent.teacher.transfer export data/sft data/frontier_export --per-task 400
```

`data/frontier_export/` holds `{task}_train.jsonl` with **one prompt per ground-truth plant** (paraphrase variants
are a student-side device), the harness and retrieved failure notes already inside the prompt, and **no ground-truth
completion**, so the notebook cannot leak the answer to the teacher. `export_manifest.json` records the counts, the
harness mix and the maximum prompt length.

## 2. Run the teacher (Kaggle)

1. Upload `data/frontier_export/` as a private Kaggle dataset (`kaggle datasets create -p data/frontier_export -u`),
   or attach it as a notebook input.
2. Open <https://www.kaggle.com/benchmarks/tasks/new>: that notebook already has `kaggle_benchmarks` installed.
3. Paste `kaggle/teacher_benchmark.py` into a cell, set `MODEL` to a key of `kbench.llms` (for example
   `anthropic/claude-sonnet-5-5@default`) and `DATASET` to the input path, and run it.
4. Download `/kaggle/working/completions.jsonl`.

The loop writes one JSON line per attempt and skips ids it has already answered, so a session that dies keeps its
progress. Failures are recorded with their error instead of aborting the run.

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
