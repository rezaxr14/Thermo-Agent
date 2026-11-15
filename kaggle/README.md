# Kaggle (deferred)

The frontier-teacher pass will run through **Kaggle's own model access** (Kaggle Benchmarks / the Kaggle CLI
models), using Claude Sonnet/Haiku, OpenAI and Gemini models. It will not be a self-hosted small model behind
a tunnel; that path was removed.

Nothing here is needed for the local corpus. `scripts/gen_data.ps1` builds and validates the whole training set
on this machine. When the Kaggle pass starts:

1. export the prompts (system message with the harness and the retrieved failure notes, plus the architecture
   corpus) to JSONL;
2. run them in a Kaggle notebook against the chosen Benchmarks models and download the completions;
3. validate every completion with `thermo_agent.teacher.validate` and merge the accepted rows with
   `thermo_agent.synth.combine`, which already reads `data/frontier/frontier_{task}_{split}.jsonl`.

The plan is recorded in `docs/RUNBOOK.md`; the export/ingest helpers will be added when that pass starts.
