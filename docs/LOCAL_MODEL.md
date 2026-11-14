# Running the model locally (LM Studio / any OpenAI-compatible server)

The evaluator does not require local weights. `--backend openai` sends the same messages the HF path would send to any
OpenAI-compatible chat endpoint: LM Studio, vLLM, llama.cpp server, Ollama.

## Start LM Studio

In LM Studio: **Developer** tab → select `qwen3.5-4b` → **Start Server** (default `http://127.0.0.1:1234/v1`).
Check it:

```powershell
Invoke-RestMethod http://127.0.0.1:1234/v1/models | Select-Object -ExpandProperty data | Select-Object id
```

## Evaluate

```bash
.venv/Scripts/python.exe -m thermo_agent.eval.model_eval data/processed \
    --backend openai --api-model qwen3.5-4b \
    --tasks ledger --limit 20 --out reports/local_ledger.json
```

Swap `--tasks` for `repair design audit ledger`. `--tasks design --refine-turns 6` runs the anytime protocol.

## Behaviour

- Greedy by default (`temperature 0`), matching the greedy evaluation protocol used for the paper.
- Reasoning is disabled through `chat_template_kwargs: {"enable_thinking": false}`. Pass `--thinking` to keep it on;
  a `<think>...</think>` block is stripped from the answer either way.
- `--max-new-tokens` defaults to 400, the same budget the HF path uses.
- Requests use `urllib` only, so the evaluation machine needs no extra package.
- `POST /v1/chat/completions` is retried 3 times with a short backoff, which absorbs LM Studio's first-call model load.

## Cost note

A 4B model at Q4 in 6 GB of VRAM generates roughly 20-40 tokens/s. With ~2k-token prompts and a 400-token budget,
`--limit 20` on the ledger task is minutes, not hours. Run small limits first, then scale.

## Reporting

`meta.backend` and `meta.model` record how a run was produced, so reports from this path are not confused with
`transformers` runs on a GPU box.
