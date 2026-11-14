# Kaggle teacher

Kaggle has no general LLM inference API, so the teacher runs where the GPUs are: a Kaggle notebook serves a
frontier open model with vLLM and publishes a cloudflared quick tunnel. The laptop samples through that tunnel
with the same `thermo_agent.teacher.sample` used for any OpenAI-compatible endpoint.

## 1. Notebook

Create a Kaggle notebook with:

- **Accelerator**: `GPU T4 x2`
- **Internet**: on (Settings panel)
- **Persistence**: files only, if you want the raw logs to survive

Cells:

```python
!pip install -q vllm
```

```python
# upload kaggle/serve_teacher.py next to the notebook, or paste it into a cell
!python serve_teacher.py
```

The last cell prints:

```text
TEACHER_BASE_URL=https://<random>.trycloudflare.com
TEACHER_MODEL=teacher
```

Keep the notebook running. Closing it stops the server and the tunnel.

## 2. Model choice

| Model | Size | Fits 2xT4 (30 GB)? |
|---|---|---|
| `Qwen/Qwen2.5-32B-Instruct-AWQ` (default) | ~19 GB | yes, TP=2 |
| `Qwen/Qwen2.5-14B-Instruct-AWQ` | ~9 GB | yes, faster, weaker |
| any fp16 32B+ | 64 GB+ | no |

Set `TEACHER_MODEL` before the script to change it. `--dtype float16` is required: T4 is Turing and has no bf16
kernels, same constraint as the training path in `docs/DUAL_T4.md`.

## 3. Sample from the laptop

```powershell
pwsh scripts/kaggle_teacher.ps1 -Url https://<random>.trycloudflare.com -Samples 2
```

The helper probes `/v1/models`, then runs `teacher.sample` for each task. Sampling is resumable: rows whose ids
are already in `data/frontier/frontier_<task>_<split>.jsonl` are skipped, so a dropped tunnel only costs the
rows in flight.

Every attempt is written raw; only answers that pass the tool suite become SFT rows, and rejections become
hindsight memory in `data/memory/`.

## 4. Quota

Kaggle gives roughly 30 GPU-hours per week. A 32B AWQ model on 2xT4 generates slowly, so start with
`-Limit 50 -Samples 2` and check the acceptance rate before committing a full pass. The sampler prints
`attempts / accepted / rejected` per task.
