"""End-to-end checkpoint smoke test: train 2 steps on a tiny CPU model, push, restore, resume.

    python scripts/smoke_hf_resume.py

Uses the project's private HF repo under the `smoke/` prefix. No data, no GPU, no large downloads: the model is
built from a config with random weights. This proves the callback, the repo guard and the restore path all work
together before the same code runs on two T4s.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thermo_agent.train import hf_sync  # noqa: E402

STAGE = "smoke"


def dataset():
    from datasets import Dataset
    # 16 short, distinct examples so the Trainer has real batches to iterate.
    return Dataset.from_list([{"text": f"example {i} " * 8} for i in range(16)])


def tokenize(ds, tok):
    def _tok(r):
        out = tok(r["text"], truncation=True, max_length=32)
        out["labels"] = out["input_ids"]
        return out
    return ds.map(_tok, remove_columns=["text"])


def build_trainer(root, tok, model_factory, steps):
    import peft
    from transformers import Trainer, TrainingArguments

    model = peft.get_peft_model(model_factory(), peft.LoraConfig(
        r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM"))
    args = TrainingArguments(
        output_dir=str(root), max_steps=steps, per_device_train_batch_size=2, learning_rate=1e-3,
        save_steps=1, save_total_limit=2, logging_steps=1, report_to="none", use_cpu=True,
        remove_unused_columns=False,
    )
    return Trainer(model=model, args=args, train_dataset=tokenize(dataset(), tok),
                   callbacks=[hf_sync.make_callback(repo, STAGE, token)]), model


if __name__ == "__main__":
    token = hf_sync.read_token(ROOT / ".env")
    repo = hf_sync.ensure_private_repo(token=token)
    print("repo:", repo, "(private)", flush=True)

    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    factory = lambda: GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32,
                                                vocab_size=len(tok), n_positions=64))

    first = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "runs/_smoke_first")
    shutil.rmtree(first, ignore_errors=True)
    trainer, model = build_trainer(first, tok, factory, steps=2)
    trainer.train()
    steps = hf_sync.local_steps(first.parent, first.name)
    assert steps and max(steps) == 2, f"the Trainer did not write both checkpoints: {steps}"
    print("first session finished; local steps:", steps, flush=True)

    # The callback uploads under <stage>/checkpoint-N. Restore it into a brand-new directory.
    second = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "runs/_smoke_second")
    shutil.rmtree(second, ignore_errors=True)
    remote = hf_sync.remote_steps(hf_sync.repo_files(repo, token), STAGE)
    print("remote smoke steps:", remote, flush=True)
    assert remote, "the callback did not push a checkpoint"
    pulled = hf_sync.pull_checkpoint(second.parent, repo, STAGE, token)
    restored = pathlib.Path(pulled)
    assert restored.name == f"checkpoint-{max(remote)}"
    state = json.loads((restored / "trainer_state.json").read_text(encoding="utf-8"))
    assert state["global_step"] == max(remote)
    assert (restored / "adapter_model.safetensors").exists() and (restored / "optimizer.pt").exists()
    print("restored", restored, "at step", state["global_step"], flush=True)

    # Resume from the restored directory and take one more step.
    resumed, _ = build_trainer(second, tok, factory, steps=max(remote) + 1)
    resumed.train(resume_from_checkpoint=str(restored))
    assert resumed.state.global_step == max(remote) + 1
    print("smoke test OK: resumed from step", max(remote), "and reached", resumed.state.global_step, flush=True)
