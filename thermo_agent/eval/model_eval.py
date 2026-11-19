"""Evaluate a model (base or LoRA adapter) on Repair, Design and Audit, with an optional agent-harness mode.

    python -m thermo_agent.eval.model_eval DATA_DIR [DATA_DIR ...] --model Qwen/Qwen3.5-4B [--adapter runs/grpo_repair] \\
        [--tasks repair design audit] [--limit 40] [--harness none|agents|skills|agents+skills|tools] --out reports/base.json
    # "tools" is a legacy arm label (frontier replays): an alias for agents+skills during
    # prompt augmentation, kept verbatim in the report as harness_label.

    # the second directory is the architecture corpus, so the held-out diagrams are evaluated too:
    python -m thermo_agent.eval.model_eval data/processed data/processed_arch --model Qwen/Qwen3.5-4B --out reports/base.json

    # a served model instead of local weights (LM Studio, vLLM, Ollama):
    python -m thermo_agent.eval.model_eval DATA_DIR --backend openai --api-model qwen3.5-4b --out reports/local.json

Nothing here depends on torch except HFGenerator; every evaluator takes any callable generate(messages) -> str."""
from __future__ import annotations
import argparse, hashlib, json, pathlib, platform, re, statistics as st, subprocess, time, urllib.request
from typing import Callable
from ..agent import harness as H
from ..agent import toolcall as TC
from ..agent.tool_prompt import FINAL_TAGS as _FINAL_TAGS
from ..agent.tool_prompt import build_harness_tool_messages, tool_contract as _tool_contract
from ..agent.loop import repair_loop
from ..core.schema import Plant
from ..design.loop import design_loop, design_refine
from ..audit import rewards as audit_rewards
from ..ledger import rewards as ledger_rewards

DEFAULT_SPLITS = {"repair": ["test_id", "test_ood_fault", "test_ood_topology"],
                  "design": ["test_id", "test_ood_climate", "test_ood_scale", "test_ood_load"],
                  "audit": ["test_id", "test_ood_topology"], "ledger": ["test_id", "test_ood_fault", "test_ood_topology"]}
_THINK = re.compile(r"<think>.*?</think>", re.S)

# --- Gate 7: tool-using multi-turn generator for the H arm -----------------
# The prompt-text --harness modes keep their historical limits exactly; the H arm
# (--tool-access tools) gets a larger output budget and the shared agent turn limit.
try:  # keep the turn limit identical to the shared Gate 2 agent environment
    from ..teacher.agent_graph import DEFAULT_MAX_ROUNDS as _DEFAULT_TOOL_TURNS
except Exception:  # langgraph is not needed for prompt-text (ablation) evals
    _DEFAULT_TOOL_TURNS = 4
_DEFAULT_TEXT_TURNS = 3
_DEFAULT_TOOL_CALLS = 8      # I6
_TOOL_MAX_NEW_TOKENS = 4096   # I7, per turn
_TEXT_MAX_NEW_TOKENS = 4096   # I7: thinking is on, 400 truncates it
_REVISE_NUDGE = ("Tool results are above inside <tool_result> blocks. Read the numbers, revise, "
                "and either call another tool or STOP and emit ONLY the final answer XML tags.")
_FINAL_NUDGE = ("Your last message was not a final answer. Emit ONLY the task's final XML tags "
               "now (no tool calls, no prose).")


# --- Harness label normalization (legacy "tools" replay arms) ---------------
# ``tools`` is a historical arm label carried by the frontier replay arms in
# kaggle/eval_runner.py.  It is NOT a member of H.MODES, so it must never reach
# H.augment.  ReplayGenerator ignores messages (the scored prompt is frozen in
# the replay file), but the label still flows through the evaluator for report
# metadata.  For augmentation it means the full agents+skills harness; the
# original label is preserved separately in the report as ``harness_label``.
_HARNESS_ALIASES = {"tools": "agents+skills"}


def normalize_harness(value) -> str:
    """Return a valid ``H.MODES`` value for actual prompt augmentation.

    ``"tools"`` maps to ``"agents+skills"``; every valid mode is returned
    unchanged and ``None``/empty means ``"none"``.  Unknown labels raise so a
    malformed arm fails loudly instead of silently changing the scored prompt.
    """
    label = "none" if value is None else str(value)
    if label in _HARNESS_ALIASES:
        return _HARNESS_ALIASES[label]
    if label == "":
        return "none"
    if label not in H.MODES:
        raise ValueError(f"unknown harness {label!r}; choose from {list(H.MODES) + list(_HARNESS_ALIASES)}")
    return label


def harness_config(mode, tool_access, max_turns=None, max_tool_calls=None, max_new_tokens=None) -> dict:
    """Resolved per-arm limits. ``None`` means 'not given on the CLI'.

    Tool turns get the larger budget / shared agent turn count; prompt-text (ablation)
    arms keep the historical 400 tokens and 3 turns. Recorded on every arm so Z/H are comparable.
    """
    mode = normalize_harness(mode)
    tools = tool_access == "tools"
    turns = max_turns if max_turns is not None else (_DEFAULT_TOOL_TURNS if tools else _DEFAULT_TEXT_TURNS)
    return {
        "mode": mode,
        "tool_access": tool_access,
        "max_turns": turns,
        "max_tool_calls": _DEFAULT_TOOL_CALLS if max_tool_calls is None else max_tool_calls,
        "max_new_tokens": ((_TOOL_MAX_NEW_TOKENS if tools else _TEXT_MAX_NEW_TOKENS)
                           if max_new_tokens is None else max_new_tokens),
    }


# --- Report provenance (frozen EVAL_PROTOCOL §1/§4) ---------------------------
# These helpers only READ scored split files / CLI config; generation is untouched.

def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def eval_set_hash(data_dirs, tasks, splits=None) -> str:
    """SHA-256 over sorted scored split-file names followed by each file's bytes (protocol §1).

    The file set is exactly the (task, split) cells `run` scores, so a report's version always
    describes the rows it actually used. With ``--limit 0`` every row of each file is scored.
    """
    dirs = [data_dirs] if isinstance(data_dirs, (str, pathlib.Path)) else list(data_dirs)
    files = []
    for task in tasks:
        for sp in (splits if splits is not None else DEFAULT_SPLITS.get(task, [])):
            for d in dirs:
                p = pathlib.Path(d) / f"{task}_{sp}.jsonl"
                if p.is_file():
                    files.append(p)
    files.sort(key=lambda p: (p.name, str(p)))
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


def harness_config_hash(cfg: dict, limit: int) -> str:
    """SHA-256 of the canonical harness/limits config.

    `LIMIT` is part of the limits config (protocol §4), so it is hashed alongside
    harness_config without mutating that dict (which the tool loop also consumes).
    """
    return hashlib.sha256(_canonical_json({**cfg, "limit": limit}).encode("utf-8")).hexdigest()


def _git_commit() -> str | None:
    """Best-effort `git rev-parse HEAD`; never raises when git/source control is absent."""
    try:
        root = pathlib.Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=5)
        sha = (out.stdout or "").strip()
        return sha if out.returncode == 0 and sha else None
    except Exception:
        return None


def _scored_ids(report: dict, tasks) -> dict:
    """Exact ids scored, per task/split, as read from the per-example arrays."""
    out = {}
    for task in tasks:
        out[task] = {}
        for sp, m in (report.get(task) or {}).items():
            out[task][sp] = ([ex["id"] for ex in m.get("examples", []) if "id" in ex]
                             if isinstance(m, dict) else [])
    return out


def check_no_truncation(report: dict, data_dirs, tasks, splits, limit) -> None:
    """Raise if any scored split has fewer ids than its manifest (all rows, no cap).

    A number computed on a truncated held-out cell is not evidence: fail loudly instead of
    writing a report. The error names the split and both counts.
    """
    problems = []
    for task in tasks:
        for sp in (splits if splits is not None else DEFAULT_SPLITS.get(task, [])):
            manifest = load_rows(data_dirs, task, sp, None)
            if not manifest:
                continue
            manifest_ids = {r["id"] for r in manifest}
            cell = (report.get(task) or {}).get(sp)
            scored = ({ex["id"] for ex in cell.get("examples", []) if "id" in ex}
                      if isinstance(cell, dict) else set())
            if len(scored) < len(manifest_ids):
                problems.append(f"{task}/{sp} scored {len(scored)} of {len(manifest_ids)} manifest ids")
    if problems:
        raise ValueError("held-out cells truncated (limit=%s); refusing to write a report: %s"
                         % (limit, "; ".join(problems)))


class HFGenerator:
    """Greedy chat generation with transformers (+ peft adapter). Untested in the authoring environment (no GPU)."""
    def __init__(self, model_id, adapter=None, dtype="bfloat16", load_4bit=False, max_new_tokens=400,
                 enable_thinking=True):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(model_id); self.max_new_tokens = max_new_tokens
        # The SFT/GRPO corpora are rendered with enable_thinking=True, so eval must use the
        # same chat-template prefix or the model sees a prompt shape it never trained on.
        self.enable_thinking = bool(enable_thinking)
        # Populated after each call so ToolLoop can record token use / truncation.
        self.last_prompt_tokens = 0; self.last_completion_tokens = 0; self.last_truncated = False
        kw = {"device_map": "auto"}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, dtype))
        try: self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype), **kw)
        except TypeError: self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=getattr(torch, dtype), **kw)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def __call__(self, messages: list[dict]) -> str:
        import torch
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                            enable_thinking=self.enable_thinking)
        inputs = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        n_in = inputs["input_ids"].shape[1]
        gen = out[0][n_in:]
        self.last_prompt_tokens = int(n_in)
        self.last_completion_tokens = int(gen.shape[0])
        self.last_truncated = self.last_completion_tokens >= self.max_new_tokens
        return _THINK.sub("", self.tok.decode(gen, skip_special_tokens=True)).strip()


class OpenAIGenerator:
    """Chat generation against any OpenAI-compatible endpoint: LM Studio, Ollama, vLLM, llama.cpp server.

    Greedy by default (temperature 0) so the protocol matches the HF evaluator. Requests use urllib, so evaluating a
    served model needs no extra package. The server applies its own chat template; for Qwen3-style checkpoints thinking
    is switched off through chat_template_kwargs, and any reasoning trace is stripped from the answer."""

    def __init__(self, base_url="http://127.0.0.1:1234/v1", model=None, api_key="lm-studio", max_new_tokens=400,
                 temperature=0.0, seed=None, timeout=900, enable_thinking=False, retries=3):
        self.base_url = base_url.rstrip("/"); self.model = model; self.api_key = api_key
        self.max_new_tokens = max_new_tokens; self.temperature = temperature; self.seed = seed
        self.timeout = timeout; self.enable_thinking = enable_thinking; self.retries = retries
        # Populated after each call so ToolLoop can record token use / truncation.
        self.last_prompt_tokens = 0; self.last_completion_tokens = 0; self.last_truncated = False

    def _request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(f"{self.base_url}{path}", data=data, method="GET" if data is None else "POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def resolve_model(self):
        """Use --api-model if given, else the first model the server lists."""
        if not self.model:
            ids = [m.get("id") for m in self._request("/models").get("data", []) if m.get("id")]
            if not ids: raise RuntimeError(f"no model served at {self.base_url}/models")
            self.model = ids[0]
        return self.model

    def __call__(self, messages: list[dict] | str) -> str:
        if isinstance(messages, str): messages = [{"role": "user", "content": messages}]
        payload = {"model": self.resolve_model(), "messages": messages, "temperature": self.temperature,
                   "max_tokens": self.max_new_tokens, "stream": False}
        if self.seed is not None: payload["seed"] = self.seed
        if self.enable_thinking is not None: payload["chat_template_kwargs"] = {"enable_thinking": bool(self.enable_thinking)}
        last = None
        for attempt in range(self.retries):
            try:
                out = self._request("/chat/completions", payload)
                choice = out["choices"][0]
                msg = choice["message"]
                usage = out.get("usage") or {}
                self.last_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                self.last_completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                self.last_truncated = choice.get("finish_reason") == "length"
                return _THINK.sub("", (msg.get("content") or msg.get("reasoning_content") or "")).strip()
            except Exception as exc:  # transient server or connection errors are retried
                last = exc; time.sleep(1 + attempt)
        raise RuntimeError(f"chat completion failed after {self.retries} attempts: {last}")


class ReplayGenerator:
    """Replay stored Kaggle completions, keyed by row id, through the same evaluator as local models.

    Frontier answers produced by `kaggle/teacher.ipynb` are static: one answer per prompt, no solver feedback inside
    the stored text. Replaying them measures the zero-shot (or harness-augmented) arm; for Repair/Design the loop may
    call the generator again with solver feedback, and this returns the same stored answer each turn. That limitation
    is recorded in the report meta so the comparison is honest.
    """

    def __init__(self, path, reasoning_key="reasoning"):
        self.answers, self.reasoning, self.current_id = {}, {}, None
        for line in pathlib.Path(path).open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("completion") and not r.get("error"):
                self.answers[r["id"]] = r["completion"]
                if r.get(reasoning_key):
                    self.reasoning[r["id"]] = r[reasoning_key]

    def set_current_id(self, rid):
        self.current_id = rid

    def __call__(self, messages):
        if self.current_id not in self.answers:
            raise KeyError(f"no stored completion for row {self.current_id}; re-export the matching split")
        return _THINK.sub("", self.answers[self.current_id]).strip()


def wilson(k: int, n: int, z: float = 1.96):
    """95 % Wilson score interval for a proportion k/n."""
    if n == 0: return None
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 4) if xs else None


def _wrap(generate, mode, task, plant, log, rid):
    def g(messages):
        if hasattr(generate, "set_current_id"):
            generate.set_current_id(rid)
        text = generate(H.augment(messages, mode, task, plant))
        log.append({"id": rid, "task": task, "completion": text}); return text
    return g


def eval_repair(generate, rows, mode="none", max_turns=3, log=None):
    log = [] if log is None else log; res = []
    for r in rows:
        g = _wrap(generate, mode, "repair", Plant.model_validate_json(r["plant_broken"]), log, r["id"])
        res.append({**repair_loop(g, r, max_turns), "deducible": r["deducible"], "fault": r["fault_type"], "id": r["id"]})
    ded = [x for x in res if x["deducible"]]
    return {"n": len(res), "success": _mean(x["success"] for x in res), "success_ci95": wilson(sum(x["success"] for x in res), len(res)),
            "first_turn_success": _mean(x["success"] and x["turns"] == 1 for x in res),
            "exact_on_deducible": _mean(x["exact"] for x in ded), "mean_turns": _mean(x["turns"] for x in res), "mean_solves": _mean(x["solves"] for x in res),
            "by_fault": {f: _mean(x["success"] for x in res if x["fault"] == f) for f in sorted({x["fault"] for x in res})},
            "examples": [{"id": x["id"], "success": bool(x["success"]), "first_turn_success": bool(x["success"] and x["turns"] == 1),
                          "exact": bool(x["exact"]) if x["deducible"] else None, "solves": x["solves"]} for x in res]}


def eval_design(generate, rows, mode="none", max_turns=3, polish_evals=0, log=None, refine_turns=0):
    from ..core.schema import Site
    log = [] if log is None else log; res = []
    for r in rows:
        g = _wrap(generate, mode, "design", None, log, r["id"])
        res.append({**(design_refine(g, r, refine_turns) if refine_turns else design_loop(g, r, max_turns, polish_evals)), "id": r["id"]})
    ok = [x["gap"] for x in res if x["valid"]]
    w1 = sum(bool(x["valid"] and x["gap"] <= 0.01) for x in res)
    return {"n": len(res), "valid": _mean(x["valid"] for x in res), "valid_ci95": wilson(sum(x["valid"] for x in res), len(res)), "mean_gap_pct": _mean(100 * g for g in ok),
            "within_1pct": _mean((x["valid"] and x["gap"] <= 0.01) for x in res), "within_1pct_ci95": wilson(w1, len(res)), "mean_evals": _mean(x["evals"] for x in res),
            "polish_evals": polish_evals, "refine_turns": refine_turns,
            "examples": [{"id": x["id"], "valid": bool(x["valid"]), "gap": x["gap"], "within_1pct": bool(x["valid"] and x["gap"] <= 0.01), "evals": x["evals"],
                          "anytime": x.get("anytime")} for x in res]}


def eval_audit(generate, rows, mode="none", log=None):
    log = [] if log is None else log; comps = []
    for r in rows:
        g = _wrap(generate, mode, "audit", Plant.model_validate_json(r["plant"]), log, r["id"])
        comps.append(g(r["prompt"]))
    kw = {k: [r[k] for r in rows] for k in ("plant", "site", "ed_json", "fuel_w", "net_w", "best_gain_w")}
    out = {"n": len(rows)}; per = {}
    for f in audit_rewards.REWARD_FUNCS: per[f.__name__[2:]] = f(None, comps, **kw); out[f.__name__[2:]] = _mean(per[f.__name__[2:]])
    out["examples"] = [{"id": r["id"], **{k: v[i] for k, v in per.items()}} for i, r in enumerate(rows)]
    return out


def eval_ledger(generate, rows, mode="none", log=None):
    log = [] if log is None else log; comps = []
    for r in rows:
        g = _wrap(generate, mode, "ledger", None, log, r["id"]); comps.append(g(r["prompt"]))
    kw = {"ledger": [r["ledger"] for r in rows]}; out = {"n": len(rows)}
    per = {}
    for f in ledger_rewards.REWARD_FUNCS: per[f.__name__[2:]] = f(None, comps, **kw); out[f.__name__[2:]] = _mean(per[f.__name__[2:]])
    flags = [bool(ledger_rewards.evaluate(c, r["ledger"]).get("all_correct")) for c, r in zip(comps, rows)]; ok = sum(flags)
    out["all_correct"] = round(ok / max(len(rows), 1), 4); out["all_correct_ci95"] = wilson(ok, len(rows))
    out["examples"] = [{"id": r["id"], "all_correct": flags[i], **{k: v[i] for k, v in per.items()}} for i, r in enumerate(rows)]
    return out


def load_rows(data_dirs, task, split, limit):
    """Rows for one split from one or more data directories, de-duplicated by id.

    Evaluation reads the ground-truth datasets (`data/processed`, plus `data/processed_arch` for the architecture
    diagrams), never the combined corpus: the combined prompts already carry the harness on a fraction of rows, so
    evaluating there would both double-augment and make the `--harness` ablation meaningless.
    """
    dirs = [data_dirs] if isinstance(data_dirs, (str, pathlib.Path)) else list(data_dirs)
    rows, seen = [], set()
    for d in dirs:
        p = pathlib.Path(d) / f"{task}_{split}.jsonl"
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            rid = row.get("id")
            if rid in seen:
                continue
            seen.add(rid)
            rows.append(row)
    if not rows:
        return None
    return rows[:limit] if limit else rows


def parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    """Parse tool calls via the shared helper (our inline format and Qwen3.5 native).

    Kept as a thin tuple-returning adapter so the tool loop below is unchanged: only
    well-formed blocks are executed and all five physics tools read ``candidate``.
    """
    return [(tc.name, tc.arguments) for tc in TC.parse_tool_calls(text)]


def has_final_xml(text: str, task: str) -> bool:
    """True when the text carries every XML tag required for the task's final answer."""
    tags = _FINAL_TAGS.get(task)
    return bool(tags) and all(t in (text or "") for t in tags)


class ToolLoop:
    """Bounded propose -> tool_call -> tool_result -> final loop around any ``generate(messages) -> str``.

    Works with HFGenerator, OpenAIGenerator, ReplayGenerator and plain fake callables: the
    backend needs no native tool calling. It returns only the final assistant text (the shape
    every existing grader expects) and records the whole episode in ``last_record``. The five
    tools' contracts are given to the model as text; no ground truth or verifier verdict is
    ever placed in the prompt and there is no answer-oracle tool.
    """

    def __init__(self, generate, row, task, mode="none", max_turns=None, max_tool_calls=_DEFAULT_TOOL_CALLS):
        self.generate = generate
        self.row = row
        self.task = task
        self.mode = mode
        cfg = harness_config(mode, "tools", max_turns=max_turns, max_tool_calls=max_tool_calls)
        self.max_turns = cfg["max_turns"]
        self.max_tool_calls = cfg["max_tool_calls"]
        self.last_record = None

    tool_contract = staticmethod(_tool_contract)

    def _base_messages(self):
        # One shared Gate 2 environment per episode: same five tools, same caps as SFT/GRPO.
        from ..agent.env import ThermoEnv
        env = ThermoEnv(self.row, self.task, self.max_turns, self.max_tool_calls)
        tools = env.tools
        plant = None
        for key in ("plant_broken", "plant", "plant_original"):
            if self.row.get(key):
                try:
                    plant = Plant.model_validate_json(self.row[key]); break
                except Exception:
                    pass
        msgs = build_harness_tool_messages(
            list(self.row["prompt"]), self.task, plant,
            mode=self.mode, max_turns=self.max_turns, max_tool_calls=self.max_tool_calls,
            tools=tools)
        return msgs, {f.__name__: f for f in tools}

    def run(self) -> str:
        """Run the bounded episode; return the final text and set ``last_record``."""
        if hasattr(self.generate, "set_current_id"):
            self.generate.set_current_id(str(self.row.get("id", "")))
        msgs, tool_map = self._base_messages()
        tool_calls, result_chars, turn_log = [], [], []
        p_tokens = c_tokens = 0
        truncated = False
        final_text = ""
        n_calls = 0
        stop = "max_turns"
        for turn in range(1, self.max_turns + 1):
            text = self.generate(msgs) or ""
            p_tokens += int(getattr(self.generate, "last_prompt_tokens", 0) or 0)
            c_tokens += int(getattr(self.generate, "last_completion_tokens", 0) or 0)
            if getattr(self.generate, "last_truncated", False):
                truncated = True
            parsed = parse_tool_calls(text)
            if not parsed:
                final_text = text
                turn_log.append({"turn": turn, "text": text, "tool_calls": [], "tool_results": []})
                if has_final_xml(text, self.task):
                    stop = "final"
                    break
                # Prose (or malformed output) is not a final answer: nudge and keep going.
                msgs = msgs + [{"role": "assistant", "content": text},
                               {"role": "user", "content": _FINAL_NUDGE}]
                continue
            executed = []
            for name, payload in parsed:
                if n_calls >= self.max_tool_calls:
                    stop = "max_tool_calls"
                    break
                fn = tool_map.get(name)
                candidate = payload.get("candidate", "")
                if fn is None:
                    result = json.dumps({"ok": False, "error": f"unknown tool {name}"}, ensure_ascii=False)
                else:
                    try:
                        result = fn(candidate)
                    except Exception as exc:  # a bad call must not kill the episode
                        result = json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)
                executed.append((name, result))
                tool_calls.append(name); result_chars.append(len(result)); n_calls += 1
            turn_log.append({"turn": turn, "text": text,
                             "tool_calls": [n for n, _ in executed],
                             "tool_results": [r for _, r in executed]})
            if executed:
                final_text = text
                blocks = "".join(f"<tool_result>{r}</tool_result>" for _, r in executed)
                msgs = msgs + [{"role": "assistant", "content": text},
                               {"role": "user", "content": blocks + "\n" + _REVISE_NUDGE}]
                if stop == "max_tool_calls":
                    break
                continue
            # A tool block was present but the call cap was already exhausted.
            final_text = text
            break
        self.last_record = {
            "turns": len(turn_log),
            "tool_calls": tool_calls,
            "tool_results_chars": result_chars,
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "final_text": final_text,
            "truncated": truncated,          # token budget hit mid-turn
            "limit_reached": stop != "final",  # turn / tool-call / token limit stopped the loop
            "stop_reason": stop,             # final | max_turns | max_tool_calls
            "turn_log": turn_log,            # structured per-turn record
        }
        return final_text

    def __call__(self, messages=None) -> str:
        return self.run()


class _AnswerReplay:
    """Feed pre-computed tool-loop final answers through the existing graders unchanged."""

    def __init__(self, answers):
        self.answers = answers
        self.current_id = None

    def set_current_id(self, rid):
        self.current_id = rid

    def __call__(self, messages):
        return self.answers[self.current_id]


def _tool_summary(records) -> dict:
    rs = list(records.values())
    return {"n": len(rs),
            "mean_turns": _mean(r["turns"] for r in rs),
            "mean_tool_calls": _mean(len(r["tool_calls"]) for r in rs),
            "truncated_rate": round(sum(r["truncated"] for r in rs) / max(len(rs), 1), 4),
            "limit_reached_rate": round(sum(r["limit_reached"] for r in rs) / max(len(rs), 1), 4)}


def eval_with_tools(generate, rows, task, mode, max_turns, max_tool_calls, log,
                    polish_evals=0, refine_turns=0):
    """H arm: run the tool episode per row, then grade its final answer with the same graders."""
    answers, records = {}, {}
    for r in rows:
        loop = ToolLoop(generate, r, task, mode, max_turns, max_tool_calls)
        answers[r["id"]] = loop.run()
        records[r["id"]] = loop.last_record
    replay = _AnswerReplay(answers)
    if task == "repair":
        m = eval_repair(replay, rows, mode="none", max_turns=1, log=log)
    elif task == "design":
        m = eval_design(replay, rows, mode="none", max_turns=1, log=log,
                        polish_evals=polish_evals, refine_turns=0)
    elif task == "ledger":
        m = eval_ledger(replay, rows, mode="none", log=log)
    else:
        m = eval_audit(replay, rows, mode="none", log=log)
    for ex in m.get("examples", []):
        if ex.get("id") in records:
            ex["tool_episode"] = records[ex["id"]]
    for entry in log:
        if entry.get("task") == task and entry.get("id") in records:
            entry["tool_episode"] = records[entry["id"]]
    m["tool_summary"] = _tool_summary(records)
    return m


def run(generate, data_dirs, tasks, splits=None, limit=40, mode="none", max_turns=None, polish_evals=0, refine_turns=0,
        tool_access="none", max_tool_calls=None, max_new_tokens=None) -> tuple[dict, list]:
    # Normalize once for every backend (hf / openai / replay). Only actual generators
    # augment prompts: ReplayGenerator still ignores the messages it is handed, so a
    # replay arm's scored prompt stays exactly the one stored in its replay file.
    mode = normalize_harness(mode)
    cfg = harness_config(mode, tool_access, max_turns, max_tool_calls, max_new_tokens)
    report, log = {}, []
    for task in tasks:
        report[task] = {}
        for sp in (splits or DEFAULT_SPLITS[task]):
            rows = load_rows(data_dirs, task, sp, limit)
            if not rows: report[task][sp] = None; continue
            t = time.time()
            if tool_access == "tools":
                m = eval_with_tools(generate, rows, task, mode, cfg["max_turns"], cfg["max_tool_calls"], log,
                                    polish_evals, refine_turns)
            elif task == "repair": m = eval_repair(generate, rows, mode, cfg["max_turns"], log)
            elif task == "design": m = eval_design(generate, rows, mode, cfg["max_turns"], polish_evals, log, refine_turns)
            elif task == "ledger": m = eval_ledger(generate, rows, mode, log)
            else: m = eval_audit(generate, rows, mode, log)
            m["seconds"] = round(time.time() - t, 1); report[task][sp] = m
    report["harness_config"] = cfg
    return report, log


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data", nargs="+", help="ground-truth data dirs, e.g. data/processed data/processed_arch"); ap.add_argument("--model", default="Qwen/Qwen3.5-4B"); ap.add_argument("--adapter")
    ap.add_argument("--tasks", nargs="+", default=["repair", "design", "audit", "ledger"]); ap.add_argument("--splits", nargs="*")
    ap.add_argument("--limit", type=int, default=40); ap.add_argument("--harness", default="none",
                    choices=list(H.MODES) + list(_HARNESS_ALIASES),
                    help="'tools' is a legacy label for replay arms: alias for agents+skills, kept as harness_label in the report")
    ap.add_argument("--max-turns", type=int, default=None, help="assistant turns; default 4 with --tool-access tools, else 3")
    ap.add_argument("--max-tool-calls", type=int, default=None, help="hard cap on tool calls per H episode (default 6)")
    ap.add_argument("--polish-evals", type=int, default=0)
    ap.add_argument("--refine-turns", type=int, default=0, help="Design: anytime protocol, the model sees the fuel of each valid design and may improve it (records an anytime curve)")
    ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--load-4bit", action="store_true"); ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=None, help="per-turn output budget; default 2048 with --tool-access tools, else 400")
    ap.add_argument("--backend", default="hf", choices=["hf", "openai", "replay"], help="hf loads weights with transformers; openai calls a served model (LM Studio, vLLM, Ollama); replay scores stored completions")
    ap.add_argument("--replay", default=None, help="replay backend: completions.jsonl from a Kaggle teacher notebook")
    ap.add_argument("--model-family", default=None, help="ours | frontier | domain-model | baseline: recorded on the report for paper comparisons")
    ap.add_argument("--tool-access", default="none", choices=["none", "tools"])
    ap.add_argument("--display-name", default=None, help="label used in paper tables")
    ap.add_argument("--zero-shot", action="store_true", help="no weight updates (frontier baselines)")
    ap.add_argument("--base-url", default="http://127.0.0.1:1234/v1", help="openai backend: endpoint root")
    ap.add_argument("--api-model", default=None, help="openai backend: served model id (default: first model listed)")
    ap.add_argument("--api-key", default="lm-studio")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--thinking", action="store_true", help="openai backend: leave the model's reasoning mode enabled")
    ap.add_argument("--no-thinking", action="store_true",
                    help="hf backend: render the prompt with enable_thinking=False (default: True, "
                         "matching how the SFT/GRPO corpora were rendered)")
    a = ap.parse_args()
    harness_label = a.harness                          # requested label (may be legacy "tools")
    harness_mode = normalize_harness(harness_label)    # effective mode for prompt augmentation
    if harness_label != harness_mode:
        print(f"[model_eval] note: --harness {harness_label} is a legacy label for {harness_mode}; "
              f"augmentation uses {harness_mode}, report keeps harness_label={harness_label}.")
    cfg = harness_config(harness_mode, a.tool_access, a.max_turns, a.max_tool_calls, a.max_new_tokens)
    if a.backend == "replay":
        if not a.replay:
            raise SystemExit("--backend replay needs --replay completions.jsonl")
        gen = ReplayGenerator(a.replay)
        model_label = a.display_name or f"replay:{pathlib.Path(a.replay).name}"
    elif a.backend == "openai":
        gen = OpenAIGenerator(a.base_url, a.api_model, a.api_key, cfg["max_new_tokens"], a.temperature, a.seed,
                              enable_thinking=(None if a.thinking and not a.no_thinking else False))
        model_label = a.api_model or f"(auto) {a.base_url}"
    else:
        gen = HFGenerator(a.model, a.adapter, a.dtype, a.load_4bit, cfg["max_new_tokens"],
                          enable_thinking=not a.no_thinking)
        model_label = a.model
    report, log = run(gen, a.data, a.tasks, a.splits, a.limit, harness_label, a.max_turns, a.polish_evals, a.refine_turns,
                      a.tool_access, a.max_tool_calls, a.max_new_tokens)
    # A truncated held-out cell is not evidence: raise before writing any number.
    check_no_truncation(report, a.data, a.tasks, a.splits, a.limit)
    version = eval_set_hash(a.data, a.tasks, a.splits)
    arm_cfg = {"display_name": a.display_name or model_label, "family": a.model_family,
               "harness": harness_mode, "harness_label": harness_label, "harness_mode": harness_mode,
               "tool_access": a.tool_access, "zero_shot": bool(a.zero_shot),
               "backend": a.backend,
               "model": (a.model if a.backend == "hf" else
                         (a.replay if a.backend == "replay" else a.api_model)),
               "data_dirs": [str(d) for d in a.data], "splits": a.splits}
    import importlib.metadata as md
    meta = {"model": model_label, "adapter": a.adapter, "backend": a.backend,
            "display_name": a.display_name or model_label, "model_family": a.model_family,
            "tool_access": a.tool_access, "zero_shot": bool(a.zero_shot), "replay_file": a.replay,
            "data": a.data, "base_url": (a.base_url if a.backend == "openai" else None),
            "harness": harness_mode, "harness_label": harness_label, "harness_mode": harness_mode,
            "limit": a.limit, "max_turns": cfg["max_turns"], "polish_evals": a.polish_evals, "refine_turns": a.refine_turns,
            "harness_config": cfg,
            "temperature": (a.temperature if a.backend == "openai" else 0.0), "max_new_tokens": cfg["max_new_tokens"],
            "python": platform.python_version(), **{p: (md.version(p) if _has(p) else None) for p in ("torch", "transformers", "peft", "trl", "CoolProp")}}
    # Frozen-protocol provenance: every report must trace back to the exact rows, config and commit.
    meta.update({"eval_set_version": version, "eval_set_hash": version,
                 "split_ids": _scored_ids(report, a.tasks),
                 "harness_config_hash": harness_config_hash(cfg, a.limit),
                 "arm_config": arm_cfg, "git_commit": _git_commit(),
                 "data_dirs": [str(d) for d in a.data]})
    payload = {"meta": meta, "arm_config": arm_cfg, **report}
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1)); out.with_suffix(".completions.jsonl").write_text("\n".join(json.dumps(x) for x in log))
    print(json.dumps(payload, indent=1))


def _has(p):
    import importlib.metadata as md
    try: md.version(p); return True
    except md.PackageNotFoundError: return False


if __name__ == "__main__":
    main()
