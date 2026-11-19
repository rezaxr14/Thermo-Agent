"""Generic Kaggle-hosted runner for every thermo-agent model workflow.

One source file, one notebook, any model, any shard, any mode. It is safe to push
as a Kaggle Benchmarks task: the default model auto-run is skipped cleanly, and
only explicitly requested models are called. `gemini-3.8-flash` and
`gemini-3.5-flash-lite` are hard-blocked.

Modes (set HARNESS_MODE in the task source or in harness_config.json):
  harness   - compact harness + real physics tools, multi-turn CoT tracegen
  eval_h    - harness + tools, evaluation rollouts
  plain     - plain prompt, no tools, no harness (zero-shot)
  eval_z    - plain prompt, no tools (zero-shot evaluation)
  cohort_h  - harness/tools over a tier shard
  cohort_z  - plain/no-tools over a tier shard

Inputs are attached Kaggle datasets (normally the HF-fed `thermo-agent-data`).
`harness_config.json` in any attached dataset can point at any shard, mode, limit,
memory dir and output name without editing this file.
"""
# %% [markdown]
# # thermo-agent generic Kaggle model runner
#
# Reusable for tracegen, evaluation, zero-shot and cohort passes. Push once, then
# run explicitly with `-m gpt-6-luna -m claude-haiku-5-5`; no other model is
# called. Gemini 3.8 / 3.5 are hard-blocked.

# %%
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import re
import sys
import time
import zipfile

OUT = pathlib.Path("/kaggle/working")
IS_KAGGLE = pathlib.Path("/kaggle/input").exists()

TASKS = ("repair", "ledger", "audit", "design")
DEFAULT_ALLOWED = ("gpt-6-luna", "claude-haiku-5-5")
BLOCKED = ("gemini-3.8-flash", "gemini-3.5-flash-lite")
TOOL_MODES = ("harness", "eval_h", "cohort_h")
PLAIN_MODES = ("plain", "eval_z", "cohort_z")
MODES = TOOL_MODES + PLAIN_MODES

DEFAULTS = {
    "mode": "harness",
    "shard": "prompts_proof.jsonl",
    "limit": 0,
    "rounds": 0,          # 0 -> project default (6 for tools, 1 for plain)
    "k_memory": 3,
    "deadline_seconds": 0,
    "max_cost_usd": 0.0,
    "journal": "completions.jsonl",
    "manifest": "manifest.json",
    "memory_dir": "",
    "export_dir": "",
    "allowed_models": list(DEFAULT_ALLOWED),
    "code_dataset": "rezanadimi/thermo-agent-data",
    "data_dataset": "rezanadimi/thermo-agent-data",
    "model_shards": {},
    "config_file": "harness_config.json",
}

_SECRET_PAT = re.compile(
    r"(?i)\b(kaggle[_-]?api[_-]?token|hf[_-]?token|api[_-]?key|authorization|bearer)\b"
    r"\s*[:=]\s*[^\s,;]+|sk-[A-Za-z0-9_\-]{8,}"
)


def redact(value) -> str:
    """Never journal a credential, even when a provider error echoes one."""
    return _SECRET_PAT.sub(
        lambda m: (m.group(1) + "=***") if m.group(1) else "sk-***", str(value))


def is_allowed(model: str, allowed=DEFAULT_ALLOWED, blocked=BLOCKED) -> bool:
    """Budget models by default; blocked models never run, whatever the config says."""
    m = str(model or "").lower()
    if any(b in m for b in blocked):
        return False
    return any(a in m for a in allowed)


def parse_models(value: str | None) -> tuple[str, ...]:
    if not value:
        return DEFAULT_ALLOWED
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return tuple(parts) or DEFAULT_ALLOWED


def task_file(task: str) -> str:
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    return f"{task}_train.jsonl"


class Config:
    """Plain class (no dataclass): importable from a file path without sys.modules tricks."""

    def __init__(self, mode, shard, limit, rounds, k_memory, deadline_seconds,
                 max_cost_usd, journal, manifest, memory_dir, export_dir,
                 allowed_models, code_dataset, data_dataset, model_shards=None):
        self.mode = mode
        self.shard = shard
        self.limit = limit
        self.rounds = rounds
        self.k_memory = k_memory
        self.deadline_seconds = deadline_seconds
        self.max_cost_usd = max_cost_usd
        self.journal = journal
        self.manifest = manifest
        self.memory_dir = memory_dir
        self.export_dir = export_dir
        self.allowed_models = tuple(allowed_models)
        self.code_dataset = code_dataset
        self.data_dataset = data_dataset
        self.model_shards = dict(model_shards or {})

    @property
    def uses_tools(self) -> bool:
        return self.mode in TOOL_MODES

    def shard_for(self, model: str) -> str:
        """Per-model shard keeps one diagram per model (no duplicate questions)."""
        m = str(model or "").lower()
        for key, value in self.model_shards.items():
            if str(key).lower() in m:
                return str(value)
        return self.shard

    @staticmethod
    def from_dict(values: dict) -> "Config":
        mode = str(values.get("mode") or DEFAULTS["mode"]).strip().lower()
        if mode not in MODES:
            raise SystemExit(f"unknown HARNESS_MODE {mode!r}; choose from {MODES}")
        def number(key, cast, default):
            try:
                return cast(values.get(key, default))
            except (TypeError, ValueError):
                return cast(default)
        allowed = values.get("allowed_models", DEFAULT_ALLOWED)
        if isinstance(allowed, str):
            allowed = parse_models(allowed)
        rounds = number("rounds", int, DEFAULTS["rounds"])
        if rounds <= 0:
            rounds = 6 if mode in TOOL_MODES else 1
        return Config(
            mode=mode,
            shard=str(values.get("shard") or DEFAULTS["shard"]),
            limit=number("limit", int, DEFAULTS["limit"]),
            rounds=rounds,
            k_memory=number("k_memory", int, DEFAULTS["k_memory"]),
            deadline_seconds=number("deadline_seconds", int, DEFAULTS["deadline_seconds"]),
            max_cost_usd=number("max_cost_usd", float, DEFAULTS["max_cost_usd"]),
            journal=str(values.get("journal") or DEFAULTS["journal"]),
            manifest=str(values.get("manifest") or DEFAULTS["manifest"]),
            memory_dir=str(values.get("memory_dir") or ""),
            export_dir=str(values.get("export_dir") or ""),
            allowed_models=tuple(str(m) for m in allowed),
            code_dataset=str(values.get("code_dataset") or DEFAULTS["code_dataset"]),
            data_dataset=str(values.get("data_dataset") or DEFAULTS["data_dataset"]),
            model_shards=values.get("model_shards") or {},
        )


def _load_config_file() -> dict:
    """harness_config.json in any attached dataset overrides Defaults; env overrides config."""
    if not IS_KAGGLE:
        return {}
    name = os.environ.get("HARNESS_CONFIG") or DEFAULTS["config_file"]
    for path in sorted(pathlib.Path("/kaggle/input").rglob(name)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                print(f"[config] {path}", flush=True)
                return data
        except Exception as exc:
            print(f"[config] ignoring {path}: {type(exc).__name__}", flush=True)
    return {}


def resolve_config() -> Config:
    values = dict(DEFAULTS)
    values.update(_load_config_file())
    for key in DEFAULTS:
        env = os.environ.get("HARNESS_" + key.upper())
        if env not in (None, ""):
            values[key] = env
    if isinstance(values.get("model_shards"), str):
        try:
            values["model_shards"] = json.loads(values["model_shards"])
        except ValueError:
            raise SystemExit("HARNESS_MODEL_SHARDS must be a JSON object")
    cfg = Config.from_dict(values)
    if cfg.rounds <= 0:
        cfg.rounds = 6 if cfg.uses_tools else 1
    return cfg


CFG = resolve_config()
SPEC = {
    "runner": "kaggle_model_runner",
    "mode": CFG.mode,
    "shard": CFG.shard,
    "dataset": CFG.data_dataset,
    "code_dataset": CFG.code_dataset,
    "limit": CFG.limit,
    "rounds": CFG.rounds,
    "k_memory": CFG.k_memory,
    "allowed_models": list(CFG.allowed_models),
    "model_shards": dict(CFG.model_shards),
}
SPEC_SHA = hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()[:16]
JOURNAL = OUT / CFG.journal
MANIFEST = OUT / CFG.manifest
COUNTS = {"attempted": 0, "answered": 0, "error": 0, "skipped": 0}
MODEL_SLUG = "unknown"
ACTIVE_SHARD = CFG.shard
ERROR = None
ROWS_SOURCE = ""
ROWS_DETAIL = ""
ROWS_TOTAL = 0
MEMORY_NOTES_LOADED = 0
REASONING_LEVEL = None


def _manifest(status: str, stage: str, **kw) -> None:
    MANIFEST.write_text(json.dumps({
        "status": status, "stage": stage, "error": ERROR,
        "runner": SPEC["runner"], "mode": CFG.mode, "harness": "agents+skills" if CFG.uses_tools else "none",
        "model": MODEL_SLUG, "shard": ACTIVE_SHARD, "rows_source": ROWS_SOURCE,
        "rows_detail": ROWS_DETAIL, "rows_total": ROWS_TOTAL,
        "memory_notes_loaded": MEMORY_NOTES_LOADED, "reasoning": REASONING_LEVEL or "backend-default",
        "limits": {"limit": CFG.limit, "rounds": CFG.rounds, "k_memory": CFG.k_memory,
                   "deadline_seconds": CFG.deadline_seconds},
        "counts": COUNTS, "spec_sha": SPEC_SHA, **kw,
    }, indent=1, ensure_ascii=False), encoding="utf-8")


# %% [markdown]
# ## Kaggle-only bootstrap and execution
# The cells below execute only on Kaggle; local import/py_compile is safe.

# %%
if IS_KAGGLE:

    def _bundle_from_input_dir(inp: pathlib.Path):
        if not inp.exists():
            return None
        for marker in inp.rglob("agent_graph.py"):
            if marker.parent.name != "teacher":
                continue
            root = marker.parents[2]
            if (root / "thermo_agent" / "teacher" / "agent_graph.py").exists():
                return root
        return None

    def _bundle_from_zip(inp: pathlib.Path, work: pathlib.Path):
        zips = sorted(inp.rglob("kaggle_bundle.zip")) if inp.exists() else []
        if not zips:
            return None
        work.mkdir(parents=True, exist_ok=True)
        marker = work / "thermo_agent" / "teacher" / "agent_graph.py"
        if not marker.exists():
            dest = work.resolve()
            with zipfile.ZipFile(zips[0]) as zf:
                for info in zf.infolist():
                    target = (work / info.filename).resolve()
                    if target != dest and dest not in target.parents:
                        continue
                    zf.extract(info, work)
        return work if marker.exists() else None

    def _bootstrap() -> pathlib.Path:
        inp = pathlib.Path("/kaggle/input")
        work = OUT / "thermo-agent"
        root = _bundle_from_input_dir(inp) or _bundle_from_zip(inp, work)
        if root is None:
            raise SystemExit(f"thermo-agent code not found; attach -d {CFG.code_dataset}")
        return root

    BUNDLE_ROOT = _bootstrap()
    sys.path.insert(0, str(BUNDLE_ROOT))

    try:
        import CoolProp  # noqa: F401
    except ModuleNotFoundError:
        target = OUT / "pylibs"
        target.mkdir(parents=True, exist_ok=True)
        try:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
                            "--target", str(target), "CoolProp==6.8.0"],
                           check=True, timeout=420, capture_output=True)
            sys.path.insert(0, str(target))
        except Exception as exc:
            print(f"[deps] CoolProp unavailable ({type(exc).__name__})", flush=True)

    import kaggle_benchmarks as kbench
    from kaggle_benchmarks import actors

    from thermo_agent.teacher import memory as mem
    from thermo_agent.teacher.agent_graph import DEFAULT_MAX_ROUNDS, build_context, run_direct
    from thermo_agent.teacher.kbench_adapter import make_respond, reasoning_for_model, usage_of
    from thermo_agent.teacher.toolkit import tool_names

    def _find_shard(name: str) -> pathlib.Path:
        candidates = []
        for base in (pathlib.Path("/kaggle/input"), BUNDLE_ROOT, pathlib.Path(".")):
            if base.exists():
                candidates += [p.parent for p in base.rglob(name)]
        for cand in candidates:
            if (cand / name).exists():
                return cand / name
        raise SystemExit(f"shard {name!r} not found in attached datasets")

    def _find_export_dir() -> pathlib.Path | None:
        if CFG.export_dir:
            p = pathlib.Path(CFG.export_dir)
            return p if p.is_dir() else None
        for base in (pathlib.Path("/kaggle/input"), BUNDLE_ROOT):
            if not base.exists():
                continue
            for marker in base.rglob("repair_train.jsonl"):
                if all((marker.parent / task_file(t)).exists() for t in TASKS):
                    return marker.parent
        return None

    def _load_rows(shard: str) -> tuple[list[dict], str, str]:
        export = _find_export_dir()
        if export is not None:
            rows = {}
            for task in TASKS:
                path = export / task_file(task)
                if not path.is_file():
                    continue
                for line in path.open(encoding="utf-8"):
                    if line.strip():
                        row = json.loads(line)
                        rows[str(row["id"])] = row
            if rows:
                return list(rows.values()), "export", str(export)
        path = _find_shard(shard)
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        return rows, "shard", str(path)

    def _fresh_notes() -> list:
        dirs = []
        if CFG.memory_dir:
            dirs.append(pathlib.Path(CFG.memory_dir))
        dirs += [BUNDLE_ROOT / "memory", BUNDLE_ROOT / "data" / "memory"]
        for base in (pathlib.Path("/kaggle/input"),):
            if base.exists():
                dirs += list(base.rglob("data/memory"))[:5] + list(base.rglob("memory"))[:5]
        notes, seen = [], set()
        for d in dirs:
            if not d.exists():
                continue
            for p in sorted(d.glob("*.jsonl")):
                if p in seen:
                    continue
                seen.add(p)
                for line in p.open(encoding="utf-8"):
                    try:
                        notes.append(mem.Note(**json.loads(line)))
                    except Exception:
                        pass
        return mem.dedupe(notes) if notes else []

    # %%
    @kbench.task(name="thermo-teacher-cotproof-s00",
                 description="Generic thermo-agent model runner (budget models only by default)")
    def thermo_teacher_cotproof_s00(llm):
        global MODEL_SLUG, ACTIVE_SHARD, ERROR, ROWS_SOURCE, ROWS_DETAIL, ROWS_TOTAL
        global MEMORY_NOTES_LOADED, REASONING_LEVEL

        llm = llm or kbench.llm
        MODEL_SLUG = getattr(llm, "name", None) or getattr(llm, "model", None) or "unknown"
        if not is_allowed(MODEL_SLUG, CFG.allowed_models):
            print(f"[skip] model {MODEL_SLUG!r} is not allowed by this task; no model call made", flush=True)
            return
        ACTIVE_SHARD = CFG.shard_for(MODEL_SLUG)
        COUNTS.update(attempted=0, answered=0, error=0, skipped=0)
        ERROR = None
        OUT.mkdir(parents=True, exist_ok=True)
        _manifest("running", "start")

        rows, ROWS_SOURCE, ROWS_DETAIL = _load_rows(ACTIVE_SHARD)
        if CFG.limit > 0:
            rows = rows[:CFG.limit]
        ROWS_TOTAL = len(rows)
        notes = _fresh_notes()
        MEMORY_NOTES_LOADED = len(notes)
        tools = tool_names() if CFG.uses_tools else []
        print(f"[runner] mode={CFG.mode} rows={ROWS_TOTAL} model={MODEL_SLUG} "
              f"tools={','.join(tools) or 'none'} rounds={CFG.rounds} notes={len(notes)}", flush=True)

        done = set()
        if JOURNAL.exists():
            for line in JOURNAL.open(encoding="utf-8"):
                try:
                    done.add(str(json.loads(line)["id"]))
                except Exception:
                    pass

        started, spent = time.time(), 0
        with JOURNAL.open("a", encoding="utf-8") as fh:
            for row in rows:
                rid = str(row.get("id", ""))
                if rid in done:
                    COUNTS["skipped"] += 1
                    continue
                if CFG.deadline_seconds > 0 and time.time() - started > CFG.deadline_seconds:
                    ERROR = "deadline"
                    break
                task = str(row.get("task") or "repair")
                t0 = time.time()
                rec = {
                    "id": rid, "record_id": str(row.get("record_id") or rid),
                    "task": task, "tier": row.get("tier"), "model": MODEL_SLUG,
                    "mode": CFG.mode, "harness": "agents+skills" if CFG.uses_tools else "none",
                    "tools": tools, "completion": "", "sft_completion": "",
                    "reasoning": "", "trajectory": {"turns": [], "notes_used": [], "final": ""},
                    "usage": {}, "error": None, "graded": False,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "shard": ACTIVE_SHARD, "row_source": ROWS_SOURCE,
                    "spec_sha": SPEC_SHA,
                }
                try:
                    if CFG.uses_tools:
                        ctx = build_context({"row": row, "task": task, "notes": notes,
                                             "k_memory": CFG.k_memory, "mode": "agents+skills"})
                        reasoning = reasoning_for_model(MODEL_SLUG)
                        REASONING_LEVEL = reasoning
                        for attempt in (0, 1):
                            holder = {}
                            respond = make_respond(llm, kbench, actors,
                                                   system_prompt=ctx["system_prompt"],
                                                   max_tool_rounds=CFG.rounds,
                                                   usage_holder=holder, reasoning=reasoning)
                            try:
                                state = run_direct(row, task, respond, notes=notes,
                                                   max_rounds=CFG.rounds,
                                                   k_memory=CFG.k_memory, mode="agents+skills")
                                break
                            except Exception as exc:
                                if "reasoning_effort" not in str(exc) or reasoning is not None or attempt:
                                    raise
                                reasoning = "none"   # provider rejects tools + reasoning
                                REASONING_LEVEL = "none"
                        rec["completion"] = state.get("final") or ""
                        rec["sft_completion"] = state.get("sft_completion") or ""
                        rec["reasoning"] = reasoning or "backend-default"
                        rec["trajectory"] = {"turns": state.get("turns") or [],
                                             "notes_used": state.get("notes_used") or [],
                                             "tools_available": tools,
                                             "final": rec["completion"]}
                        rec["usage"] = usage_of(holder["chat"]) if "chat" in holder else {}
                    else:
                        messages = row.get("prompt") if isinstance(row.get("prompt"), list) else []
                        system = "\n\n".join(m["content"] for m in messages
                                             if m.get("role") == "system" and isinstance(m.get("content"), str))
                        user = "\n\n".join(m["content"] for m in messages
                                           if m.get("role") != "system" and isinstance(m.get("content"), str))
                        if not user:
                            user = json.dumps(row, ensure_ascii=False)[:4000]
                        reasoning = reasoning_for_model(MODEL_SLUG)
                        REASONING_LEVEL = reasoning
                        with kbench.chats.new(f"thermo_{rid[:24]}", system_instructions=system or None) as chat:
                            try:
                                answer = llm.prompt(user, reasoning=reasoning)
                            except TypeError:
                                answer = llm.prompt(user)
                        rec["completion"] = str(answer)
                        rec["reasoning"] = reasoning or "backend-default"
                        rec["trajectory"] = {"turns": [], "notes_used": [], "final": rec["completion"]}
                        rec["usage"] = usage_of(chat)
                    rec["usage"]["seconds"] = round(time.time() - t0, 2)
                    COUNTS["answered"] += 1
                except Exception as exc:
                    rec["error"] = redact(f"{type(exc).__name__}: {exc}")[:500]
                    COUNTS["error"] += 1
                    ERROR = rec["error"]
                COUNTS["attempted"] += 1
                fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
                fh.flush()
                done.add(rid)
                spent += int((rec.get("usage") or {}).get("total_cost_nanodollars") or 0)
                if CFG.max_cost_usd > 0 and spent / 1e9 >= CFG.max_cost_usd:
                    ERROR = "cost_cap"
                    break
        _manifest("completed" if ERROR is None else "partial", "done")

    # %%
    thermo_teacher_cotproof_s00.run(kbench.llm)
