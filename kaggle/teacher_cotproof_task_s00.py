"""Hosted LangGraph + REAL-tools teacher pass over one compact prompt shard.

Self-contained: bootstraps the thermo-agent source bundle using the STANDARD LIBRARY
ONLY, then reuses our own agent_graph / kbench_adapter so the hosted model sees
AGENTS.md + component skills + compact failure memory + the real tool contract.

No secrets, no ground truth. One JSON line is appended (and flushed) per row, so a
dead worker loses at most one answer; re-running skips ids already in the journal.
The model id on every record is the REAL ``kbench.llm.name`` -- never an invented one.
"""
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

SPEC = {
    "task_name": "thermo-teacher-cotproof-s00",
    "dataset": "rezanadimi/thermo-agent-data",
    "code_dataset": "rezanadimi/thermo-agent-data",
    "shard": "prompts_proof.jsonl",
    "limit": 5,           # 0 = every row in the shard (no assistant-invented row cap)
    "max_cost_usd": 0.0,  # 0 = no internal cost stop (account-level quotas still apply)
    # "rounds" is filled from agent_graph.DEFAULT_MAX_ROUNDS after the bootstrap import,
    # so the hosted prompt, adapter and graph never disagree on the round budget.
    "k_memory": 3,
    "deadline_seconds": 0,  # 0 = no internal wall-clock stop (the Kaggle session limit still applies)
}

OUT = pathlib.Path("/kaggle/working")
JOURNAL = OUT / "harness_completions.jsonl"
MANIFEST = OUT / "manifest.json"

# Hosted slugs we expect to see. This is ONLY a warning guard: an unrecognised name
# is still recorded verbatim from kbench.llm, never replaced with a guessed slug.
EXPECTED_MODELS = (
    "gpt-6-luna", "claude-haiku-5-5", "gemini-3.8-flash",
    "gpt-6.2-sol", "claude-sonnet-5-5", "claude-opus-5-5", "gemini-4-argon",
)

TASKS = ("repair", "ledger", "audit", "design")

# A verdict is NEVER fabricated on the hosted side. Compact shard rows have no plant/spec
# columns, so their only acceptance source is the local transfer.ingest re-grade by id.
DEFERRED_VERDICT = {"accepted": None, "graded": False, "reason": "deferred_to_ingest"}
NOT_RUN_VERDICT = {"accepted": None, "graded": False, "reason": "not_run"}

# Columns each in-task validator needs (mirrors thermo_agent.teacher.validate).
GRADABLE_KEYS = {
    "repair": ("plant_broken", "plant_original"),
    "ledger": ("ledger",),
    "audit": ("plant", "ed_json", "fuel_w", "net_w"),
    "design": ("site",),
}


# --------------------------------------------------------------------------- #
# bootstrap (standard library only)
# --------------------------------------------------------------------------- #
def _bundle_from_input_dir(inp: pathlib.Path):
    """An attached code dataset may already contain the extracted source tree."""
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
    """Otherwise unpack kaggle_bundle.zip from the attached code dataset."""
    if not inp.exists():
        return None
    zips = sorted(inp.rglob("kaggle_bundle.zip"))
    if not zips:
        return None
    work.mkdir(parents=True, exist_ok=True)
    marker = work / "thermo_agent" / "teacher" / "agent_graph.py"
    if not marker.exists():
        dest = work.resolve()
        with zipfile.ZipFile(zips[0]) as zf:
            for info in zf.infolist():
                target = (work / info.filename).resolve()
                # zip-slip guard: never write outside the extraction root.
                if target != dest and dest not in target.parents:
                    continue
                zf.extract(info, work)
    return work if marker.exists() else None


def _bootstrap() -> pathlib.Path:
    inp = pathlib.Path("/kaggle/input")
    work = OUT / "thermo-agent"
    root = _bundle_from_input_dir(inp)
    if root is None:
        root = _bundle_from_zip(inp, work)
    if root is None:
        # Local development / py_compile convenience: the repo is one level up.
        local = pathlib.Path(__file__).resolve().parents[1]
        if (local / "thermo_agent" / "teacher" / "agent_graph.py").exists():
            root = local
    if root is None:
        raise SystemExit(
            "thermo-agent source not found: attach -d %s (or set up /kaggle/working/thermo-agent)"
            % SPEC["code_dataset"]
        )
    return root


BUNDLE_ROOT = _bootstrap()
sys.path.insert(0, str(BUNDLE_ROOT))


def _ensure_optional_deps() -> None:
    """The benchmarks venv is minimal (no scipy, sometimes no CoolProp).

    scipy is handled by thermo_agent.core._roots (pure-Python Brent fallback), so the
    graph imports without it. CoolProp cannot be reimplemented: install the wheel into a
    writable directory when the network allows, and let gas-only plants run when it does
    not. Never fail the whole task because a real-fluid backend is missing.
    """
    try:
        import CoolProp  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    target = OUT / "pylibs"
    target.mkdir(parents=True, exist_ok=True)
    for cmd in ([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
                 "--target", str(target), "CoolProp==6.8.0"],):
        try:
            import subprocess
            subprocess.run(cmd, check=True, timeout=420, capture_output=True)
            sys.path.insert(0, str(target))
            import CoolProp  # noqa: F401
            print("[deps] CoolProp installed into", target, flush=True)
            return
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[deps] CoolProp unavailable ({type(exc).__name__}); gas-only plants will run", flush=True)


_ensure_optional_deps()

# Repo imports come AFTER the stdlib-only bootstrap.
import kaggle_benchmarks as kbench  # noqa: E402
from kaggle_benchmarks import actors  # noqa: E402

from thermo_agent.teacher import memory as mem  # noqa: E402
from thermo_agent.teacher.agent_graph import (  # noqa: E402
    DEFAULT_MAX_ROUNDS, build_context, run_direct,
)
from thermo_agent.teacher.kbench_adapter import make_respond, reasoning_for_model, usage_of  # noqa: E402
from thermo_agent.teacher.toolkit import tool_names  # noqa: E402

SPEC["rounds"] = DEFAULT_MAX_ROUNDS
SPEC_SHA = hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()[:16]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


SHARD = os.environ.get("HARNESS_SHARD") or SPEC["shard"]
LIMIT = _env_int("HARNESS_LIMIT", SPEC["limit"])
# Default to the project constant, not a local copy of the number.
ROUNDS = _env_int("HARNESS_ROUNDS", DEFAULT_MAX_ROUNDS)
K_MEMORY = _env_int("HARNESS_K_MEMORY", SPEC["k_memory"])
DEADLINE = _env_int("HARNESS_DEADLINE", SPEC["deadline_seconds"])
MAX_COST_USD = float(os.environ.get("HARNESS_MAX_COST_USD", SPEC["max_cost_usd"]))

# Never journal a credential, even if a provider error happens to echo one.
_SECRET_PAT = re.compile(
    r"(?i)(kaggle[_-]?api[_-]?token|hf[_-]?token|api[_-]?key|authorization|bearer\s+\S+|sk-[A-Za-z0-9_\-]{8,})"
)


def _redact(value) -> str:
    return _SECRET_PAT.sub(lambda m: m.group(1).split()[0] + "=***", str(value))


# --------------------------------------------------------------------------- #
# row loading
# --------------------------------------------------------------------------- #
def _find_shard(root: pathlib.Path) -> pathlib.Path:
    candidates = []
    for base in (pathlib.Path("/kaggle/input"), root, pathlib.Path(".")):
        if base.exists():
            candidates += [p.parent for p in base.rglob(SHARD)]
    candidates.append(root / "data" / "kaggle_datasets" / "teacher_big_prompts")
    seen = set()
    for cand in candidates:
        key = str(cand)
        if key in seen or not (cand / SHARD).exists():
            continue
        seen.add(key)
        return cand / SHARD
    raise SystemExit(
        "prompt shard %s not found; attach -d %s or set HARNESS_SHARD"
        % (SHARD, SPEC["dataset"])
    )


def _find_export_dir(root: pathlib.Path):
    """Full export rows (prompt + plant/spec columns), if any dataset carries them."""
    for base in (pathlib.Path("/kaggle/input"), root):
        if not base.exists():
            continue
        for marker in base.rglob("repair_train.jsonl"):
            if all((marker.parent / "%s_train.jsonl" % t).exists() for t in TASKS):
                return marker.parent
    local = root / "data" / "frontier_export"
    if (local / "repair_train.jsonl").exists():
        return local
    return None


def _load_exports_callable(root: pathlib.Path):
    """Prefer the driver's own load_exports; fall back to the transfer function it imports."""
    cli = root / "scripts" / "kaggle_harness_cli.py"
    if cli.exists():
        try:
            spec = importlib.util.spec_from_file_location("_harness_cli_for_exports", cli)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if callable(getattr(module, "load_exports", None)):
                return module.load_exports
        except Exception:
            pass
    try:
        from thermo_agent.teacher.transfer import load_exports
        return load_exports
    except Exception:
        return None


def _load_exports_replica(export_dir, tasks=TASKS, split: str = "train") -> dict:
    """Exact replicate of transfer.load_exports (id-keyed, one row per id)."""
    rows = {}
    for task in tasks:
        path = pathlib.Path(export_dir) / ("%s_%s.jsonl" % (task, split))
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def _adapt_row(row: dict) -> dict:
    """Give compact shard rows the shape agent_graph expects, keeping every source column."""
    out = dict(row)
    if not isinstance(out.get("prompt"), list):
        messages = []
        if out.get("system"):
            messages.append({"role": "system", "content": out["system"]})
        if out.get("user"):
            messages.append({"role": "user", "content": out["user"]})
        out["prompt"] = messages
    out.setdefault("task", "repair")
    out.setdefault("split", "train")
    return out


def _load_rows(root: pathlib.Path):
    export_dir = _find_export_dir(root)
    if export_dir is not None:
        loader = _load_exports_callable(root) or _load_exports_replica
        try:
            rows = list(loader(export_dir, split="train").values())
        except Exception:
            rows = []
        if rows:
            return [_adapt_row(r) for r in rows], "load_exports", str(export_dir)
    path = _find_shard(root)
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return [_adapt_row(r) for r in rows], "shard", str(path)


def _memory_dir_candidates(root: pathlib.Path):
    """Every location retrieval memory could live, in priority order (existing or not)."""
    out = []
    if os.environ.get("HARNESS_MEMORY_DIR"):
        out.append(pathlib.Path(os.environ["HARNESS_MEMORY_DIR"]))
    # The attached code bundle is built with Compress-Archive, which stripped the `data`
    # parent: the notes land at BUNDLE_ROOT/memory. Check that FIRST, then keep the legacy
    # data/memory locations and the /kaggle/input rglob fallbacks below.
    out += [root / "memory", root / "data" / "memory", OUT / "thermo-agent" / "data" / "memory"]
    inp = pathlib.Path("/kaggle/input")
    if inp.exists():
        out += list(inp.rglob("data/memory"))[:5]
        out += list(inp.rglob("memory"))[:5]
    return out


def _memory_dirs(root: pathlib.Path):
    return [d for d in _memory_dir_candidates(root) if d.exists()]


def _fresh_notes(root: pathlib.Path):
    notes, seen = [], set()
    searched = _memory_dirs(root)
    for directory in searched:
        for path in sorted(directory.glob("*.jsonl")):
            if path in seen:
                continue
            seen.add(path)
            for line in path.open(encoding="utf-8"):
                try:
                    notes.append(mem.Note(**json.loads(line)))
                except Exception:
                    pass
    try:
        notes = mem.dedupe(notes)
    except Exception:
        pass
    if not notes:
        # Never hide a silent loss of all retrieval memory: the manifest records the count.
        print("[warn] no retrieval memory found; searched: %s"
              % ", ".join(str(d) for d in _memory_dir_candidates(root)), flush=True)
    return notes


def _row_is_gradable(task: str, row: dict) -> bool:
    """True only when the row carries the plant/spec columns the real validator reads."""
    keys = GRADABLE_KEYS.get(task)
    if not keys:
        return False
    return all(row.get(k) not in (None, "", {}, []) for k in keys)


def _install_safe_validator():
    """Let finalize finish on compact rows WITHOUT claiming any acceptance.

    On a missing-column failure the wrapper returns ``DEFERRED_VERDICT`` (accepted=None,
    graded=False). It never returns accepted=True or accepted=False: local
    ``transfer.ingest`` re-grades against the full export by id and is the ONLY source
    of acceptance for those rows. Full rows are unaffected -- the real validator runs
    first and its verdict is used as-is.
    """
    import thermo_agent.teacher.validate as V

    if getattr(V.validate, "_harness_safe", False):
        return
    original = V.validate

    def safe(task, completion, row):
        try:
            return original(task, completion, row)
        except (KeyError, ValueError, TypeError, IndexError):
            return dict(DEFERRED_VERDICT)

    safe._harness_safe = True
    V.validate = safe


# --------------------------------------------------------------------------- #
# journal / manifest
# --------------------------------------------------------------------------- #
COUNTS = {"attempted": 0, "answered": 0, "error": 0, "skipped": 0,
          "graded": 0, "deferred": 0}
MODEL_SLUG = "unknown"
MODEL_OK = False
ERROR = None
ROWS_SOURCE = ""
ROWS_DETAIL = ""
ROWS_TOTAL = 0
MEMORY_NOTES_LOADED = 0  # set from _fresh_notes so a silent zero-note loss is recorded
REASONING_LEVEL = None  # "none" when the model rejects tools + reasoning_effort


def _manifest(status: str, stage: str, **kw):
    MANIFEST.write_text(json.dumps({
        "status": status, "stage": stage, "error": ERROR,
        "task": SPEC["task_name"], "harness": "tools",
        "model": MODEL_SLUG, "model_recognized": MODEL_OK,
        "shard": SHARD, "rows_source": ROWS_SOURCE,
        "rows_detail": ROWS_DETAIL, "rows_total": ROWS_TOTAL,
        "memory_notes_loaded": MEMORY_NOTES_LOADED,
        "reasoning": REASONING_LEVEL or "backend-default",
        "limits": {"limit": LIMIT, "rounds": ROUNDS, "k_memory": K_MEMORY,
                   "deadline_seconds": DEADLINE},
        "counts": COUNTS, "tools": tool_names(), "spec_sha": SPEC_SHA, **kw,
    }, indent=1, ensure_ascii=False), encoding="utf-8")


@kbench.task(name=SPEC["task_name"],
             description="Thermo teacher: LangGraph + real physics tools over a prompt shard")
def thermo_teacher_cotproof_s00(llm):
    global MODEL_SLUG, MODEL_OK, ERROR, ROWS_SOURCE, ROWS_DETAIL, ROWS_TOTAL
    global MEMORY_NOTES_LOADED, REASONING_LEVEL

    llm = llm or kbench.llm
    MODEL_SLUG = getattr(llm, "name", None) or getattr(llm, "model", None) or "unknown"
    MODEL_OK = any(s in str(MODEL_SLUG).lower() for s in EXPECTED_MODELS)
    if not MODEL_OK:
        print("[warn] model %r not in the expected hosted list; recording its real name"
              % MODEL_SLUG, flush=True)

    COUNTS.update(attempted=0, answered=0, error=0, skipped=0, graded=0, deferred=0)
    ERROR = None
    OUT.mkdir(parents=True, exist_ok=True)
    _manifest("running", "start")

    rows, ROWS_SOURCE, ROWS_DETAIL = _load_rows(BUNDLE_ROOT)
    if LIMIT > 0:
        rows = rows[:LIMIT]
    ROWS_TOTAL = len(rows)
    _install_safe_validator()
    notes = _fresh_notes(BUNDLE_ROOT)
    MEMORY_NOTES_LOADED = len(notes)
    REASONING = reasoning_for_model(MODEL_SLUG)
    REASONING_LEVEL = REASONING
    print("[harness] rows=%d source=%s model=%s tools=%s rounds=%d notes=%d reasoning=%s"
          % (ROWS_TOTAL, ROWS_SOURCE, MODEL_SLUG, ",".join(tool_names()), ROUNDS,
             len(notes), REASONING or "backend-default"), flush=True)

    done = set()
    if JOURNAL.exists():
        for line in JOURNAL.open(encoding="utf-8"):
            try:
                done.add(str(json.loads(line)["id"]))
            except Exception:
                pass

    started, written, spent_nanodollars = time.time(), 0, 0
    with JOURNAL.open("a", encoding="utf-8") as fh:
        for index, row in enumerate(rows):
            rid = str(row.get("id", ""))
            if rid in done:
                COUNTS["skipped"] += 1
                continue
            if DEADLINE > 0 and time.time() - started > DEADLINE:
                ERROR = "deadline"
                break
            task = row.get("task") or "repair"
            gradable = _row_is_gradable(task, row)
            t0 = time.time()
            rec = {
                "id": rid, "task": task, "model": MODEL_SLUG, "harness": "tools",
                "tools": tool_names(), "completion": "", "sft_completion": "",
                "verdict": dict(DEFERRED_VERDICT) if not gradable else dict(NOT_RUN_VERDICT),
                "graded": False,
                "trajectory": {"turns": [], "notes_used": [],
                               "tools_available": tool_names(), "final": ""},
                "usage": {}, "error": None,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "shard": SHARD, "row_source": ROWS_SOURCE, "spec_sha": SPEC_SHA,
            }
            try:
                ctx = build_context({"row": row, "task": task, "notes": notes,
                                     "k_memory": K_MEMORY, "mode": "agents+skills"})
                # gpt-6 chat-completions rejects tools + reasoning_effort. We know the slug
                # up front, but keep a one-shot retry so a new model slug cannot kill a run.
                for attempt in (0, 1):
                    usage_holder = {}
                    respond = make_respond(
                        llm, kbench, actors,
                        system_prompt=ctx["system_prompt"],
                        max_tool_rounds=ROUNDS, usage_holder=usage_holder,
                        reasoning=REASONING,
                    )
                    try:
                        state = run_direct(row, task, respond, notes=notes, max_rounds=ROUNDS,
                                           k_memory=K_MEMORY, mode="agents+skills")
                        break
                    except Exception as exc:
                        retryable = ("reasoning_effort" in str(exc) and REASONING is None
                                     and attempt == 0)
                        if not retryable:
                            raise
                        REASONING = "none"
                        REASONING_LEVEL = "none"
                        print("[harness] tools+reasoning rejected by the API; retrying %s "
                              "with reasoning=none" % rid, flush=True)
                final = state.get("final") or ""
                rec["reasoning"] = REASONING or "backend-default"
                rec["completion"] = final
                rec["sft_completion"] = state.get("sft_completion") or ""
                verdict = {k: v for k, v in (state.get("verdict") or {}).items()
                           if k != "failed"}
                if not gradable:
                    # Compact shard: transfer.ingest is the ONLY grading authority.
                    verdict = dict(DEFERRED_VERDICT)
                elif not verdict:
                    verdict = dict(NOT_RUN_VERDICT)
                rec["verdict"] = verdict
                rec["graded"] = bool(
                    gradable and verdict.get("accepted") is not None
                    and verdict.get("reason") != "deferred_to_ingest"
                )
                rec["trajectory"] = {
                    "turns": state.get("turns") or [],
                    "notes_used": state.get("notes_used") or [],
                    "tools_available": tool_names(),
                    "final": final,
                }
                rec["usage"] = usage_of(usage_holder["chat"]) if "chat" in usage_holder else {}
                rec["usage"]["seconds"] = round(time.time() - t0, 2)
                COUNTS["answered"] += 1
            except Exception as exc:
                message = "%s: %s" % (type(exc).__name__, exc)
                rec["error"] = _redact(message)[:500]
                COUNTS["error"] += 1
                ERROR = rec["error"]
            COUNTS["graded" if rec["graded"] else "deferred"] += 1
            COUNTS["attempted"] += 1
            fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
            fh.flush()
            done.add(rid)
            written += 1
            spent_nanodollars += int((rec.get("usage") or {}).get("total_cost_nanodollars") or 0)
            if written % 10 == 0:
                _manifest("running", "rows", next_index=index + 1,
                          spent_usd=round(spent_nanodollars / 1e9, 6))
            if MAX_COST_USD > 0 and spent_nanodollars / 1e9 >= MAX_COST_USD:
                ERROR = "cost_cap"
                _manifest("paused", "cost_cap", spent_usd=round(spent_nanodollars / 1e9, 6))
                break
            if COUNTS["error"] >= 5 and COUNTS["answered"] == 0:
                _manifest("paused", "too_many_errors")
                break

    _manifest("completed" if ERROR is None else "partial", "done")


thermo_teacher_cotproof_s00.run(kbench.llm)
