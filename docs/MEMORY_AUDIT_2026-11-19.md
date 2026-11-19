# RAG success/failure memory audit — 2026-11-19

**Scope.** Read-only audit of `thermo_agent/teacher/memory.py`, `thermo_agent/teacher/compact_memory.py`,
`data/memory/*.jsonl`, `data/frontier/frontier_*_train.jsonl` and `docs/CORPUS_INVENTORY_2026-11-19.md`,
followed by one additive write: a **new** file `data/memory/agent_success_patterns.jsonl` (7 notes). No
existing memory note, corpus, model, config or doc was deleted or overwritten. All counts below were measured
by loading each JSONL through `thermo_agent.teacher.memory.load` with `.venv/Scripts/python.exe`.

**Key result.** Before this audit the memory contained **zero verifier-accepted success notes**. Every one of
the 6 `accepted=true` notes was a hand-written `seed:method` note — guidance, *not* verifier evidence. The
635 verifier-accepted frontier rows had no corresponding "what worked" notes. I added 7 compact,
verifier-grounded success notes (capped well below the 12-note limit); audit remains uncovered because **no
accepted audit frontier row exists**.

---

## 1. Counts by source / task / fault type / acceptance

### 1a. Pre-audit ledger — 407 notes across 3 files

| file | notes | accepted | rejected |
|---|---:|---:|---:|
| `frontier_notes.jsonl` | 369 | 0 | 369 |
| `adversarial_repair.jsonl` | 24 | 0 | 24 |
| `agent_patterns.jsonl` | 14 | 6 | 8 |
| **TOTAL (pre-audit)** | **407** | **6** | **401** |

### 1b. By `source` (pre-audit)

| source | notes | evidence class | accepted |
|---|---:|---|---:|
| `frontier:google-gemini-3-1-flash-lite-preview` | 176 | verifier-mined rejections | 0 |
| `frontier:openai-gpt-5-4-nano-2026-03-17` | 141 | verifier-mined rejections | 0 |
| `frontier:google-gemini-3-8-flash` | 52 | verifier-mined rejections | 0 |
| `adversarial:echo_broken` | 13 | verifier-mined (verifier reason codes) | 0 |
| `adversarial:wrong_scale` | 8 | verifier-mined (verifier reason codes) | 0 |
| `adversarial:no_edits` | 3 | verifier-mined (verifier reason codes) | 0 |
| `seed:method` | **14** | **seed method notes — NOT verifier evidence** | 6 |

- Verifier-mined (frontier + adversarial): **393 notes, 0 accepted**.
- **Seed notes: 14 (all `source="seed:method"`, all `extra.seed=true`). Of these, 6 carry `accepted=true`,
  and those 6 are NOT verifier evidence** — they are documented method guidance ("verify before answer",
  "iterate on flag", …), not a verifier-accepted row. `_is_seed()` detects all 14 via both the flag and the
  `seed:` prefix, and `retrieve()` deliberately caps them to one slot per served block so they cannot crowd
  out mined evidence.

### 1c. By task × acceptance (pre-audit)

| task | rejected | accepted | accepted source |
|---|---:|---:|---|
| repair | 70 | 2 | both `seed:method` |
| design | 180 | 2 | both `seed:method` |
| ledger | 54 | 1 | `seed:method` |
| audit | 97 | 1 | `seed:method` |

### 1d. By fault type × acceptance (pre-audit)

Top classes (full distribution measured): empty fault type 273 (all rejected — most design/audit/ledger
frontier notes were never fault-classified), `unit_slip` 41, `wrong_fluid` 21, `efficiency_gt_1` 13,
`missing_m` 7, `sign_T0` 6, `wrong_factor` 6, `dropped_term` 5, `missing_ech` 5, `sign_flip` 5,
`t0_literal` 5, `t0_celsius` 4, `t0_offset` 3, plus 6 singleton failure classes.

**Every accepted fault type was a seed-only class**: `minimal_edit`, `verify_before_answer`,
`rank_by_evidence`, `closure_first`, `iterate_on_flag`, `closure_check` (one note each). No accepted note
had a verifier fault class such as `wrong_fluid` or `efficiency_gt_1`.

### 1e. Frontier train rows (cross-check with the corpus inventory)

668 rows: 635 accepted (`reward == 1.0`), 33 rejected, **0 containing tool calls** — matches
`docs/CORPUS_INVENTORY_2026-11-19.md` §3. Accepted reasons are 4 generic strings: "restored a valid plant
exactly" (189), "ledger repaired" (160), "restored a valid plant" (146), "design accepted" (140). The 33
rejected rows are the audit rows, all with fractional reward (0.53–0.99) and a literal reason string
`"audit accepted"`; their detailed failure reasons (nDCG, prediction error, grounding %) live in the mined
notes, not in `validation.reasons`.

---

## 2. Served-memory size and leakage

Method: `memory.retrieve(q, notes, k=3, task=task)` → `compact_memory.compact_notes(top, k=3)` on the **final**
pool (414 notes, including the 7 new ones). Production (`teacher/agent_graph.py` lines 105–110) calls exactly
this pair with `task=task`. 5 representative queries, one per task plus a generic-units query. "Tokens" is a
regex word-token proxy `len([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)` — **no real tokenizer was run**, so treat it as
an approximation, not a model-specific count.

| # | task | chars | approx tokens |
|---|---|---:|---:|
| 1 | repair (fluid mismatch / efficiency / temperature cross) | 538 | 80 |
| 2 | design (gas turbine + absorption + dry rejection) | 892 | 102 |
| 3 | ledger (negative destruction / closure / T0 / 1000×) | 522 | 80 |
| 4 | audit (rank / change / prediction / grounding) | 552 | 82 |
| 5 | repair (unit slip / sign / second-law limits) | 439 | 61 |
| | **p50** | **538** | **80** |
| | **p95** | **892** | **102** |

(Unfiltered `retrieve(q, notes, k=3)` over the same queries gave chars 378–616, p50 586 / p95 616; tokens
56–93, p50 88 / p95 93. Either way a served block is well under ~1 KB / ~100 tokens.)

**Leakage scan** over all five served blocks — patterns for model slugs (`google-`, `openai-`, `gemini`,
`gpt-`, `claude`, `haiku`, `luna`), ISO timestamps, `cost`, `tokens=`, `latency`, `record_id` /
`rep-|des-|led-|aud-|arch-\d` / 8-hex digests, `run_id`, `ts=`, `seed:`, and `$`: **all clean**. No note id
leaked into any block text. `compact_memory._clean()` plus the fact that it only emits `fault_type`, `fix`
and `lesson` means `source`, `record_id`, `id`, `architecture`, `symptoms` and `extra` never reach the model.

**Full served block — example (query #1, repair):**

```
FAILURE MEMORY (what failed -> what works; OK=accepted, FAIL=rejected; classes only, not this row's answer):
- FAIL wrong_fluid: Attempt rejected (edit: unknown fluid FuelGas). Copying the broken label back, or inventing a third fluid, does not work; the destination component fixes the fluid class.
- OK   wrong_fluid: FLUID @s5 = FLUE
- FAIL wrong_fluid: Seen 24 times: Attempt rejected (verifier: invalid plant). Copying the broken label back, or inventing a third fluid, does not work; the destination component fixes the fluid class.
```

Note the `retrieve()` guarantees hold here: one seed note maximum, at least one FAIL slot reserved, and the
new `OK` note sits beside the failure that teaches the same class.

---

## 3. Success coverage

**Were there accepted-success notes?** Effectively no. All 6 pre-audit `accepted=true` notes were
`seed:method` (stated explicitly in §1b); **0 were verifier-mined**. So the harness could only ever show
generic method guidance as "OK", never an actual accepted frontier answer.

**Which tasks had none?** In verifier terms, *all four* had none: repair, design, ledger and audit each had
only seed "OK" notes. After the addition, repair (3), ledger (3) and design (1) have verifier-grounded OK
notes; **audit still has none, and cannot get one from this data** — there are 0 accepted audit frontier rows
(§1e).

**Do accepted frontier rows yield reusable success patterns?** Yes, but with a caveat. The acceptance
*reasons* are too generic on their own ("design accepted", "ledger repaired", "restored a valid plant");
they only say "this passed". However, each accepted row's **completion** contains a concrete, reusable edit
pattern that the verifier accepted, and `validate.success_note()` lifts exactly that into `fix` while keeping
the class-level statement in `lesson`. Measured distinct reusable patterns in the accepted pool:

- **repair (335 accepted):** `FLUID @stream = <class>` for fluid mismatches (110 rows use `FLUID`);
  `SET <comp>.eta_s = <fraction>` to bring an efficiency back into (0,1]; `SET DC.T_return = <above supply>`
  to clear a temperature cross. All 335 accepted repairs contain an `<edits>` block.
- **ledger (160 accepted):** `FIX T0 = <AMBIENT T0_K>` (wrong 273.15 literal), `FIX ED[DC] = E[chw_sup] -
  E[chw_ret] - W[DC]` (remove a spurious 1000× factor), `FIX ED[ENG] = E[air]+E[fuel]-E[s5]-W[ENG]`
  (missing outlet stream / closure). All 160 contain a `<fix>` block.
- **design (140 accepted):** gas-turbine prime mover + absorption cooling + dry rejection, turbine PR≈18 /
  T4≈1500 K, hot-water ≈363 K, all streams inside fluid limits, no grid import; only numbers quoted from the
  input. All 140 contain a `<rationale>` block.

So the patterns are **not too few** (335/160/140 accepted examples), but the *reason strings* are too generic
to stand alone — the reusable value is in the accepted completion. All 635 accepted rows are zero-shot
(`harness=none`, 0 tool calls), so they are success *answers*, not tool traces (consistent with the corpus
inventory §3d).

---

## 4. Duplication / conflicts / grounding

- **Exact duplicate ids:** 4 (`freq-audit-19493645` appears 3× — "Seen 344/36/57 times"; `freq-audit-f43891b0`,
  `freq-design-bab6218a`, `freq-ledger-f62c78a3` once each). These frequency notes were generated from
  different model sources but collide on an md5-of-lesson id. They are not content-identical (counts differ),
  but the id collision is a real ledger hygiene issue.
- **Exact duplicate content** (same task/fault/lesson/fix/accepted): **0 groups** — `dedupe()` keys are
  effectively unique.
- **"Contradictory fix" candidates:** one group shares the exact lesson "Attempt accepted (closure residual
  ~0, ledger repaired)" with two different fixes — `FIX T0 = 289.715…; FIX ED[DC] = …` vs just
  `FIX ED[DC] = …`. This is **not** a contradiction: one accepted repair needed both edits, the other only
  the ED fix (different plants). Similarly, 5 `(task, fault, param)` keys have multiple numeric targets
  (e.g. `SET GT.eta_s` = 0.843 vs 0.902) — these are **plant-specific values for different records**, not
  class-level conflicts, and `retrieve()` excludes a note from its own `record_id`.
- **Not grounded in a verifier:** the 6 accepted `seed:method` notes (no `record_id`, `extra.seed=true`) are
  method guidance, not verifier evidence — the single most important caveat in this audit. Separately, 24
  `frontier:` notes have an empty `record_id`: these are the `freq-*` aggregates whose "Seen N times"
  lessons are derived from verifier rejection counts but are not individually traceable to one record. They
  are verifier-derived statistics, not fabricated lessons, but should not be cited as a single verifier
  verdict.
- The 7 new notes are all grounded: each cites a real accepted row (`record_id`), a verifier reason
  (`symptoms`), and a `fix` lifted verbatim from that row's accepted completion (verified programmatically —
  `fix_from_completion=True` for all 7).

---

## 5. Recommendation

**Add now (done):** 7 compact success notes in the new `data/memory/agent_success_patterns.jsonl`, using the
canonical `validate.success_note()` format with `source="frontier:<model>"`, `record_id` and
`accepted=true`. They cover the three tasks that have accepted frontier rows and both frontier models:

| id (abbreviated) | task / source model | one-line lesson |
|---|---|---|
| `repair-rep-70-00009-…-ok-71b7256f` | repair / gemini-3-1-flash-lite | accepted repair fixes the fluid class on the destination port: `FLUID @s5 = FLUE` |
| `repair-rep-19-00067-…-ok-b98f3284` | repair / gemini-3-1-flash-lite | accepted repair rewrites an efficiency as a fraction (0,1]: `SET AC.eta_s = 0.848…` |
| `repair-rep-33-00085-…-ok-77f6f58d` | repair / gemini-3-1-flash-lite | accepted repair raises a below-ambient return above supply: `SET DC.T_return = 302.16…` |
| `ledger-led-0-04143-…-ok-9dac5ab2` | ledger / gemini-3-1-flash-lite | accepted ledger fixes both T0 and the DC exergy line to close the ledger |
| `ledger-led-0-05413-…-ok-2972682a` | ledger / gpt-5-4-nano | accepted ledger removes the spurious 1000× factor from `ED[DC]` |
| `ledger-led-0-06893-…-ok-57f57665` | ledger / gemini-3-1-flash-lite | accepted ledger adds the missing outlet stream `E[s5]` to `ED[ENG]` |
| `design-des-0-02514-…-ok-addb582a` | design / gpt-5-4-nano | accepted design: turbine + absorption + dry rejection, all numbers from the input |

**Why this is capped and safe.** 7 is ≤12 and is outnumbered ~57:1 by the 401 verifier-mined failure notes.
`retrieve()` already (a) caps seed notes at one slot, (b) reserves at least one FAIL slot for `k≥2`, and
(c) excludes notes from the same record. In a k=3 block an accepted note can occupy at most 2 of 3 slots,
and the measured served blocks above still contain FAIL lessons. Do **not** add more accepted notes: the
remaining 628 accepted rows mostly repeat these same 7 patterns with different plant-specific numbers, which
would inflate the pool without new class-level information and would risk crowding out failures.

**Refused to add, and why:**
- **No accepted audit success note.** There are 0 accepted audit frontier rows (all 33 audit rows have
  reward < 1.0), so there is no verifier acceptance to ground it in. The 52 `google-gemini-3-8-flash` audit
  notes are all rejections. Fabricating an "OK audit" note would violate the grounding requirement. The real
  fix is to obtain ≥1 accepted audit row (as the corpus inventory §5 notes) before mining an audit success.
- **No tool-trace / "how to call tools" success note.** All 635 accepted rows are zero-shot with 0 tool tags
  (§1e), so nothing verifier-grounded can be said about successful tool use.
- **No notes for the other 628 accepted rows.** Their reason strings are the same 4 generic sentences and
  their edits repeat the 7 patterns already added; adding them would be duplicate-by-lesson crowding, not new
  evidence.

---

## Validation performed

- Each new line loads through `thermo_agent.teacher.memory.load` as a `Note` with `accepted=True`, a non-empty
  `record_id`, and `source.startswith("frontier:")` — 7/7 pass.
- Each renders through `compact_memory.compact_notes` with an `OK` marker; no bookkeeping, model id, timestamp,
  cost, record id, run id, seed slug or `$` in served text (§2).
- Every new `fix` is a verbatim substring of its source row's accepted completion (checked programmatically).
- `pytest tests -k memory -q`: **34 passed, 624 deselected in 48.03s**.
- Existing files were not modified; the new file is gitignored along with the rest of `data/memory/` but is
  present on disk (3,335 bytes, 7 lines).
