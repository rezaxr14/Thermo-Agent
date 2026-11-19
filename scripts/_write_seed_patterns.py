import json, pathlib

rows = [
    # ---------------- repair ----------------
    dict(id="seed-repair-01", task="repair", fault_type="no_edits",
         architecture="", symptoms="plant still fails the same check after submit",
         lesson="Inspecting the broken plant is not a repair; the answer must change the flagged field.",
         fix="Read the failed check, then emit <edits> with a SET on the stream/component named in that check; re-inspect.",
         source="seed:method", accepted=False),
    dict(id="seed-repair-02", task="repair", fault_type="unit_slip",
         architecture="", symptoms="value is off by a power of ten; solver reports out-of-range property",
         lesson="Correct the power of ten, not the digits; MW vs W and kJ/kg vs J/kg are the usual slips.",
         fix="Rescale by the model unit (e.g. 5.0e5 Pa, not 5.0e6) and re-run inspect_plant.",
         source="seed:method", accepted=False),
    dict(id="seed-repair-03", task="repair", fault_type="minimal_edit",
         architecture="", symptoms="several checks fail at once",
         lesson="Fix one violated check at a time, smallest edit first, and re-inspect after each; do not rewrite the whole plant.",
         fix="Apply the single SET the first failed check asks for, re-inspect, then continue only if another check still fails.",
         source="seed:method", accepted=True),
    dict(id="seed-repair-04", task="repair", fault_type="verify_before_answer",
         architecture="", symptoms="candidate looks plausible but was never solved",
         lesson="An edit is a hypothesis until inspect_plant or validity_check returns ok; only then answer.",
         fix="Call inspect_plant after the edits and confirm failed_checks is empty before the final XML.",
         source="seed:method", accepted=True),
    # ---------------- audit ----------------
    dict(id="seed-audit-01", task="audit", fault_type="no_resolve",
         architecture="", symptoms="proposed change without a solved result",
         lesson="An audit change is only justified by the re-solved plant: exergy destruction and fuel must move.",
         fix="Apply the proposed SET with validity_check or solve_plant, read the new destruction/fuel, then rank.",
         source="seed:method", accepted=False),
    dict(id="seed-audit-02", task="audit", fault_type="negative_destruction",
         architecture="", symptoms="one line has negative exergy destruction",
         lesson="Negative destruction is a sign or efficiency error in that line, not a small rounding effect.",
         fix="Fix the sign/efficiency of the offending line first, then rank components by destruction.",
         source="seed:method", accepted=False),
    dict(id="seed-audit-03", task="audit", fault_type="rank_by_evidence",
         architecture="", symptoms="several components could be improved",
         lesson="Rank by the solved exergy destruction and pick the largest irreversible loss that the proposed edit actually reduces.",
         fix="Propose one SET on the top-destruction component, re-solve, and confirm the fuel or destruction change.",
         source="seed:method", accepted=True),
    # ---------------- ledger ----------------
    dict(id="seed-ledger-01", task="ledger", fault_type="closure_first",
         architecture="", symptoms="energy/exergy closure residual is nonzero",
         lesson="Check closure before line values: a closure error means some line or stream balance is wrong.",
         fix="Find the line whose balance or destruction is negative/zero-only and correct it, then re-run the closure check.",
         source="seed:method", accepted=True),
    dict(id="seed-ledger-02", task="ledger", fault_type="sign_error",
         architecture="", symptoms="exergy destruction negative or exergy rises across a real component",
         lesson="Destruction cannot be negative; a negative line is a sign or efficiency error, not a rounding issue.",
         fix="Flip the misplaced sign or efficiency in that line and re-check closure.",
         source="seed:method", accepted=False),
    dict(id="seed-ledger-03", task="ledger", fault_type="units",
         architecture="", symptoms="ledger numbers off by 1000 or by 3.6",
         lesson="Ledger powers are MW; heat rates may be kJ/kg or kJ/s. Convert before summing.",
         fix="Normalise every line to MW and re-check the closure residual.",
         source="seed:method", accepted=False),
    # ---------------- design ----------------
    dict(id="seed-design-01", task="design", fault_type="second_law_limit",
         architecture="", symptoms="efficiency or COP above the reversible limit",
         lesson="No cycle can beat its Carnot bound; an efficiency above the limit is invalid, not optimistic.",
         fix="Lower the efficiency/COP below the bound for the given hot and cold temperatures, then re-run validity_check.",
         source="seed:method", accepted=False),
    dict(id="seed-design-02", task="design", fault_type="property_range",
         architecture="", symptoms="solver rejects T below melt or P outside the fluid table",
         lesson="Every stream must sit inside the fluid's valid range at the proposed pressure.",
         fix="Raise the cold-end temperature above the melt point (or change fluid) and re-solve.",
         source="seed:method", accepted=False),
    dict(id="seed-design-03", task="design", fault_type="iterate_on_flag",
         architecture="", symptoms="candidate spec fails one check",
         lesson="Change only the field the tool flags, then re-run; repeated full redesigns hide which field was wrong.",
         fix="Apply the single flagged correction, re-run validity_check/solve_plant, stop when ok.",
         source="seed:method", accepted=True),
    dict(id="seed-design-04", task="design", fault_type="closure_check",
         architecture="", symptoms="spec solves but closure residual is high",
         lesson="A solved spec is not a valid design until mass, energy and exergy closure are ~0.",
         fix="Adjust the stream the residual points at and re-run solve_plant until closure < 1e-6.",
         source="seed:method", accepted=True),
]

out = pathlib.Path("data/memory/agent_patterns.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as fh:
    for r in rows:
        rec = dict(r)
        rec.setdefault("architecture", "")
        rec.setdefault("record_id", "")
        rec.setdefault("extra", {"seed": True, "evidence": "documented method; not a verifier-accepted row"})
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
print("wrote", out, len(rows), "notes")
