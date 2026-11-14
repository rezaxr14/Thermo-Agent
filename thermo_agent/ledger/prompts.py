from __future__ import annotations
from . import program as P

SYSTEM = (
    "You debug exergy ledgers of energy plants. A ledger computes the exergy flow E[x] (MW) of every stream x from its state values, "
    "and the exergy destruction ED[c] (MW) of every component c as the exergy of its inlet streams minus the exergy of its outlet "
    "streams minus the work it delivers. A correct ledger gives ED >= 0 for every component and closes: the sum of all ED equals "
    "the exergy entering at the boundaries minus the exergy leaving minus the net work. You are given the ledger, the topology, the "
    "values and the result. Answer with a short <diagnosis> citing only numbers shown (MW), then <fix> with at most 4 lines "
    "`FIX T0 = <expr>`, `FIX E[<stream>] = <expr>` or `FIX ED[<component>] = <expr>`. Expressions use numbers, + - * /, "
    "parentheses, T0, and lookups m[x], h[x], s[x], g0[x], ech[x], E[x], W[c]. Change only what is wrong."
)


def _g(v: float) -> str:
    return f"{v:.4g}"


def numbers_shown(data: dict, result: dict) -> list[float]:
    """Every number printed in the prompt (used to check that a diagnosis cites only what it was shown)."""
    nums = [float(_g(v)) for v in result["ED"].values()] + [float(_g(v)) for v in result["E"].values()]
    nums += [float(_g(v)) for v in data["tables"]["W"].values()]
    if result["closure"] is not None: nums.append(float(_g(result["closure"])))
    return nums


def ledger_messages(data: dict, lines: dict, result: dict) -> list[dict]:
    t = data["tables"]
    ledger = "\n".join(f"{k} = {v}" for k, v in lines.items())
    topo = "\n".join(f"{c} {v['type']} in: {' '.join(v['in'])} out: {' '.join(v['out'])}" for c, v in data["topo"].items())
    topo += f"\nBOUNDARY in: {' '.join(data['ext_in'])} out: {' '.join(data['ext_out'])}"
    vals = "\n".join(f"{x}: m={_g(t['m'][x])} h={_g(t['h'][x])} s={_g(t['s'][x])} g0={_g(t['g0'][x])} ech={_g(t['ech'][x])}" for x in data["stream_order"])
    vals += "\nW (MW): " + " ".join(f"{c}={_g(w)}" for c, w in t["W"].items())
    res = []
    if result["errors"]: res.append("ERRORS: " + "; ".join(f"{k}: {m}" for k, m in result["errors"].items()))
    res.append("ED (MW): " + " ".join(f"{c}={_g(v)}{'(NEGATIVE)' if v < -1e-9 else ''}" for c, v in result["ED"].items()))
    res.append("E (MW): " + " ".join(f"{x}={_g(v)}" for x, v in result["E"].items()))
    if result["closure"] is not None: res.append(f"CLOSURE residual = {_g(result['closure'])} MW")
    user = f"LEDGER\n{ledger}\n\nAMBIENT T0_K={data['consts_true']['T0']!r}\n\nTOPOLOGY\n{topo}\n\nVALUES\n{vals}\n\nRESULT\n" + "\n".join(res)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
