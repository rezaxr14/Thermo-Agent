import json, random
import pytest
from thermo_agent.core.schema import Site, Ambient
from thermo_agent.design import spec as S
from thermo_agent.design.data import make_row
from thermo_agent.design.loop import design_refine
from thermo_agent.design.search import polish, anytime_enumeration, anytime_reference, enumerate_designs
from thermo_agent.design import curves as C

SITE = Site(it_load_w=20e6, cooling_load_w=10e6, grid_import_allowed=False, ambient=Ambient(T0=298.15, RH=0.6), t_supply_k=291.15, t_return_k=301.15)
ROW = make_row("c-1", SITE, max_evals=12)

def test_polish_trace_is_monotone_and_matches_the_count():
    tr = []; sp, f, n = polish(SITE, S.default_spec("turbine", "absorption", "wet"), 20, tr)
    vals = [x for _, x in tr if x is not None]; assert len(tr) == n and vals == sorted(vals, reverse=True) and abs(vals[-1] - f) < 1e-6

def test_anytime_curves_never_get_worse_and_end_at_the_best_found():
    c = anytime_enumeration(SITE); assert len(c) == 32; v = [x for x in c if x is not None]; assert v == sorted(v, reverse=True)
    r = anytime_reference(SITE, max_evals=10); v = [x for x in r if x is not None]; assert len(r) > 32 and v == sorted(v, reverse=True) and v[-1] <= min(f for _, e in enumerate_designs(SITE) if (f := e.fuel_w))

def test_refine_loop_reports_an_anytime_curve_and_stops_when_stale():
    good = ROW["completion"][0]["content"]; worse = "<rationale>x</rationale>\n" + S.format_design(S.default_spec("turbine", "electric", "dry"))
    seq = iter([worse, good, good, good, good]); fb = []
    out = design_refine(lambda m: (fb.append(m[-1]["content"]) or next(seq)), ROW, max_turns=6, patience=2)
    gaps = [g for _, g in out["anytime"]]; assert out["valid"] and gaps[0] > 0.01 and gaps[1] <= 1e-6 and out["turns"] == 4 and out["evals"] == 4   # stops after 2 stale turns
    assert all(a >= b for a, b in zip(gaps, gaps[1:])) and any("fuel input" in x and "best so far" in x for x in fb)

def test_refine_loop_with_invalid_designs_counts_evaluations_only_when_the_design_built():
    bad = "<rationale>x</rationale>\n<design>\nprime_mover = rocket\n</design>"
    out = design_refine(lambda m: bad, ROW, max_turns=3); assert not out["valid"] and out["evals"] == 0 and out["anytime"] == [(0, None)] * 3

def test_summary_at_budget_and_model_curve_conversion():
    s = C.summarize([[None, 0.05, 0.01], [0.0, 0.0, 0.0], [None, None, 0.2]], [1, 2, 3])
    assert s[0]["valid"] == 0.333 and s[1]["valid"] == 0.667 and s[2]["within_tol"] == 0.667 and s[2]["mean_gap_pct"] == pytest.approx(7.0)
    rep = {"design": {"test_id": {"examples": [{"anytime": [[1, None], [2, 0.04], [4, 0.01]]}]}}}
    assert C.model_curves(rep, "test_id") == [[None, 0.04, 0.04, 0.01]]

def test_plot_writes_a_figure(tmp_path):
    p = tmp_path / "f.png"; C.plot({"a": C.summarize([[None, 0.01]], [1, 2]), "b": C.summarize([[0.0, 0.0]], [1, 2])}, p); assert p.stat().st_size > 1000
