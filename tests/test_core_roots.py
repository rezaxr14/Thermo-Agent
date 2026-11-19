"""The hosted benchmarks venv has numpy but no scipy; importing the teacher harness must still work."""
import builtins
import subprocess
import sys


def test_teacher_graph_imports_without_scipy():
    code = r"""
import builtins, sys
real = builtins.__import__
def guard(name, *a, **k):
    if name.split('.')[0] in ('scipy', 'CoolProp'):
        raise ModuleNotFoundError('No module named %s (blocked by test)' % name)
    return real(name, *a, **k)
builtins.__import__ = guard
import thermo_agent.teacher.agent_graph as g
import thermo_agent.teacher.toolkit as t
from thermo_agent.core import components, gasprops, cgam_plant, props, fluids
from thermo_agent.core._roots import using_scipy
assert using_scipy is False
assert 'solve_plant' in t.tool_names()
assert abs(gasprops.T_from_h({'N2': 1.0}, gasprops.h_mix({'N2': 1.0}, 900.0)) - 900.0) < 1e-6
print('ok')
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, out.stderr[-2000:]
    assert "ok" in out.stdout


def test_fallback_brentq_matches_bracket_contract():
    from thermo_agent.core._roots import _fallback_brentq
    root = _fallback_brentq(lambda x: x * x - 2.0, 0.0, 2.0, xtol=1e-14)
    assert abs(root - 2 ** 0.5) < 1e-10
    try:
        _fallback_brentq(lambda x: x * x + 1.0, 0.0, 2.0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("non-bracketing interval must raise ValueError")
