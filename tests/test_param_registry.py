import inspect, re
from thermo_agent.core import components as C

def _used(fn):
    src = inspect.getsource(fn)
    return set(re.findall(r'_p\(c, "(\w+)"', src)) | set(re.findall(r'c\.params\.get\("(\w+)"', src)) | set(re.findall(r'"(\w+)" in c\.params', src))

def test_registry_matches_what_each_solver_reads():
    for ctype, fn in C.SOLVERS.items():
        assert ctype in C.PARAMS, ctype
        assert _used(fn) == set(C.PARAMS[ctype]) - {"useful_heat"}, (ctype, _used(fn) ^ set(C.PARAMS[ctype]))

def test_every_component_type_with_ports_is_registered():
    assert set(C.REQUIRED_PORTS) == set(C.PARAMS) == set(C.SOLVERS)
