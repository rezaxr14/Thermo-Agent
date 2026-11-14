"""The architecture catalog must cover the diagram space the objective asks for."""
from thermo_agent.synth import architectures as A


def test_catalog_is_large_unique_and_labelled():
    names = [a["name"] for a in A.ARCHITECTURES]
    assert len(names) >= 60
    assert len(set(names)) == len(names)
    assert all(a["prime"] in ("turbine", "engine") for a in A.ARCHITECTURES)


def test_catalog_contains_the_requested_families():
    names = {a["name"] for a in A.ARCHITECTURES}
    assert "cgam_steam_orc" in names                     # gas turbine + HRSG + steam + ORC
    assert any(n.endswith("-orc") for n in names)        # ORC bottoming
    assert any("dh_heat_pump" in n for n in names)       # heat pump district heat
    assert any(n.startswith("engine-") for n in names)   # gas engine cogen
    assert any(a["rejection"] == "wet" for a in A.ARCHITECTURES)
    assert any(a["cooling"] == "absorption" for a in A.ARCHITECTURES)


def test_by_name_round_trips_and_rejects_unknown():
    a = A.by_name("engine-absorption-wet-orc-dh_heat_pump")
    assert a["orc"] and a["heat"] == "heat_pump" and a["rejection"] == "wet"
    try:
        A.by_name("no-such-plant")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown architecture should raise")


def test_orc_pressure_windows_stay_inside_each_fluid_range():
    for fluid, (lo, hi) in A.ORC_P_HIGH.items():
        assert 0 < lo < hi
        assert hi <= 14e5, f"{fluid} evaporator pressure is above what the exhaust can drive"
