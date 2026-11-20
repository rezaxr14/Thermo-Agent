"""Tests for the continuous, CoolProp-backed teacher tools in thermo_tools.

No network access: every number comes from CoolProp 8 / the project's NASA-7 data
through the nine public callables, each invoked through the same
``fn(candidate_json_string) -> json_string`` contract the tool parser uses.
"""
from __future__ import annotations

import inspect
import json
import math

import pytest

from thermo_agent.teacher import thermo_tools as tt


def call(name: str, **kwargs) -> dict:
    """Invoke a tool by name with a JSON object and parse its JSON result."""
    result = getattr(tt, name)(json.dumps(kwargs))
    assert isinstance(result, str)
    return json.loads(result)


# --------------------------------------------------------------------------- #
# fluid_state
# --------------------------------------------------------------------------- #
def test_fluid_state_water_liquid_tp():
    o = call("fluid_state", fluid="Water", T_K=300.0, P_kPa=101.325, m_dot_kg_s=2.0)
    assert o["ok"] is True
    assert o["model"] == "CoolProp"
    assert o["T_K"] == pytest.approx(300.0)
    assert o["P_kPa"] == pytest.approx(101.325)
    assert 104 < o["h_kJ_kg"] < 113                       # task's stated range
    assert o["h_kJ_kg"] == pytest.approx(112.65, rel=0.01)
    assert o["rho_kg_m3"] == pytest.approx(996.6, rel=0.01)
    assert o["cp_kJ_kgK"] == pytest.approx(4.18, rel=0.02)
    assert o["phase"] == "liquid"
    assert o["quality"] is None
    assert o["m_dot_kg_s"] == 2.0


def test_fluid_state_quality_pair_uses_saturation():
    o = call("fluid_state", fluid="Water", T_K=373.15, q=0.5)
    assert o["ok"] is True
    assert o["P_kPa"] == pytest.approx(101.418, rel=0.02)   # 101.3 kPa +/- 2%
    assert o["quality"] == pytest.approx(0.5)
    assert o["phase"] == "two-phase"
    sat = call("saturation_state", fluid="Water", T_K=373.15)
    assert sat["h_f_kJ_kg"] < o["h_kJ_kg"] < sat["h_g_kJ_kg"]


def test_fluid_state_h_and_s_pairs_roundtrip():
    base = call("fluid_state", fluid="Water", T_K=300.0, P_kPa=101.325)
    from_h = call("fluid_state", fluid="Water", P_kPa=101.325, h_kJ_kg=base["h_kJ_kg"])
    from_s = call("fluid_state", fluid="Water", P_kPa=101.325, s_kJ_kgK=base["s_kJ_kgK"])
    assert from_h["ok"] and from_s["ok"]
    assert from_h["T_K"] == pytest.approx(300.0, abs=0.01)
    assert from_s["T_K"] == pytest.approx(300.0, abs=0.01)
    assert from_h["s_kJ_kgK"] == pytest.approx(base["s_kJ_kgK"], rel=1e-4)


def test_fluid_state_gas_uses_ideal_gas_model():
    o = call("fluid_state", fluid="CGAM_AIR", T_K=300.0, P_kPa=101.325)
    assert o["ok"] is True
    assert o["model"] == "ideal-gas-NASA7"
    assert o["phase"] == "gas"
    assert o["quality"] is None
    assert o["rho_kg_m3"] == pytest.approx(1.16, rel=0.03)
    assert o["cp_kJ_kgK"] == pytest.approx(1.02, rel=0.05)


def test_fluid_state_missing_pair_fails():
    o = json.loads(tt.fluid_state(json.dumps({"fluid": "Water", "T_K": 300.0})))
    assert o["ok"] is False
    assert "pair" in o["error"]


# --------------------------------------------------------------------------- #
# saturation_state
# --------------------------------------------------------------------------- #
def test_saturation_water_boiling_point():
    o = call("saturation_state", fluid="Water", T_K=373.15)
    assert o["ok"] is True
    assert o["P_sat_kPa"] == pytest.approx(101.3, rel=0.02)
    assert o["h_f_kJ_kg"] < o["h_g_kJ_kg"]
    assert 2250 < o["h_fg_kJ_kg"] < 2270
    assert o["s_f_kJ_kgK"] < o["s_g_kJ_kgK"]
    assert 0.0 < o["v_f_m3_kg"] < o["v_g_m3_kg"]
    assert o["h_kJ_kg"] == pytest.approx(o["h_f_kJ_kg"])   # default quality 0


def test_saturation_r245fa_quality_endpoints():
    lo = call("saturation_state", fluid="R245fa", P_kPa=100.0, quality=0)
    hi = call("saturation_state", fluid="R245fa", P_kPa=100.0, quality=1)
    mid = call("saturation_state", fluid="R245fa", P_kPa=100.0, quality=0.5)
    assert lo["ok"] and hi["ok"] and mid["ok"]
    assert lo["h_f_kJ_kg"] < hi["h_g_kJ_kg"]
    assert lo["T_sat_K"] == pytest.approx(hi["T_sat_K"])
    assert lo["h_kJ_kg"] < mid["h_kJ_kg"] < hi["h_kJ_kg"]
    assert mid["h_kJ_kg"] == pytest.approx((lo["h_f_kJ_kg"] + hi["h_g_kJ_kg"]) / 2)


# --------------------------------------------------------------------------- #
# turbomachine
# --------------------------------------------------------------------------- #
def test_turbomachine_compressor_air():
    o = call("turbomachine", kind="compressor", fluid="CGAM_AIR", T_in_K=300.0,
             P_in_kPa=101.325, pressure_ratio=8.0, eta_s=0.85, m_dot_kg_s=1.0)
    assert o["ok"] is True
    assert o["outlet"]["T_K"] > o["inlet"]["T_K"]
    assert o["outlet"]["P_kPa"] == pytest.approx(8 * 101.325, rel=1e-3)
    assert o["specific_work_kJ_kg"] > 0
    assert o["power_W"] == pytest.approx(o["specific_work_kJ_kg"] * 1000, rel=1e-6)
    assert o["outlet"]["s_kJ_kgK"] > o["inlet"]["s_kJ_kgK"]
    assert o["exergy_destruction_W"] > 0


def test_turbomachine_turbine_air():
    o = call("turbomachine", kind="turbine", fluid="CGAM_AIR", T_in_K=1200.0,
             P_in_kPa=800.0, pressure_ratio=8.0, eta_s=0.9, m_dot_kg_s=2.0)
    assert o["ok"] is True
    assert o["specific_work_kJ_kg"] > 0
    assert o["outlet"]["T_K"] < o["inlet"]["T_K"]
    assert o["outlet"]["P_kPa"] == pytest.approx(100.0, rel=1e-3)
    assert o["power_W"] == pytest.approx(2.0 * o["specific_work_kJ_kg"] * 1000, rel=1e-6)
    assert o["exergy_destruction_W"] >= 0


def test_turbomachine_pump_work_is_small_and_positive():
    o = call("turbomachine", kind="pump", fluid="Water", T_in_K=300.0,
             P_in_kPa=101.325, P_out_kPa=1000.0, eta_s=0.8, m_dot_kg_s=1.0)
    assert o["ok"] is True
    assert 0.5 < o["specific_work_kJ_kg"] < 3.0            # ~v*dP ~ 0.9 kJ/kg
    assert o["power_W"] > 0
    assert o["outlet"]["P_kPa"] == pytest.approx(1000.0)
    assert o["outlet"]["T_K"] == pytest.approx(300.0, abs=1.0)


# --------------------------------------------------------------------------- #
# heat_exchanger
# --------------------------------------------------------------------------- #
_HX = {
    "hot": {"fluid": "Water", "T_K": 350.0, "P_kPa": 1000.0, "m_dot_kg_s": 1.0},
    "cold": {"fluid": "Water", "T_K": 300.0, "P_kPa": 1000.0, "m_dot_kg_s": 1.0},
}


def test_heat_exchanger_energy_residual_is_tiny():
    o = call("heat_exchanger", **_HX, hot_T_out_K=330.0, flow_arrangement="counter")
    assert o["ok"] is True
    assert o["duty_W"] > 0
    assert abs(o["energy_residual_W"]) < 1e-2
    assert o["relative_energy_residual"] < 1e-3
    assert o["hot_outlet"]["T_K"] == pytest.approx(330.0)
    assert o["cold_outlet"]["T_K"] > o["cold_inlet"]["T_K"]
    assert o["min_approach_K"] > 0
    assert o["LMTD_K"] > 0
    assert o["UA_W_per_K"] > 0


def test_heat_exchanger_effectiveness_one_equal_capacity_rates():
    o = call("heat_exchanger", **_HX, effectiveness=1.0, flow_arrangement="counter")
    assert o["ok"] is True
    assert o["hot_outlet"]["T_K"] == pytest.approx(o["cold_inlet"]["T_K"], abs=0.2)
    assert o["cold_outlet"]["T_K"] == pytest.approx(o["hot_inlet"]["T_K"], abs=0.2)
    assert o["relative_energy_residual"] < 1e-3


def test_heat_exchanger_parallel_pinch_is_endpoint_based():
    o = call("heat_exchanger", **_HX, duty_W=100000.0, flow_arrangement="parallel")
    assert o["ok"] is True
    dT_in = o["hot_inlet"]["T_K"] - o["cold_inlet"]["T_K"]
    dT_out = o["hot_outlet"]["T_K"] - o["cold_outlet"]["T_K"]
    assert o["min_approach_K"] == pytest.approx(min(dT_in, dT_out), abs=1e-3)
    assert o["duty_W"] == pytest.approx(100000.0)


# --------------------------------------------------------------------------- #
# combustor
# --------------------------------------------------------------------------- #
def test_combustor_ch4_stoichiometry():
    o = call("combustor", fuel="CH4", air_excess=1.0, T_air_K=298.15,
             fuel_flow_kg_s=0.5, eta_comb=0.98, P_kPa=101.325)
    assert o["ok"] is True
    assert o["air_fuel_ratio"] == pytest.approx(17.16, rel=0.01)
    assert o["air_flow_kg_s"] == pytest.approx(0.5 * 17.16, rel=0.01)
    assert o["flue_flow_kg_s"] == pytest.approx(o["air_flow_kg_s"] + 0.5, rel=1e-6)
    assert o["heat_release_W"] == pytest.approx(0.5 * 50e6, rel=1e-6)
    assert o["useful_heat_W"] == pytest.approx(0.98 * 0.5 * 50e6, rel=1e-6)
    assert o["CO2_kg_s"] == pytest.approx(2.75 * 0.5, rel=1e-6)
    assert o["adiabatic_flame_K"] > o["T_air_K"]


def test_combustor_h2_alias_has_no_co2():
    o = call("combustor", fuel="H2", air_excess=1.0, T_air_K=298.15,
             fuel_flow_kg_s=1.0, eta_comb=1.0)
    assert o["ok"] is True
    assert o["CO2_kg_s"] == 0.0
    assert o["heat_release_W"] == pytest.approx(120e6, rel=1e-6)
    assert o["air_fuel_ratio"] == pytest.approx(34.05, rel=0.01)
    ng = call("combustor", fuel="NG", air_excess=1.0, T_air_K=298.15, fuel_flow_kg_s=1.0)
    assert ng["ok"] is True
    assert ng["air_fuel_ratio"] == pytest.approx(17.16, rel=0.01)


# --------------------------------------------------------------------------- #
# cycle_performance
# --------------------------------------------------------------------------- #
def test_cycle_rankine_efficiency_below_carnot():
    o = call("cycle_performance", kind="rankine", T_hot_K=573.15, T_cold_K=303.15,
             fluid="Water", eta_turbine=0.85, eta_pump=0.75, eta_second_law=0.55)
    assert o["ok"] is True
    assert o["carnot_efficiency"] == pytest.approx(1 - 303.15 / 573.15, rel=1e-6)
    assert 0 < o["real_efficiency"] < o["carnot_efficiency"]
    assert o["specific_work_kJ_kg"] > 0
    assert o["model"] == "CoolProp saturated Rankine cycle"


def test_cycle_heat_pump_cop_below_carnot():
    o = call("cycle_performance", kind="heat_pump", T_hot_K=333.15, T_cold_K=278.15,
             eta_second_law=0.5)
    assert o["ok"] is True
    assert o["carnot_cop"] == pytest.approx(333.15 / 55.0, rel=1e-6)
    assert o["real_cop"] == pytest.approx(0.5 * o["carnot_cop"], rel=1e-6)
    assert o["real_cop"] < o["carnot_cop"]
    assert o["lorenz_cop"] == pytest.approx(o["carnot_cop"])


def test_cycle_lorenz_cop_from_stream_temperatures():
    o = call("cycle_performance", kind="chiller", T_hot_K=380.0, T_cold_K=275.0,
             T_hot_in_K=400.0, T_hot_out_K=310.0, T_cold_in_K=280.0,
             T_cold_out_K=290.0, eta_second_law=0.5)
    assert o["ok"] is True
    assert o["lorenz_cop"] != pytest.approx(o["carnot_cop"])
    assert o["real_cop"] == pytest.approx(0.5 * o["lorenz_cop"], rel=1e-6)
    assert o["real_cop"] < o["carnot_cop"]


# --------------------------------------------------------------------------- #
# flow_network
# --------------------------------------------------------------------------- #
def test_flow_network_mixer_mass_and_energy_residuals():
    o = call("flow_network", kind="mixer", streams=[
        {"fluid": "Water", "T_K": 350.0, "P_kPa": 101.325, "m_dot_kg_s": 1.0},
        {"fluid": "Water", "T_K": 300.0, "P_kPa": 101.325, "m_dot_kg_s": 1.0},
    ])
    assert o["ok"] is True
    assert len(o["outlets"]) == 1
    assert o["outlets"][0]["m_dot_kg_s"] == pytest.approx(2.0)
    assert o["mass_residual_kg_s"] == pytest.approx(0.0, abs=1e-9)
    assert abs(o["energy_residual_W"]) < 1e-6
    assert o["entropy_generation_W_per_K"] >= 0
    assert o["exergy_destruction_W"] >= 0


def test_flow_network_splitter_conserves_mass():
    o = call("flow_network", kind="splitter", streams=[
        {"fluid": "Water", "T_K": 300.0, "P_kPa": 101.325, "m_dot_kg_s": 2.0},
    ], fraction=0.3)
    assert o["ok"] is True
    assert len(o["outlets"]) == 2
    flows = [s["m_dot_kg_s"] for s in o["outlets"]]
    assert flows == pytest.approx([0.6, 1.4])
    assert sum(flows) == pytest.approx(2.0)
    assert o["mass_residual_kg_s"] == pytest.approx(0.0, abs=1e-9)
    assert abs(o["energy_residual_W"]) < 1e-6
    assert abs(o["entropy_generation_W_per_K"]) < 1e-6


def test_flow_network_valve_isenthalpic():
    inlet = call("fluid_state", fluid="Water", T_K=300.0, P_kPa=500.0)
    o = call("flow_network", kind="valve", streams=[
        {"fluid": "Water", "T_K": 300.0, "P_kPa": 500.0, "m_dot_kg_s": 1.0},
    ], dp_kPa=400.0)
    assert o["ok"] is True
    assert o["outlets"][0]["P_kPa"] == pytest.approx(100.0)
    assert o["outlets"][0]["h_kJ_kg"] == pytest.approx(inlet["h_kJ_kg"], abs=1e-3)
    assert o["mass_residual_kg_s"] == pytest.approx(0.0, abs=1e-9)
    assert o["entropy_generation_W_per_K"] > 0
    assert o["exergy_destruction_W"] > 0


# --------------------------------------------------------------------------- #
# psychrometrics
# --------------------------------------------------------------------------- #
def test_psychrometrics_saturated_air():
    o = call("psychrometrics", T_dry_K=300.0, RH=1.0, P_kPa=101.325, T_water_K=299.0)
    assert o["ok"] is True
    assert o["wet_bulb_K"] == pytest.approx(300.0, abs=0.05)
    assert o["dew_point_K"] == pytest.approx(300.0, abs=0.05)
    assert o["humidity_ratio_kg_per_kg"] > 0
    assert o["density_kg_m3"] == pytest.approx(1.16, rel=0.05)
    assert o["approach_K"] == pytest.approx(-1.0)
    assert o["warnings"]


def test_psychrometrics_wet_bulb_input_reproduces_rh():
    base = call("psychrometrics", T_dry_K=300.0, RH=0.5, P_kPa=101.325)
    assert base["ok"] is True
    assert 0.010 < base["humidity_ratio_kg_per_kg"] < 0.013
    assert base["dew_point_K"] < base["wet_bulb_K"] < 300.0
    from_wb = call("psychrometrics", T_dry_K=300.0,
                   T_wet_K=base["wet_bulb_K"], P_kPa=101.325)
    assert from_wb["ok"] is True
    assert from_wb["relative_humidity"] == pytest.approx(0.5, abs=1e-4)
    assert from_wb["humidity_ratio_kg_per_kg"] == pytest.approx(
        base["humidity_ratio_kg_per_kg"], rel=1e-4)


# --------------------------------------------------------------------------- #
# exergy_stream
# --------------------------------------------------------------------------- #
def test_exergy_stream_positive_flow_exergy():
    o = call("exergy_stream", fluid="Water", T_K=500.0, P_KPa=1000.0,
             m_dot_kg_s=1.0, T0_K=298.15)
    assert o["ok"] is True
    assert o["specific_physical_exergy_kJ_kg"] > 0
    assert o["specific_chemical_exergy_kJ_kg"] == 0
    assert o["flow_exergy_W"] > 0
    assert o["flow_exergy_W"] == pytest.approx(
        o["specific_physical_exergy_kJ_kg"] * 1000.0, rel=1e-6)
    assert o["h_kJ_kg"] > 0


def test_exergy_stream_ch4_chemical_exergy():
    o = call("exergy_stream", fluid="CH4", T_K=300.0, P_kPa=101.325, m_dot_kg_s=0.1)
    assert o["ok"] is True
    assert 50000 < o["specific_chemical_exergy_kJ_kg"] < 55000
    assert o["flow_exergy_W"] > 0


# --------------------------------------------------------------------------- #
# contract / robustness
# --------------------------------------------------------------------------- #
def test_malformed_json_returns_ok_false():
    out = json.loads(tt.fluid_state("{not valid json"))
    assert out["ok"] is False
    assert out["error"]


def test_unknown_fluid_returns_ok_false():
    out = json.loads(tt.fluid_state(json.dumps(
        {"fluid": "DefinitelyNotAFluid", "T_K": 300.0, "P_kPa": 101.325})))
    assert out["ok"] is False
    assert "unknown fluid" in out["error"].lower()


def test_dynamic_tool_registry_matches_names():
    expected = ("fluid_state", "saturation_state", "turbomachine", "heat_exchanger",
                "combustor", "cycle_performance", "flow_network", "psychrometrics",
                "exergy_stream")
    assert tt.DYNAMIC_TOOL_NAMES == expected
    fns = tt.dynamic_tools()
    assert [f.__name__ for f in fns] == list(expected)
    assert all(inspect.isfunction(f) for f in fns)
    for fn in fns:
        sig = inspect.signature(fn)
        assert list(sig.parameters) == ["candidate"]
        assert sig.parameters["candidate"].default == ""
        # parser contract: fn("") must return a JSON string, never raise
        assert json.loads(fn(""))["ok"] is False


def test_all_returned_numbers_are_finite():
    results = [
        call("fluid_state", fluid="Water", T_K=300.0, P_kPa=101.325),
        call("saturation_state", fluid="Water", T_K=373.15),
        call("turbomachine", kind="compressor", fluid="CGAM_AIR", T_in_K=300.0,
             P_in_kPa=101.325, pressure_ratio=8.0, eta_s=0.85),
        call("heat_exchanger", **_HX, duty_W=100000.0),
        call("combustor", fuel="CH4", air_excess=1.1, T_air_K=298.15, fuel_flow_kg_s=0.5),
        call("cycle_performance", kind="rankine", T_hot_K=573.15, T_cold_K=303.15,
             fluid="Water"),
        call("flow_network", kind="valve", streams=[
            {"fluid": "Water", "T_K": 300.0, "P_kPa": 500.0, "m_dot_kg_s": 1.0}],
             dp_kPa=400.0),
        call("psychrometrics", T_dry_K=300.0, RH=0.5),
        call("exergy_stream", fluid="Water", T_K=500.0, P_kPa=1000.0, m_dot_kg_s=1.0),
    ]
    def finite(obj):
        if isinstance(obj, dict):
            return all(finite(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return all(finite(v) for v in obj)
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            return math.isfinite(obj)
        return True
    assert all(r["ok"] and finite(r) for r in results)
