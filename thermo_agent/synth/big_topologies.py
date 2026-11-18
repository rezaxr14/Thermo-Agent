"""Expert-scale thermodynamic cycles built ONLY from existing component models.

The steam turbine is an ``orc_turbine`` carrying Water; multi-pressure levels are several
``hrsg`` units in series on the exhaust; reheat is a ``recuperator`` between two steam
turbines. Closed Rankine loops use tear streams with fixed ``{P, x=0}`` condensate.

Nothing here changes the simulator, the verifiers or any existing small-cycle builder.
"""
from __future__ import annotations

from ..core import fluids as fl, gasprops as gp
from ..core.components import rejector_outlet_T
from ..core.schema import Ambient, Component, FixedState, Plant, Stream

ATM = 101325.0

C = lambda i, t, **p: Component(id=i, type=t, params=p)
S = lambda i, a, b, fluid, **kw: Stream(id=i, **{"from": a, "to": b}, fluid=fluid, **kw)


def big_cchp_steam2p_reheat_hp(
    *, n_air: float = 5200.0, m_hp: float = 14.0, m_lp: float = 20.0,
    P_hp: float = 40e5, P_lp: float = 8e5, T_reheat: float = 500.0,
    T_cond: float = 313.15, P_it: float = 80e6, chilled_fraction: float = 0.5,
    T_amb: float = 298.15, name: str = "big_cchp_steam2p_reheat_hp", **kw,
) -> Plant:
    """Two-pressure steam Rankine with reheat + chilled water + district heat + heat pump.

    Gas path: AC -> CC -> GT -> AP -> H1(HP HRSG) -> RH(reheat) -> H2(LP HRSG) -> HXD(DH) -> stack.
    HP steam: PS -> H1 -> ST1 -> RH -> ST2 -> CS -> PS. LP steam: PS2 -> H2 -> ST3 -> CS2 -> PS2.
    Cooling: DC -> CH -> DC; CH rejection -> HP -> REJ2 -> CH. Heat pump delivers a second DH level.
    """
    P_cond = fl.Psat_T("Water", T_cond)
    M_air = gp.mw(gp.AIR_CGAM) / 1000.0
    P_exh = ATM / ((1 - 0.02) * (1 - 0.01) * (1 - 0.02) * (1 - 0.03))
    T_chw = 291.15
    T_dh = 333.15
    comps = [
        C("AC", "compressor", PR=11.0, eta_s=0.86),
        C("CC", "combustor", T_out=1550.0, dp=0.04, q_loss=0.01),
        C("GT", "turbine", P_out=P_exh, eta_s=0.86),
        C("AP", "recuperator", T_cold_out=650.0, dp_cold=0.05, dp_hot=0.03),
        C("H1", "hrsg", P_steam=P_hp, dp_gas=0.02, econ_approach_K=15.0),
        C("RH", "recuperator", T_cold_out=T_reheat, dp_hot=0.01),
        C("H2", "hrsg", P_steam=P_lp, dp_gas=0.02, econ_approach_K=15.0),
        C("HXD", "recuperator", T_cold_out=343.15, dp_cold=0.01, useful_heat=True),
        C("ST1", "orc_turbine", P_out=P_lp, eta_s=0.86),
        C("ST2", "orc_turbine", P_out=P_cond, eta_s=0.86),
        C("PS", "pump", P_out=P_hp, eta_s=0.75),
        C("CS", "condenser", dT_cw=5.0),
        C("ST3", "orc_turbine", P_out=P_cond, eta_s=0.84),
        C("PS2", "pump", P_out=P_lp, eta_s=0.75),
        C("CS2", "condenser", dT_cw=5.0),
        C("DC", "dc_load", P_it=P_it, chilled_fraction=chilled_fraction, T_return=301.15),
        C("CH", "electric_chiller", T_chw_out=T_chw, dT_rej=7.0, eps=0.5),
        C("HP", "heat_pump", T_sink_out=T_dh, Q_sink=5e6, eps=0.5),
        C("REJ2", "heat_rejector", kind="wet", approach_K=4.0, fan_frac=0.015),
    ]
    streams = [
        # gas path
        S("air", "ext:air", "AC.in", "CGAM_AIR", m_dot=n_air * M_air, fixed=FixedState(T=T_amb, P=ATM)),
        S("s2", "AC.out", "AP.cold_in", "CGAM_AIR"),
        S("s3", "AP.cold_out", "CC.air_in", "CGAM_AIR"),
        S("fuel", "ext:fuel", "CC.fuel_in", "CH4", fixed=FixedState(T=T_amb, P=12e5)),
        S("s4", "CC.out", "GT.in", "FLUE"),
        S("s5", "GT.out", "AP.hot_in", "FLUE"),
        S("s6", "AP.hot_out", "H1.hot_in", "FLUE"),
        S("s7", "H1.hot_out", "RH.hot_in", "FLUE"),
        S("s8", "RH.hot_out", "H2.hot_in", "FLUE"),
        S("s9", "H2.hot_out", "HXD.hot_in", "FLUE"),
        S("stack", "HXD.hot_out", "ext:stack", "FLUE"),
        # high-pressure steam loop (tear on condensate into the pump)
        S("hp_cond", "CS.hot_out", "PS.in", "Water", m_dot=m_hp, fixed=FixedState(P=P_cond, x=0.0)),
        S("hp_fw", "PS.out", "H1.water_in", "Water"),
        S("hp_steam", "H1.steam_out", "ST1.in", "Water"),
        S("hp_reheat", "ST1.out", "RH.cold_in", "Water"),
        S("hp_reheat2", "RH.cold_out", "ST2.in", "Water"),
        S("hp_exhaust", "ST2.out", "CS.hot_in", "Water"),
        S("hp_cw_in", "ext:cw", "CS.cold_in", "Water", fixed=FixedState(T=T_amb, P=ATM)),
        S("hp_cw_out", "CS.cold_out", "ext:cw_out", "Water"),
        # low-pressure steam loop
        S("lp_cond", "CS2.hot_out", "PS2.in", "Water", m_dot=m_lp, fixed=FixedState(P=P_cond, x=0.0)),
        S("lp_fw", "PS2.out", "H2.water_in", "Water"),
        S("lp_steam", "H2.steam_out", "ST3.in", "Water"),
        S("lp_exhaust", "ST3.out", "CS2.hot_in", "Water"),
        S("lp_cw_in", "ext:cw2", "CS2.cold_in", "Water", fixed=FixedState(T=T_amb, P=ATM)),
        S("lp_cw_out", "CS2.cold_out", "ext:cw2_out", "Water"),
        # chilled-water loop (tear on the supply into the data hall)
        S("chw_sup", "CH.evap_out", "DC.in", "Water", fixed=FixedState(T=T_chw, P=ATM)),
        S("chw_ret", "DC.out", "CH.evap_in", "Water"),
        # chiller rejection -> heat pump source -> wet tower -> chiller (tear on tower outlet)
        S("rej_hot", "CH.rej_out", "HP.src_in", "Water"),
        S("rej_mid", "HP.src_out", "REJ2.in", "Water"),
        S("rej_cold", "REJ2.out", "CH.rej_in", "Water",
          fixed=FixedState(T=rejector_outlet_T(T_amb, ATM, 0.5, "wet", 4.0), P=ATM)),
        # district heat levels
        S("dh_exh", "ext:dh", "HXD.cold_in", "Water", m_dot=60.0,
          fixed=FixedState(T=313.15, P=ATM)),
        S("dh_sup", "HXD.cold_out", "ext:dh_out", "Water"),
        S("dh2_exh", "ext:dh2", "HP.sink_in", "Water", fixed=FixedState(T=318.15, P=ATM)),
        S("dh2_sup", "HP.sink_out", "ext:dh2_out", "Water"),
    ]
    return Plant(name=name, ambient=Ambient(T0=T_amb, P0=ATM, RH=0.5),
                 components=comps, streams=streams)


# kg/s of steam per mol/s of combustion air, used when sizing the plant to the IT load.
STEAM_PER_AIR_HP = 14.0 / 5200.0
STEAM_PER_AIR_LP = 20.0 / 5200.0

BIG_BUILDERS = {"big_cchp_steam2p_reheat_hp": big_cchp_steam2p_reheat_hp}
