"""Reference plant: CGAM gas turbine + HRSG + R245fa ORC, solved sequentially in plain Python.

Independent of the generic engine; used to cross-check it. Molar SI internally, powers reported in MW.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from ._roots import brentq
from . import gasprops as gp
from . import fluids as fl

ATM = 101325.0
BAR = 1e5
ORC, WATER = "R245fa", "Water"


@dataclass
class CgamParams:
    T1: float = 298.15; PR: float = 10.0
    eta_c: float = 0.86; eta_t: float = 0.86
    T3: float = 850.0; T4: float = 1520.0
    dp_ap_air: float = 0.05; dp_cc: float = 0.05; dp_gas_ap: float = 0.03; dp_hrsg: float = 0.05
    LHV: float = 802361.0          # J/mol (kJ/kmol)
    q_loss_cc: float = 0.02
    W_net_gt: float = 30e6         # W
    m_steam: float = 14.0          # kg/s
    P_steam: float = 20 * BAR; T_fw: float = 298.15
    T_fuel: float = 298.15; P_fuel: float = 12 * BAR
    P12: float = 10 * BAR; PR_orc: float = 6.76589986468200
    eta_pump: float = 0.75; eta_orct: float = 0.86
    n_orc: float = 0.3102          # kmol/s
    dT_pinch_eva: float = 3.0; dT_cw: float = 10.0
    T0: float = 298.0; P0: float = ATM
    ex_ch_CH4: float = 824384.0; ex_ch_flue: float = 111.43
    ex_ch_water: float = 45.0; ex_ch_orc_filler: float = 1484.82


def _flue(lam: float) -> dict:
    xc = (lam + 0.0003) / (1 + lam)
    xh = (4 * lam + 0.038) / (2 * (1 + lam))
    xn = 0.7748 / (1 + lam)
    xo = ((0.4118 + 0.0006 + 0.019) / (1 + lam) - 2 * xc - xh) / 2
    return {"CO2": xc, "H2O": xh, "O2": xo, "N2": xn}


def solve(p: CgamParams | None = None) -> dict:
    p = p or CgamParams()
    out: dict = {}
    air = gp.AIR_CGAM
    P1 = ATM; P2 = P1 * p.PR; P3 = P2 * (1 - p.dp_ap_air); P4 = P3 * (1 - p.dp_cc)
    P7 = ATM; P6 = P7 / (1 - p.dp_hrsg); P5 = P6 / (1 - p.dp_gas_ap)
    h1, s1 = gp.h_mix(air, p.T1), gp.s_mix(air, p.T1, P1)
    T2s = gp.T_from_s(air, s1, P2); h2s = gp.h_mix(air, T2s)
    h2 = h1 + (h2s - h1) / p.eta_c; T2 = gp.T_from_h(air, h2); s2 = gp.s_mix(air, T2, P2)
    h3, s3 = gp.h_mix(air, p.T3), gp.s_mix(air, p.T3, P3)
    h10 = gp.h_sp("CH4", p.T_fuel); s10 = gp.s_sp("CH4", p.T_fuel, p.P_fuel)

    def cc_resid(lam):
        return (1 + lam) * gp.h_mix(_flue(lam), p.T4) - h3 - lam * h10 + p.q_loss_cc * lam * p.LHV
    lam = brentq(cc_resid, 1e-3, 0.5, xtol=1e-14)
    x = _flue(lam)
    h4, s4 = gp.h_mix(x, p.T4), gp.s_mix(x, p.T4, P4)
    T5s = gp.T_from_s(x, s4, P5); h5s = gp.h_mix(x, T5s)
    h5 = h4 - p.eta_t * (h4 - h5s); T5 = gp.T_from_h(x, h5); s5 = gp.s_mix(x, T5, P5)
    w_per_na = (1 + lam) * (h4 - h5) - (h2 - h1)           # J per mol air
    n_a = p.W_net_gt / w_per_na / 1e3                        # kmol/s (W / (J/mol) = mol/s -> /1e3)
    n_F = lam * n_a; n_g = n_a * (1 + lam)
    h6 = h5 - (h3 - h2) / (1 + lam); T6 = gp.T_from_h(x, h6); s6 = gp.s_mix(x, T6, P6)

    # --- HRSG (water -> saturated steam at 20 bar)
    Mw = 18.015e-3
    n8 = p.m_steam / (Mw * 1e3)                              # kmol/s
    h8 = fl.h_TP(WATER, p.T_fw, p.P_steam); s8 = fl.s_TP(WATER, p.T_fw, p.P_steam)
    h9 = fl.h_Pq(WATER, p.P_steam, 1); s9 = fl.s_Pq(WATER, p.P_steam, 1); T9 = fl.Tsat_P(WATER, p.P_steam)
    h7 = h6 - n8 * (h9 - h8) / n_g; T7 = gp.T_from_h(x, h7); s7 = gp.s_mix(x, T7, P7)
    h_s = fl.h_TP(WATER, T9 - 15, p.P_steam)
    h_PP = h6 - n8 * (h9 - h_s) / n_g; T_PP = gp.T_from_h(x, h_PP)
    dT_pp_hrsg = T_PP - T9
    T_DP = fl.Tsat_P(WATER, x["H2O"] * P7)

    # --- ORC
    P12 = p.P12; P11 = P12 / p.PR_orc; P13 = P12; P14 = P11
    T11 = fl.Tsat_P(ORC, P11); h11 = fl.h_Pq(ORC, P11, 0); s11 = fl.s_Pq(ORC, P11, 0)
    v_pump = fl.v_Pq(ORC, P11, 0)
    w_p = v_pump * (P12 - P11) / p.eta_pump                  # J/mol
    h12 = h11 + w_p; T12 = fl.T_hP(ORC, h12, P12); s12 = fl.s_hP(ORC, h12, P12)
    T_s1 = fl.Tsat_P(ORC, P13); h_s1 = fl.h_Pq(ORC, P13, 0)
    h_PP1 = gp.h_mix(x, T_s1 + p.dT_pinch_eva)
    n_o = p.n_orc
    h13 = h_s1 + n_g * (h7 - h_PP1) / n_o
    T13 = fl.T_hP(ORC, h13, P13); s13 = fl.s_hP(ORC, h13, P13)
    T_sh = T13 - T_s1
    h17 = h7 - n_o * (h13 - h12) / n_g; T17 = gp.T_from_h(x, h17); s17 = gp.s_mix(x, T17, ATM)
    h14s = fl.h_sP(ORC, s13, P14); h14 = h13 - p.eta_orct * (h13 - h14s)
    T14 = fl.T_hP(ORC, h14, P14); s14 = fl.s_hP(ORC, h14, P14)
    T15 = T11; T16 = T11 + p.dT_cw
    h15, s15 = fl.h_TP(WATER, T15, ATM), fl.s_TP(WATER, T15, ATM)
    h16, s16 = fl.h_TP(WATER, T16, ATM), fl.s_TP(WATER, T16, ATM)
    n_w = n_o * (h14 - h11) / (h16 - h15)
    w_t = h13 - h14
    W_orc_net = n_o * (w_t - w_p) / 1e3                      # MW  (J/mol * kmol/s = kW -> /1e3)
    W_orct = n_o * w_t / 1e3; W_pump = n_o * w_p / 1e3

    # --- Exergy
    T0, P0 = p.T0, p.P0
    h0 = gp.h_mix(air, T0); s0 = gp.s_mix(air, T0, P0)
    h0cc = gp.h_mix(x, T0); s0cc = gp.s_mix(x, T0, P0)
    h0w, s0w = fl.h_TP(WATER, T0, P0), fl.s_TP(WATER, T0, P0)
    h0F, s0F = gp.h_sp("CH4", T0), gp.s_sp("CH4", T0, P0)
    h0o, s0o = fl.h_TP(ORC, T0, P0), fl.s_TP(ORC, T0, P0)
    n = {1: n_a, 2: n_a, 3: n_a, 4: n_g, 5: n_g, 6: n_g, 7: n_g, 8: n8, 9: n8, 10: n_F,
         11: n_o, 12: n_o, 13: n_o, 14: n_o, 15: n_w, 16: n_w, 17: n_g}
    H = {1: h1, 2: h2, 3: h3, 4: h4, 5: h5, 6: h6, 7: h7, 8: h8, 9: h9, 10: h10, 11: h11, 12: h12,
         13: h13, 14: h14, 15: h15, 16: h16, 17: h17}
    S = {1: s1, 2: s2, 3: s3, 4: s4, 5: s5, 6: s6, 7: s7, 8: s8, 9: s9, 10: s10, 11: s11, 12: s12,
         13: s13, 14: s14, 15: s15, 16: s16, 17: s17}
    ref = {**{i: (h0, s0) for i in (1, 2, 3)}, **{i: (h0cc, s0cc) for i in (4, 5, 6, 7, 17)},
           8: (h0w, s0w), 9: (h0w, s0w), 10: (h0F, s0F),
           **{i: (h0o, s0o) for i in (11, 12, 13, 14)}, 15: (h0w, s0w), 16: (h0w, s0w)}
    ch = {**{i: 0.0 for i in (1, 2, 3)}, **{i: p.ex_ch_flue for i in (4, 5, 6, 7, 17)},
          8: p.ex_ch_water, 9: p.ex_ch_water, 10: p.ex_ch_CH4,
          **{i: p.ex_ch_orc_filler for i in (11, 12, 13, 14, 15, 16)}}
    e = {i: n[i] * (H[i] - ref[i][0] - T0 * (S[i] - ref[i][1])) / 1e3 + n[i] * ch[i] / 1e3 for i in n}  # MW

    W_AC = n_a * (h2 - h1) / 1e3; W_GT = n_g * (h4 - h5) / 1e3; W_net = W_GT - W_AC
    ED = {"CC": e[3] + e[10] - e[4], "AP": e[2] + e[5] - e[3] - e[6], "AC": W_AC + e[1] - e[2],
          "GT": e[4] - e[5] - W_GT, "HRSG": e[6] + e[8] - e[7] - e[9], "EVA": e[7] + e[12] - e[17] - e[13],
          "ORCT": e[13] - e[14] - W_orct}
    ED["COND"] = e[14] + e[15] - e[11] - e[16]
    ED["P"] = W_pump + e[11] - e[12]
    ED_total = sum(ED.values())

    Q_in = n_F * p.LHV / 1e3
    Q_hrsg = n_g * (h6 - h7) / 1e3
    eta_I = (W_net + Q_hrsg + W_orc_net) / Q_in
    eta_I_gt = W_net / Q_in; eta_I_cogen = (W_net + Q_hrsg) / Q_in
    ex_in = e[10] + e[1]
    eta_II = (W_net + (e[9] - e[8]) + W_orc_net) / ex_in
    eta_II_cogen = (W_net + (e[9] - e[8])) / ex_in

    # Plant-level exergy closure: sum(E_D) = inputs - products - exhaust/cooling-water outflows
    closure_rhs = e[1] + e[10] + e[8] + e[15] - e[9] - e[17] - e[16] - W_net - W_orc_net
    out.update(dict(
        lam=lam, n_a=n_a, n_F=n_F, n_g=n_g, n_w=n_w, x=x, T={1: p.T1, 2: T2, 3: p.T3, 4: p.T4, 5: T5, 6: T6, 7: T7,
        9: T9, 11: T11, 12: T12, 13: T13, 14: T14, 15: T15, 16: T16, 17: T17},
        P={1: P1, 2: P2, 3: P3, 4: P4, 5: P5, 6: P6, 7: P7, 11: P11, 12: P12, 13: P13, 14: P14},
        h=H, s=S, n=n, e=e, ED=ED, ED_total=ED_total, ED_cogen=sum(ED[k] for k in ("CC", "AP", "AC", "GT", "HRSG")),
        W_AC=W_AC, W_GT=W_GT, W_net=W_net, W_orc_net=W_orc_net, W_orct=W_orct, W_pump=W_pump, Q_in=Q_in, Q_hrsg=Q_hrsg,
        eta_I=eta_I, eta_I_gt=eta_I_gt, eta_I_cogen=eta_I_cogen, eta_II=eta_II, eta_II_cogen=eta_II_cogen,
        T_DP_C=T_DP - 273.15, dT_pinch_hrsg=dT_pp_hrsg, T_superheat=T_sh, closure_residual=ED_total - closure_rhs))
    return out


if __name__ == "__main__":
    r = solve()
    for k in ("lam", "n_a", "W_GT", "W_AC", "W_net", "W_orct", "W_pump", "Q_in", "eta_I", "eta_II", "T_DP_C",
              "dT_pinch_hrsg", "T_superheat", "ED_total", "closure_residual"):
        print(f"{k:18s} {r[k]:.5g}")
    print({k: round(v, 4) for k, v in r["ED"].items()})
    print("T:", {k: round(v, 1) for k, v in r["T"].items()})
