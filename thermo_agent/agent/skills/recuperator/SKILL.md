---
name: recuperator
description: Use when a plant contains a recuperator: two streams exchange heat to a set cold outlet with pressure drops, a minimum approach and an optional delivered-heat flag.
components: [recuperator]
variants: [gas_gas_recuperator, economiser, air_preheater]
---
# Recuperator

Objective: transfer the specified duty from hot to cold without a temperature cross, and decide whether the duty counts as useful heat or as internal recovery.

## Variants
| variant | when it fits | key limits |
| --- | --- | --- |
| gas_gas_recuperator | turbine exhaust preheats compressed air | small approach, gas-side dp |
| economiser | feedwater heated by exhaust | approach to saturation, pinch |
| air_preheater | combustion air heated by flue gas | fouling raises dp and lowers duty |

## Physics and equations
- Cold outlet is fixed by `T_cold_out`; duty `Q = m_cold*(h_cold_out - h_cold_in)` in W. The hot outlet follows `h_hot_out = h_hot_in - Q/m_hot` at `P_hot*(1 - dp_hot)`.
- Minimum approach is `min(T_hot_in - T_cold_out, T_hot_out - T_cold_in)` in K; a negative value is a temperature cross and raises E_TEMP_CROSS.
- Effectiveness is the realised duty over the maximum possible duty set by the inlet temperatures and flows; it rises as the approach shrinks, at the cost of area and pressure drop.
- Pressure drops `dp_cold` and `dp_hot` are fractions of each inlet pressure, not fixed Pa. Fouling and leakage show up as a larger drop or a smaller achievable outlet temperature.

## Parameters and constraints
- `T_cold_out`: cold-side outlet temperature (K), required, no default; it must be below the hot inlet and above the cold inlet.
- `dp_cold`, `dp_hot`: pressure-drop fractions (-), default 0.0; keep each below 1.
- `useful_heat`: flag, default False; set true for a district-heat exchanger so its duty counts as delivered heat.

## Failure modes
- E_TEMP_CROSS -> `T_cold_out` above the hot inlet, or duty larger than the hot stream can supply -> lower `T_cold_out`.
- V4 min approach below 0 K -> crossed streams -> swap hot/cold inlets or reduce the duty.
- V8 heat shortfall -> `useful_heat` left False on the district exchanger -> set it true.

## Tool checks
- `heat_exchanger`: pass a JSON object string in the `candidate` slot with `hot` and `cold` objects (`fluid`, `T_K`, `P_kPa`, `m_dot_kg_s`) and one of `hot_T_out_K`, `cold_T_out_K`, `duty_W`, `effectiveness`; read `duty_W` and `pinch_K`.
- `validity_check` -> `min_approach_K`; `second_law_check` -> exergy destruction grows with the approach.

## Answer habits
- Quote the cold outlet, duty and both pressure drops in the solver's numbers and units.
- For district heat, check the `useful_heat` flag before changing temperatures.
- A large duty at a tiny approach is not automatically better: it raises area and irreversibility.
