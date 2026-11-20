---
name: combustor
description: Use when a plant contains a combustor: fuel and air burned to a set outlet temperature with pressure drop and heat loss, limited by oxygen, excess air and emissions.
components: [combustor]
variants: [gas_turbine_canular, boiler_burner, engine_cylinder]
---
# Combustor

Objective: reach the required hot-gas temperature with the available oxygen and acceptable pressure loss, while limiting heat loss and incomplete-combustion products.

## Variants
| variant | when it fits | key limits |
| --- | --- | --- |
| gas_turbine_canular | continuous high-pressure firing | pressure drop and blade inlet temperature |
| boiler_burner | low-pressure steam generation | excess air and stack oxygen |
| engine_cylinder | intermittent firing | knock, peak temperature, CO/NOx |

## Physics and equations
- Fuel flow is solved from the oxygen balance at the target `T_out`; the outlet is `FLUE` at `T_out` and `P_air*(1 - dp)`, and `fuel.n` is set by the solver.
- Heat balance: `(1 + lam)*h_mix(y, T_out) = h_air + lam*h_fuel - q_loss*lam*LHV`, with `lam` the fuel-to-air molar ratio; heat loss `Q_loss = q_loss * n_fuel * LHV` in W.
- Adiabatic flame temperature is the `T_out` obtained with `q_loss = 0`; a `T_out` above what the available oxygen can reach gives E_NOT_CONVERGED.
- Excess air is set by the air stream composition and flow, not by a combustor parameter: more air lowers flame temperature but raises mass flow and stack loss. CO falls and NOx rises as excess air and flame temperature change.

## Parameters and constraints
- `T_out`: outlet temperature (K), required, no default; it must lie inside the gas property range and below the available-oxygen limit.
- `dp`: gas-side pressure-drop fraction (-), default 0.0; a fraction above 1 is invalid.
- `q_loss`: heat-loss fraction of fuel LHV (-), default 0.0; larger loss needs more fuel for the same `T_out`.

## Failure modes
- E_NOT_CONVERGED -> `T_out` unreachable with the oxygen present -> lower `T_out` or add air.
- E_PROP_RANGE -> non-gas inlets or a temperature outside the data -> check fluids and units (K).
- V8 fuel excess -> `q_loss` or `dp` raised too far -> restore a small loss and drop.

## Tool checks
- `combustor`: pass a JSON object string in the `candidate` slot, e.g. `{"fuel": "CH4", "air_excess": 1.2, "T_air_K": 300, "eta_comb": 0.99, "T_out_K": 1400}`; read fuel flow and `adiabatic_flame_K`.
- `solve_plant` -> `FLUE` outlet state; `second_law_check` -> the usually largest exergy destruction (MW).

## Answer habits
- State fuel and air flows as solved values; never fix a fuel `m_dot` by hand.
- Report `dp` and `q_loss` as fractions, and temperatures in K.
- Attribute the largest plant exergy destruction to combustion, not to the heat rejector.
