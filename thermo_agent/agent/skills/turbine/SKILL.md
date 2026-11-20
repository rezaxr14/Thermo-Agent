---
name: turbine
description: Use when a plant contains a turbine: gas or steam expansion through an outlet pressure and isentropic efficiency, with backpressure and blade-temperature limits.
components: [turbine, orc_turbine]
variants: [steam, gas, orc_radial_axial]
---
# Turbine

Objective: choose the outlet pressure and isentropic efficiency that turn the largest safe enthalpy drop into shaft work, without condensation damage or an impossible backpressure.

## Variants
| variant | when it fits | key limits |
| --- | --- | --- |
| steam | high-temperature Rankine exhaust | wetness at outlet; condenser backpressure |
| gas | gas-turbine power expansion | inlet blade-metal temperature limit |
| orc_radial_axial | organic bottoming loop | dry vs wet fluid expansion, small enthalpy drop |

## Physics and equations
- `Pout` is the outlet pressure in Pa; isentropic target `h2s = h(s1, Pout)`; actual `h2 = h1 - eta_s*(h1 - h2s)`; shaft power `W = m_dot*(h1 - h2)` in W.
- A larger pressure drop gives more work until the outlet state leaves the valid region: a gas/steam `turbine` uses a gas-class inlet, while `orc_turbine` uses a liquid/two-phase working fluid.
- Steam and wet ORC fluids can end expansion below saturated vapour (quality x < 1): liquid droplets erode blades, so a minimum outlet quality and a condenser backpressure set `Pout`.
- Gas-turbine inlet temperature is limited by blade metal; the combustor sets it, not this component. Backpressure above ambient reduces the available drop.

## Parameters and constraints
- `P_out`: outlet pressure (Pa), required, no default; it must be below the inlet pressure, and for a condensing machine it must match the downstream condenser or tear pressure.
- `eta_s`: isentropic efficiency (-), required, no default; `eta_s` <= 0 raises E_PROP_RANGE; keep below 1.
- The `orc_turbine` `P_out` must equal the condensing pressure set by the ORC loop tear stream.

## Failure modes
- E_TEMP_CROSS or non-converged loop -> `P_out` above condensing pressure -> lower it to the tear pressure.
- E_PROP_RANGE -> `eta_s` <= 0 or an outlet pressure outside the fluid data -> correct the unit (Pa, not bar) and range.
- V3 negative exergy destruction -> efficiency above 1 -> reduce `eta_s`.

## Tool checks
- `turbomachine`: pass a JSON object string in the `candidate` slot, e.g. `{"kind": "turbine", "fluid": "FLUE", "T_in_K": 1200, "P_in_kPa": 1000, "P_out_kPa": 101.325, "eta_s": 0.85}`; read `power_W` and outlet quality.
- `cycle_performance` -> cycle efficiency; `solve_plant` -> outlet quality/phase; `second_law_check` -> exergy destruction (MW).

## Answer habits
- Never write a bar value into `P_out`: it is Pa.
- For repair, first check the outlet pressure against the downstream loop, then the efficiency.
- Report work, outlet temperature and efficiency only as the tools show them.
