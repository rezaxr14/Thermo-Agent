---
name: evaporator
description: Use when exhaust boils a working fluid in an orc_evaporator: bubble-point pinch, superheat, flow position, dryout and freeze limits.
components: [orc_evaporator]
variants: [flooded, direct_expansion, falling_film]
---
# Evaporator

Objective: boil the working fluid at the highest useful pressure that keeps the pinch above the bubble point, the vapour inside its temperature limit and the exhaust above its stack floor.

## Variants
| variant | when it fits | key limits |
| --- | --- | --- |
| flooded | stable boiling, good wetting | large charge; bubble-point pinch controls |
| direct_expansion | compact, low charge | superheat control; dryout at high quality |
| falling_film | temperature-sensitive fluid | needs adequate wetting to avoid dryout |

## Physics and equations
- Bubble temperature `T_b = Tsat_P(fluid, P_cold)`; the hot inlet must satisfy `T_hot_in > T_b + pinch_K`, otherwise E_TEMP_CROSS.
- When flow is not given, `flow_frac` interpolates between the minimum flow that respects `T_max_K` and the maximum that respects `T_stack_min`: `n = n_min + flow_frac*(n_max - n_min)`.
- Duty `Q = n_wf*(h_cold_out - h_cold_in)`; the hot outlet is `h_hot_out = h_hot_in - Q/m_hot` at `P_hot*(1 - dp_hot)`, and the reported minimum approach includes the pinch and both ends.
- Superheat raises the vapour enthalpy but also the metal temperature; dryout and freeze protection are operational limits, not solver parameters.

## Parameters and constraints
- `pinch_K`: bubble-point pinch (K), default 3.0; `dp_hot`: exhaust pressure-drop fraction (-), default 0.0.
- `flow_frac`: flow position (-), required when flow is not given, in (0, 1]; `T_stack_min`: exhaust floor (K), default 333.15; `T_max_K`: vapour limit (K), default 420.0.
- The cold side is a liquid/two-phase CoolProp fluid and the hot side a gas; a fluid-class mismatch raises E_FLUID_MISMATCH.

## Failure modes
- "exhaust too hot" E_TEMP_CROSS -> `T_max_K` below the needed vapour temperature -> raise the fluid limit or recover heat upstream.
- "pinch below stack" E_TEMP_CROSS -> `T_b + pinch_K` above `T_stack_min` -> lower the evaporating pressure or pinch.
- E_PROP_RANGE -> `flow_frac` outside (0, 1] -> bring it into range.

## Tool checks
- `heat_exchanger`: pass a JSON object string in the `candidate` slot with `hot`/`cold` objects and one outlet or duty spec; read `duty_W`, both outlets and `pinch_K`.
- `saturation_state` -> bubble temperature and enthalpy at the cold pressure; `validity_check` -> `min_approach_K`.

## Answer habits
- Keep pressure in Pa, temperature in K and flow position as a fraction.
- Check the pinch and the fluid limit before adjusting `flow_frac`.
- Quote superheat and dryout as qualitative limits; the solver does not expose them as parameters.
