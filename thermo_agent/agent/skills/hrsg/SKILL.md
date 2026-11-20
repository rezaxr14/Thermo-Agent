---
name: hrsg
description: Use when a plant contains an hrsg: exhaust heats feedwater to steam at a set steam pressure, limited by the economiser approach, pinch and stack temperature.
components: [hrsg]
variants: [single_pressure, multi_pressure, once_through]
---
# Heat recovery steam generator

Objective: raise the required steam at `P_steam` from exhaust heat without crossing the pinch or dropping the gas below its stack limit.

## Variants
| variant | when it fits | key limits |
| --- | --- | --- |
| single_pressure | simple steam demand | one evaporating temperature and pinch |
| multi_pressure | several steam levels | model as separate HRSGs in series |
| once_through | no distinct boiling plateau | high gas inlet temperature needed |

## Physics and equations
- Saturation at `P_steam` fixes `T_sat` and the saturated-vapour enthalpy; duty `Q = m_water*(h_vap - h_feedwater)` in W, and `steam_out` is saturated vapour.
- Gas outlet follows `h_gas_out = h_gas_in - Q/m_gas` at `P_gas*(1 - dp_gas)`; the stack temperature is this outlet temperature and must stay above the acid-dew floor in real plants.
- Pinch: gas temperature at the bubble point must exceed `T_sat` by at least `econ_approach_K`; the reported minimum approach is the smaller of the pinch and the two end approaches.
- Blowdown and steam superheat are not modelled: the outlet is saturated vapour, so do not add a blowdown parameter.

## Parameters and constraints
- `P_steam`: steam pressure (Pa), required, no default; a higher pressure raises `T_sat` and the pinch, needing hotter exhaust.
- `dp_gas`: gas-side pressure-drop fraction (-), default 0.0; keep below 1.
- `econ_approach_K`: economiser approach (K), default 15.0; too small a value gives a temperature cross, too large wastes exhaust heat.
- Feedwater flow must be specified; the solver does not size it.

## Failure modes
- E_TEMP_CROSS or V4 -> pinch below `econ_approach_K` -> lower `P_steam` or raise exhaust temperature.
- E_UNDER_SPECIFIED on feedwater flow -> add the known water `m_dot`.
- V8 heat shortfall -> exhaust duty insufficient -> reduce steam production or recover less heat upstream.

## Tool checks
- `heat_exchanger`: pass a JSON object string in the `candidate` slot with `hot` (FLUE) and `cold` (Water) objects plus one outlet or duty spec; read `duty_W`, both outlets and `pinch_K`.
- `validity_check` -> `min_approach_K`; `solve_plant` -> `steam_out` temperature and phase.

## Answer habits
- Keep steam pressure in Pa and approach in K; quote only defaults that exist.
- Check the pinch before changing feedwater flow or `P_steam`.
- A higher steam pressure is not free: it consumes exhaust temperature needed elsewhere.
