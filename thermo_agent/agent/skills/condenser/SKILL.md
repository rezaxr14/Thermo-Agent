---
name: condenser
description: Use when a plant contains a condenser: a hot working fluid condenses into a cooling-water stream whose temperature rise sets the duty and approach.
components: [condenser]
variants: [surface, evaporative, air_cooled]
---
# Condenser

Objective: reject the cycle's latent heat to cooling water without a temperature cross, while keeping the condensing temperature and water rise consistent with the rejection loop.

## Variants
| variant | when it fits | key limits |
| --- | --- | --- |
| surface | closed cooling-water loop | approach to water outlet |
| evaporative | water available | wet-bulb sets the loop temperature |
| air_cooled | water-limited site | high approach in hot weather |

## Physics and equations
- Hot outlet is saturated liquid at the hot-side pressure (quality x = 0); duty `Q = m_hot*(h_hot_in - h_sat_liquid)` in W.
- Cold outlet is `T_cold_in + dT_cw`; water flow is solved as `m_water = Q/(h_cold_out - h_cold_in)` and must not be fixed by hand.
- Minimum approach is `T_condensing - T_cold_out` in K; a negative value raises V4/E_TEMP_CROSS. The condensing temperature follows from the rejection-loop outlet plus `dT_cw` plus the approach.
- Subcooling and non-condensables are not modelled: the outlet is saturated liquid, and vacuum operation (pressure well below ambient) is allowed for the liquid side.

## Parameters and constraints
- `dT_cw`: cooling-water temperature rise (K), required, no default; `dT_cw` <= 0 raises E_PROP_RANGE.
- Both hot and cold ports take liquid/two-phase CoolProp fluids; a gas on these ports raises E_FLUID_MISMATCH.
- The hot inlet pressure sets the condensing temperature; the downstream pump/turbine pressures must match this loop pressure.

## Failure modes
- V4 temperature cross -> condensing temperature below water outlet -> raise condensing pressure or lower `dT_cw`.
- E_FLUID_MISMATCH -> gas on a liquid port -> correct the fluid class.
- E_NOT_CONVERGED in the loop -> turbine `P_out` not equal to condensing pressure -> align the tear stream.

## Tool checks
- `heat_exchanger`: pass a JSON object string in the `candidate` slot with `hot`/`cold` objects and `cold_T_out_K` (water rise); read `duty_W`, `pinch_K` and the hot outlet phase.
- `solve_plant` -> hot outlet quality and cold outlet `T_K`; `validity_check` -> `min_approach_K`.

## Answer habits
- Quote the condensing temperature, duty and water flow as solved, not assumed.
- Never specify the cooling-water flow: it follows from the duty and `dT_cw`.
- Treat the condenser pressure as the loop's low side shared with the turbine.
