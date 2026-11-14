---
name: orc-bottoming
description: Use when a plant contains an organic Rankine cycle (orc_evaporator, orc_turbine, pump, condenser) on exhaust heat.
components: [orc_evaporator, orc_turbine, pump, condenser]
---
# ORC bottoming cycle

<!-- AUTO:components -->
### orc_evaporator
Ports: hot_in (gas), hot_out (gas), cold_in (liquid), cold_out (liquid)
Params: `pinch_K`: pinch at the bubble point (K), 3.0; `dp_hot`: hot-side pressure-drop fraction (-), 0.0; `flow_frac`: working-fluid flow between its minimum (0) and maximum (1); used when the flow is not given, required if flow not given; `T_stack_min`: minimum exhaust outlet temperature (K), 333.15; `T_max_K`: maximum working-fluid vapour temperature (K), 420.0

### orc_turbine
Ports: in (liquid), out (liquid)
Params: `P_out`: outlet pressure (Pa), required; `eta_s`: isentropic efficiency (-), required

### pump
Ports: in (liquid), out (liquid)
Params: `P_out`: outlet pressure (Pa), required; `eta_s`: isentropic efficiency (-), required

### condenser
Ports: hot_in (liquid), hot_out (liquid), cold_in (liquid), cold_out (liquid)
Params: `dT_cw`: cooling-water temperature rise (K), required
<!-- /AUTO -->

## Behaviour
- Loop: condenser `hot_out` (saturated liquid, tear stream with fixed P and x = 0) -> pump -> evaporator `cold_in` -> `cold_out` -> turbine -> condenser `hot_in`. The turbine `P_out` must equal the pressure of the tear stream (the condensing pressure).
- When no flow is given, the evaporator sets the working-fluid flow from `flow_frac`: 0 is the smallest flow that keeps the vapour at or below `T_max_K`, 1 is the largest flow that keeps the exhaust at or above `T_stack_min`. If the smallest flow already exceeds the largest, the exhaust is too hot for that fluid: E_TEMP_CROSS ("too hot"). Use a fluid with a higher limit (R245fa 420 K, Cyclopentane 480 K, Toluene 560 K in these plants) or recover heat upstream first.
- The pinch is at the bubble point: the exhaust must be at least `pinch_K` above the working fluid's saturation temperature at the evaporation pressure, otherwise E_TEMP_CROSS.
- Condenser: cooling water rises by `dT_cw`; the condensing temperature must be above the water outlet temperature, otherwise V4 fails. Condensing temperature is set by the rejection loop's outlet temperature plus the water rise plus an approach.
- The pump needs a known flow; it receives it from the evaporator.

## Typical faults
- Pump `P_out` at or below the condensing pressure; turbine `P_out` not matching the tear pressure (E_NOT_CONVERGED).
- Working-fluid stream labelled with a gas fluid: E_FLUID_MISMATCH.
- `flow_frac` outside (0, 1].
