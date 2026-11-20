---
name: orc
description: Use when a plant has an organic Rankine cycle: an orc_evaporator and orc_turbine on exhaust heat, with fluid limits, pinch, flow position and a dry or wet expansion.
components: [orc_evaporator, orc_turbine]
variants: [subcritical, supercritical, with_recuperator]
---
# Organic Rankine cycle

Objective: pick a working fluid, evaporating pressure and flow that convert leftover exhaust heat to power while respecting the pinch, the fluid temperature limit and the stack floor.

## Variants
| variant | when it fits | key limits |
| --- | --- | --- |
| subcritical | standard exhaust temperatures | vapour limit and condensing pressure |
| supercritical | high exhaust temperature | harder pinch and control |
| with_recuperator | dry fluid leaving superheated | use a recuperator upstream of the condenser |

## Physics and equations
- The evaporator sets the working-fluid flow when it is not given: `flow_frac` 0 gives the smallest flow that keeps vapour at or below `T_max_K`, 1 the largest that keeps exhaust at or above `T_stack_min`; the flow is `n_min + flow_frac*(n_max - n_min)`.
- Pinch at the bubble point: `T_hot_in >= T_bubble + pinch_K`; if `n_min > n_max` the exhaust is too hot for that fluid at that stack floor and E_TEMP_CROSS is raised.
- Turbine work `W = m_wf*(h_in - h_out)` with `h_out = h_in - eta_s*(h_in - h(s_in, P_out))`; `P_out` must equal the condensing pressure of the loop tear stream.
- Dry fluids leave the turbine superheated, so a recuperator can recover that heat; wet fluids end near or inside the two-phase region and risk droplets.

## Parameters and constraints
- `pinch_K`: bubble-point pinch (K), default 3.0; `dp_hot`: hot-side drop fraction (-), default 0.0.
- `flow_frac`: working-fluid flow position (-), required when flow is not given and in (0, 1]. `T_stack_min`: exhaust floor (K), default 333.15. `T_max_K`: vapour limit (K), default 420.0; in these plants R245fa allows 420 K, Cyclopentane 480 K and Toluene 560 K.
- `P_out`: turbine outlet pressure (Pa), required; `eta_s`: isentropic efficiency (-), required, positive and below 1, with no default.

## Failure modes
- "exhaust too hot" E_TEMP_CROSS -> `T_max_K` below the needed vapour temperature -> choose a higher-limit fluid or recover heat upstream.
- E_NOT_CONVERGED -> turbine `P_out` not the condensing pressure -> match the tear stream.
- E_PROP_RANGE -> `flow_frac` outside (0, 1] -> bring it back into range.

## Tool checks
- `heat_exchanger` with `hot`/`cold` objects and a duty or outlet spec for the evaporator; `turbomachine` with `kind`, `fluid`, `T_in_K`, `P_in_kPa`, `P_out_kPa` and `eta_s` for the expander, each a JSON object string in the `candidate` slot.
- `cycle_performance` -> cycle efficiency; `validity_check` -> pinch and stack floor; `second_law_check` -> exergy destruction.

## Answer habits
- Keep pressure in Pa, temperature in K and flow position as a fraction.
- Select the fluid by its vapour limit against the exhaust temperature after upstream recovery.
- Quote work and efficiency from the tools, never from the fluid name alone.
