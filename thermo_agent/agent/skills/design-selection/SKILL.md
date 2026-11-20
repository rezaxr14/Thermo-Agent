---
name: design-selection
description: Use for Design tasks: select prime mover, cooling, rejection, bottoming cycle and district-heat source from a site's constraints using a selection matrix.
task: design
---
# Design selection

Objective: choose the plant that serves the site and passes every check with the least fuel, using the existing design-catalog as the task entry point.

## Variants
| variant | when it fits | key limits |
| --- | --- | --- |
| engine_plus_absorption | moderate chill, exhaust enough | limited exhaust heat and temperature |
| turbine_plus_absorption | large chill load | hotter, larger exhaust |
| with_bottoming | exhaust left after recovery | fluid temperature limit and stack floor |

## Physics and equations
- Prime mover: engine electrical efficiency near 0.43 with jacket loss unavailable; turbine exhaust is hotter and larger, so it can drive more absorption cooling.
- Cooling: absorption uses exhaust heat and adds no electricity; electric cooling adds parasitic power and needs a bigger prime mover.
- Rejection: a wet tower reaches wet-bulb plus its approach, a dry cooler dry-bulb plus its; a water limit rules wet towers out and a hot dry climate hurts dry coolers.
- Bottoming: an ORC turns leftover exhaust into power; the fluid's `T_max_K` and the `T_stack_min` floor decide feasibility, and district heat via heat pump leaves exhaust for the ORC.

## Parameters and constraints
- Pressures Pa, temperatures K, powers W and fractions not percent; quote only defaults present in the component registry.
- Key knobs are turbine `PR`, hot-water loop temperatures, `dT_rej`, `P_orc_bar` and `orc_frac`; each must keep the loop feasible.
- Verify feasibility (exhaust duty, temperature levels, water) before optimising fuel.

## Failure modes
- E_TEMP_CROSS on the exhaust exchanger -> chilled load beyond the exhaust heat -> switch to a larger prime mover or add electric cooling.
- ORC "exhaust too hot" -> pick a fluid with a higher vapour limit or recover heat upstream.
- V8 water limit -> replace a wet tower with a dry cooler and re-check rejection temperature.

## Tool checks
- `inspect_plant` cannot run before a design; build a candidate first, then use `validity_check` and `second_law_check`.
- `cycle_performance` (`kind`, `T_hot_K`, `T_cold_K`) and `psychrometrics` (`T_dry_K` and `RH` or `T_wet_K`): pass JSON object strings in the `candidate` slot to compare cycles and ambient limits.

## Answer habits
- Emit `<rationale>` then `<design>` with `key = value` lines only for the keys your choice requires.
- State the selection reason as a site constraint, not a preference.
- Use the design-catalog skill as the task entry; this matrix adds the selection reasoning.
