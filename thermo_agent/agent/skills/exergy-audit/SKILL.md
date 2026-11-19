---
name: exergy-audit
description: Use for Audit tasks: ranking exergy destruction, choosing a fuel-saving change, and predicting its effect.
---
# Exergy audit

## Ranking
- Exergy destruction is the entropy generated times ambient temperature. It is not the same as energy dissipated: a heat rejector dissipates the most energy but destroys little exergy because the heat leaves near ambient temperature. A first-law view therefore points at the wrong component.
- The combustor (gas turbine) or the engine was the largest destroyer in every plant of this dataset: burning fuel across a large temperature gap is the main irreversibility. Rank the remaining components: large temperature differences across heat exchangers (hot-water exchanger, ORC evaporator), compression and expansion inefficiency, and chillers destroy exergy; the IT load is not part of the plant.
- Use the temperatures in the summary: a component with a large gap between the hottest and coldest stream it touches, at high duty, is a candidate.

## Changes
- Allowed: set-points and loss parameters within about 20 percent of the current value (pressure ratio, combustor outlet temperature, pressure drops, ORC pinch and flow position, ORC pump pressure, fan fraction, engine air excess and loss fraction, chiller water temperature rise). Not allowed: component efficiencies, and the loop temperatures that close tear streams.
- Pick the change whose mechanism removes irreversibility in the biggest destroyer, and check that it keeps the plant valid (no temperature cross, ranges).

## Predicting the effect
- Fuel change is measured after resizing the prime mover so the net electricity is unchanged: a more efficient cycle needs less fuel for the same electricity, a parasitic increase needs more.
- Express it in MW of fuel input, negative when fuel is saved. The score is the relative error against the actual effect (with a floor of 0.2 percent of fuel), so predicting no effect scores zero.

## State bookkeeping (what is constant and what changes)
- **Recuperator / preheater**: the two streams keep their fluid identity and mass flow through the exchanger; pressure is reduced only by `dp_cold`/`dp_hot`. What changes is temperature and enthalpy: the cold side warms toward the hot inlet but never above it, and the hot side cools. A temperature cross means the specified `T_cold_out` (or duty) is impossible. Exergy destruction grows with the hot-cold approach at each end.
- **Combustor / engine**: fuel composition and LHV are fixed; mass flow grows by the fuel, pressure drops by `dp` or `P_exh`, and temperature rises to `T_out` (or follows from `P_el` and `eta_el`). The largest exergy destruction is the burning across that temperature gap.
- **HRSG / ORC evaporator**: the water/working-fluid mass flow is fixed by `P_steam` or `flow_frac`; the exhaust flow is fixed by the prime mover. Pressures are nearly constant (only `dp_hot`); temperatures and phase change. A pinch below `pinch_K` or exhaust below `T_stack_min` is infeasible.
- **Chillers and rejectors**: the refrigerant loop is closed (same fluid flows in and out), cooling-water flow is solved from the duty; only temperatures and duties change. Heat rejected near ambient carries little exergy even when its energy is large.
- Across every component: mass of each species and total energy are conserved; exergy is *not* conserved (it is destroyed), so the exergy column must fall by exactly the destruction term.

## Environment: tools, limits, final format
Inventory the skills and physics tools first, then examine the plant against known failure modes before proposing a candidate. At most 4 turns and 6 tool calls. Tools return JSON (no answer oracle): `inspect_plant` (as-given state, failed checks; call it first), `first_law_check` (mass/energy residuals, balance), `second_law_check` (exergy destruction, reversibility limits), `validity_check` (unit/range, site, pinch, verdict), `solve_plant` (stream states). Read the numbers, revise and re-check; stop when checks pass. Final answer: `<analysis>...</analysis><ranking>A > B > C</ranking><change>SET ...</change><prediction>fuel_change_mw = ...</prediction>`.
