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
