---
name: design-catalog
description: Use for Design tasks: choosing prime mover, cooling, heat rejection, bottoming cycle and district-heat source for a site.
---
# Design catalog

Objective: among plants that solve, pass all checks and meet the site, burn the least fuel. Mechanisms below hold at the default technology constants of these plants (engine electrical efficiency 0.43, chiller COP cap 0.8); they are not laws.

- Prime mover: an engine converts more fuel to electricity, but only its exhaust is recovered. A turbine has hotter, larger exhaust and can drive absorption cooling for a larger chilled share. If the chilled load is too large for engine exhaust, the exhaust heat exchanger fails (temperature cross).
- Cooling: absorption cooling uses exhaust heat and adds no electrical load; electric cooling adds parasitic electricity, so a larger prime mover. Absorption needs enough exhaust heat at a high enough temperature.
- Rejection: a wet tower rejects at wet-bulb plus 5 K, a dry cooler at dry-bulb plus 8 K. Hot, dry climates hurt dry coolers most (lower chiller COP, possible temperature cross). A site water limit rules out wet towers.
- Bottoming cycle: an ORC makes electricity from exhaust that is left after other recoveries, so it lowers fuel. Exhaust above the fluid's vapour limit cannot be used (about 850 K turbine exhaust is too hot for all three fluids here); choose the fluid whose limit fits the exhaust temperature after upstream recovery.
- District heat: exhaust heat is free if enough remains after cooling; a heat pump costs electricity but leaves exhaust for an ORC.
- Continuous parameters: turbine PR and T4 trade cycle efficiency against exhaust temperature; hot-water `T_hw` and `dT_hw` must keep the generator duty feasible; `dT_rej` sets rejection flow and temperature level; `P_orc_bar` and `orc_frac` must keep the pinch and the exhaust floor feasible.
- Check the feasibility mechanisms (exhaust heat available vs duty, temperature levels, water) before optimising fuel.
