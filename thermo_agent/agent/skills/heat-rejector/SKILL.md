---
name: heat-rejector
description: Use when a plant contains a heat_rejector (dry cooler or evaporative cooling tower) closing a cooling-water loop.
components: [heat_rejector]
---
# Heat rejector

<!-- AUTO:components -->
### heat_rejector
Ports: in (liquid), out (liquid)
Params: `kind`: 'dry' (dry cooler) or 'wet' (evaporative tower), dry; `approach_K`: approach to dry-bulb or wet-bulb (K), required; `fan_frac`: fan electricity as a fraction of heat rejected (-), 0.015
<!-- /AUTO -->

## Behaviour
- Outlet temperature = dry-bulb (`kind` dry) or wet-bulb (`kind` wet) plus `approach_K`. Wet-bulb comes from ambient temperature and relative humidity, so a tower cools colder than a dry cooler, most of all in dry air.
- Fans use `fan_frac` of the heat rejected as electricity (a parasitic load).
- A wet tower consumes make-up water. The model reports an upper bound: all heat rejected as latent heat. A site with a water limit rules wet towers out (V8e).
- The loop is closed by a tear stream: the stream leaving the rejector carries `fixed` T and P, and its T must equal the rejector outlet temperature. Changing `approach_K` or `kind` without changing that fixed T gives E_NOT_CONVERGED.
- The inlet must be warmer than the achievable outlet, otherwise E_TEMP_CROSS.
- Heat leaves the system at ambient: it counts as exergy destroyed in the rejector although it carries little exergy.

## Typical faults
- `kind` not `dry` or `wet`, `approach_K` of 0 or negative, `fan_frac` above 0.1.

## State: constant versus changing
Fixed: water flow and ambient T/P; outlet T = wet/dry bulb + `approach_K`, fan uses `fan_frac` of rejected heat, and the tear stream's fixed T must match. Heat leaves near ambient: large duty, small exergy loss.
