---
name: absorption-chiller
description: Use when a plant contains an absorption_chiller (single-effect, hot-water driven) or when hot-water loops and chilled-water loops appear.
components: [absorption_chiller]
---
# Absorption chiller

<!-- AUTO:components -->
### absorption_chiller
Ports: gen_in (liquid), gen_out (liquid), evap_in (liquid), evap_out (liquid), rej_in (liquid), rej_out (liquid)
Params: `T_chw_out`: chilled-water outlet temperature (K), required; `T_gen_out`: generator outlet temperature (K), required; `dT_rej`: heat-rejection water temperature rise (K), required; `eps`: fraction of the reversible COP (-), 0.27; `COP_max`: first-law COP ceiling (-), 0.8
<!-- /AUTO -->

## Behaviour
- COP = min(`eps` x reversible COP, `COP_max`). The reversible COP uses log-mean temperatures of the generator, evaporator and rejection streams. Measured single-effect LiBr/H2O units run at COP 0.55 to 0.70; `COP_max` 0.8 is the model's first-law ceiling.
- The chilled-water load fixes the generator duty (load / COP). The generator flow and the rejection flow are solved: do not specify them.
- Hot-water loop: heat exchanger `cold_out` feeds `gen_in`; `gen_out` returns to the exchanger through a tear stream whose fixed T must equal `T_gen_out`.
- Driving temperature (generator inlet) must exceed the rejection temperature, and the rejection temperature must exceed the evaporator temperature, otherwise E_TEMP_CROSS.
- A hotter rejection loop (dry cooler, hot climate) lowers the reversible COP; the cap hides this only while the reversible limit is far above it.

## Typical faults
- `eps` above 1 or `COP_max` far above 1 breaks V6 or V3 (a machine beyond the reversible limit shows negative exergy destruction).
- `T_chw_out` not below the inlet, `T_gen_out` not below the generator inlet: E_PROP_RANGE.
- `T_gen_out` edited without the tear stream's fixed T: E_NOT_CONVERGED.
- Swapped `gen_in` and `rej_in` streams: the generator sees cooling water, E_PROP_RANGE.
