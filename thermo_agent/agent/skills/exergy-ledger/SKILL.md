---
name: exergy-ledger
description: Use for ledger-repair tasks: the stream exergy lines E[x], component destruction lines ED[c], the constant T0, and how to find and fix a wrong line.
---
# Exergy ledger

## Correct forms (MW)
- `T0 = <ambient temperature in K>`; the AMBIENT line of the task gives it. Never 25 or 273.15 for an ambient of about 298 K.
- Stream line: `E[x] = m[x]*(h[x] - T0*s[x] - g0[x] + ech[x])/1e6`. `g0[x]` is the dead-state reference of the stream's fluid, `ech[x]` its chemical exergy (zero for liquids and ambient air). All stream lines share this template.
- Component line: `ED[c] = E[inlet streams] - E[outlet streams] - W[c]`, with `+` for every inlet stream and `-` for every outlet stream listed in TOPOLOGY. The `- W[c]` term exists only for components that appear in the W list (compressor, turbine, pump, engine, IT load, electric chiller, heat pump, heat rejector fans).

## Finding the wrong line
- A negative `ED` is impossible: its line, or an `E` line it uses, is wrong.
- Closure: the sum of all `ED` must equal boundary exergy in minus boundary exergy out minus net work. Internal streams appear once with each sign and cancel, so an error in an internal stream's `E` line does not show in the closure residual; an error in a boundary stream, a work term, or a component line usually does.
- Compare each line with its siblings. A stream line that differs from the template (missing `m[x]`, a missing `ech[x]`, `/1e3`, `+ T0*s[x]`, the number 25, another stream's `g0`) is wrong. A component line that differs from in minus out minus work (a flipped sign, a missing or extra stream, a wrong stream, a numeric factor such as 1000 or 0.1) is wrong.
- If every `E` is off, check `T0` first.

## Fixing
- Write the whole corrected expression: `FIX ED[HX] = E[s5] + E[hw_ret] - E[s6] - E[hw_hot]`. Fix only lines that are wrong; changing a correct line is penalised.
- After your fixes the ledger must close, every `ED` must be non-negative, and every fixed line must equal its true value.
