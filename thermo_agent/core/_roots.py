"""Root finding with a scipy-free fallback.

The Kaggle *benchmarks* task venv (`/benchmarks/.venv`) ships numpy but not scipy. The
teacher harness imports the physics core at module load, so a hard `from scipy.optimize
import brentq` made the whole hosted task fail before a single model call.

`brentq` here is the real scipy routine when scipy is importable (identical numerics, so
every local/GPU trace is unchanged). When scipy is absent it falls back to a Brent
implementation with the same bracketing contract and tolerance meaning, which is enough
for the tool loop: the verifiers, not this solver, decide whether an answer is accepted.
"""
from __future__ import annotations

import math

try:  # pragma: no cover - branch depends on the interpreter
    from scipy.optimize import brentq as _scipy_brentq
except ModuleNotFoundError:  # hosted benchmarks venv
    _scipy_brentq = None


def _fallback_brentq(f, a: float, b: float, xtol: float = 2e-12, rtol: float = 8.881784197001252e-16,
                     maxiter: int = 100, **_ignored) -> float:
    """Brent's method on a sign-changing bracket; raises ValueError like scipy."""
    fa, fb = f(a), f(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have different signs")
    c, fc = a, fa
    d = e = b - a
    for _ in range(maxiter):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol = 2 * rtol * abs(b) + 0.5 * xtol
        m = 0.5 * (c - b)
        if abs(m) <= tol or fb == 0.0:
            return b
        if abs(e) >= tol and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2 * m * s
                q = 1 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2 * m * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            if p > 0:
                q = -q
            p = abs(p)
            if 2 * p < min(3 * m * q - abs(tol * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = m
        else:
            d = e = m
        a, fa = b, fb
        b += d if abs(d) > tol else (tol if m > 0 else -tol)
        fb = f(b)
        if not math.isfinite(fb):
            raise ValueError("function value is not finite inside the bracket")
    raise ValueError("brentq did not converge")


brentq = _scipy_brentq or _fallback_brentq

__all__ = ["brentq", "using_scipy"]

using_scipy = _scipy_brentq is not None
