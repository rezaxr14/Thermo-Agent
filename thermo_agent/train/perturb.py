"""Perturbations of answer text, shared by the robustness tests and the reward diagnostics."""
from __future__ import annotations
import random, re

EXTREMES = ["1e308", "-1e308", "1e-308", "0", "-1", "1e999", "0.0", "123456789012345678901234567890", "1e-9", "5e5", "1e12"]
_NUM = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def mutate(text: str, rng: random.Random) -> str:
    """One to three random damages: a number replaced by an extreme one, truncation, duplication, a deleted slice, a broken `=`."""
    t = text
    for _ in range(rng.randint(1, 3)):
        k = rng.random(); nums = list(_NUM.finditer(t))
        if k < 0.45 and nums:
            m = rng.choice(nums); t = t[:m.start()] + rng.choice(EXTREMES) + t[m.end():]
        elif k < 0.6: t = t[: rng.randint(0, len(t))]
        elif k < 0.75: t = t + "\n" + t
        elif k < 0.9 and len(t) > 4:
            i = rng.randint(0, len(t) - 2); t = t[:i] + t[i + rng.randint(1, 30):]
        else: t = t.replace("=", rng.choice(["", "==", " := "]), 1)
    return t


def simulate_policy(expert: str, rng: random.Random, level: float) -> str:
    """A policy of quality 1 - level: the expert answer with probability 1 - level, otherwise a damaged answer, rarely plain noise."""
    if rng.random() >= level: return expert
    return "I am not sure." if rng.random() < 0.1 else mutate(expert, rng)
