"""A simulated retail world with known ground truth.

It stands in for a real business in the demo: the engine only ever sees the
logged columns, while the simulator can answer do() questions exactly, so every
number the engine returns can be checked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NODES = [
    "segment", "loyalty", "frustration",
    "discount", "free_shipping", "support_call",
    "retained", "refund",
]

# (cause, effect, provenance). Provenance strings are part of the simulated world.
EDGES = [
    ("segment", "loyalty", "observational: CRM export"),
    ("segment", "discount", "expert: pricing policy v3"),
    ("loyalty", "discount", "expert: pricing policy v3"),
    ("loyalty", "free_shipping", "expert: logistics policy"),
    ("frustration", "support_call", "expert: support team"),
    ("frustration", "retained", "expert: support team"),
    ("segment", "retained", "observational: CRM export"),
    ("loyalty", "retained", "observational: CRM export"),
    ("discount", "retained", "experiment: ab-2025-11"),
    ("free_shipping", "retained", "experiment: ab-2026-02"),
    ("support_call", "retained", "expert: support team"),
    ("discount", "refund", "observational: finance ledger"),
    ("segment", "refund", "observational: finance ledger"),
]

# What the business logs. `frustration` exists in the graph but is not logged.
RECORDED = ["segment", "loyalty", "discount", "free_shipping", "support_call", "retained", "refund"]

ACTIONS = ["discount", "free_shipping", "support_call"]
OUTCOMES = ["retained", "refund"]


def simulate(n: int = 200_000, seed: int = 0, do: dict | None = None, segment: str | None = None) -> pd.DataFrame:
    """Draw n customers. `do` forces action variables; `segment` fixes the population."""
    do = do or {}
    u = np.random.default_rng(seed).random((8, n))

    seg_b = (u[0] < 0.40) if segment is None else np.full(n, segment == "B")
    loyalty = (u[1] < np.where(seg_b, 0.40, 0.60)).astype(int)
    frustration = (u[2] < 0.30).astype(int)

    def acted(name, natural):
        return np.full(n, int(do[name])) if name in do else natural.astype(int)

    # Logged policy: discounts go mostly to non-loyal customers, calls to frustrated ones.
    discount = acted("discount", u[3] < 0.10 + 0.50 * (1 - loyalty) + 0.10 * seg_b)
    free_shipping = acted("free_shipping", u[4] < 0.20 + 0.20 * loyalty)
    support_call = acted("support_call", u[5] < 0.10 + 0.50 * frustration)

    p_retained = (0.55 + 0.20 * loyalty + 0.05 * (~seg_b) - 0.25 * frustration
                  + 0.03 * discount + 0.02 * free_shipping + 0.04 * support_call)
    p_refund = 0.06 + 0.03 * seg_b + 0.05 * discount

    return pd.DataFrame({
        "segment": np.where(seg_b, "B", "A"),
        "loyalty": loyalty,
        "frustration": frustration,
        "discount": discount,
        "free_shipping": free_shipping,
        "support_call": support_call,
        "retained": (u[6] < p_retained).astype(int),
        "refund": (u[7] < p_refund).astype(int),
    })


def logs(n: int = 200_000, seed: int = 11, record_frustration: bool = False) -> pd.DataFrame:
    """The observational log the engine is allowed to see."""
    cols = RECORDED + (["frustration"] if record_frustration else [])
    return simulate(n, seed)[cols]


def true_effect(action: str, outcome: str, segment: str | None = None,
                n: int = 400_000, seed: int = 123) -> float:
    """Ground truth E[Y | do(action=1)] - E[Y | do(action=0)], with common random numbers."""
    on = simulate(n, seed, do={action: 1}, segment=segment)[outcome].mean()
    off = simulate(n, seed, do={action: 0}, segment=segment)[outcome].mean()
    return float(on - off)
