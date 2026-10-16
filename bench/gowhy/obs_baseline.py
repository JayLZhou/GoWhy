"""Log-only (observational) baselines for how-to, scored against true replay effects.

Statistical debugging (Liblit et al., PLDI 2005) ranks a predicate P by
    Increase(P) = P(fail | P fired) - P(fail | P observed),
here with "observed" = every episode (the predicate can be evaluated on every step), so
Increase(P) = P(fail | F(P)) - P(fail). Importance multiplies by coverage (harmonic mean
with log support in CBI; we report Increase x coverage, which is the naive log-based
estimate of what removing P's failures would gain). Neither uses replays, so the effect
column is ignored: the same predicate gets the same score under every effect.

  python gowhy/obs_baseline.py --db store/fresh_q25.db --half B [--universe store/universe500.json]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from offline_bench import load  # noqa: E402


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--half", default="B")
    ap.add_argument("--effects", nargs="*", default=["hint:2", "replan:2", "reject:3"])
    ap.add_argument("--metric", default="won")
    ap.add_argument("--universe", default=None)
    a = ap.parse_args()
    L, cands = load(a.db, a.half, a.effects, a.metric, a.universe)
    fail = 1.0 - L.Y
    base = fail.mean()
    truth = np.array([c.truth for c in cands])
    scores = {
        "increase": np.array([fail[c.members].mean() - base for c in cands]),
        "increase x coverage": np.array([(fail[c.members].mean() - base) * c.w for c in cands]),
        "failures covered (UB)": np.array([fail[c.members].sum() / L.N for c in cands]),
    }
    best = np.sort(truth)[::-1]
    print(f"{len(cands)} candidates; true effect best {100 * best[0]:+.2f}, 10th {100 * best[9]:+.2f}, "
          f"mean {100 * truth.mean():+.2f} pts")
    for name, s in scores.items():
        order = np.argsort(-s, kind="stable")
        top = truth[order[:10]]
        print(f"  {name:22s} Spearman with truth {spearman(s, truth):+.2f}; its top-1 true {100 * top[0]:+.2f}, "
              f"top-10 mean {100 * top.mean():+.2f} pts; log estimate of its top-1 {100 * s[order[0]]:+.2f}")
    # per-predicate view: does the log score at least pick the right predicate for the best effect?
    pred_best = {}
    for c in cands:
        k = c.combo
        pred_best[k] = max(pred_best.get(k, -9), c.truth)
    keys = list(pred_best)
    tb = np.array([pred_best[k] for k in keys])
    ic = {c.combo: (fail[c.members].mean() - base) * c.w for c in cands}
    s = np.array([ic[k] for k in keys])
    o = np.argsort(-s)
    print(f"  predicates only ({len(keys)}), best effect per predicate: Spearman {spearman(s, tb):+.2f}; "
          f"top-1 by log {100 * tb[o[0]]:+.2f}, top-10 mean {100 * tb[o[:10]].mean():+.2f} pts")


if __name__ == "__main__":
    main()
