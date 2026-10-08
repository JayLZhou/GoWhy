"""HOWTO selection strategies, compared on recorded replays.

Each candidate has R recorded replays per affected game (seeded reps 100..). For
every way of splitting the reps into a selection part and a held-out part, run a
selection strategy on the selection reps only, then score its pick on the held-out
reps. Held-out scoring is independent of selection, so it is free of the winner's
curse.

Strategies
  uniform-1   one replay per affected game for every candidate, pick the best estimate
  uniform-2   two replays per affected game for every candidate
  elim        successive elimination: one replay for all; drop candidates whose upper
              bound (est + z*SE) is below the best lower bound; a second replay only for
              the survivors; pick the best estimate

Reported per strategy: mean cost (model calls), mean held-out effect of the pick, and
how often the pick is the held-out best candidate.

  python gowhy/optimizer.py --db store/fresh.db --cands gowhy/pool_qwen3.txt --half B
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import sqlite3
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from changes import Change  # noqa: E402
from engine import half_of  # noqa: E402


def load(db, cands, half, rep0, R):
    con = sqlite3.connect(db)
    Y = {t: w for t, w in con.execute("SELECT tid, won FROM traces") if not half or half_of(t) == half}
    data = {}
    for text in cands:
        ch = Change.parse(text)
        aff = {t for t, _ in con.execute(ch.first_match_sql()) if t in Y}
        runs = defaultdict(dict)
        for t, rep, w, c in con.execute("SELECT tid, rep, won, calls FROM runs WHERE change=? AND k0>=0", (ch.text,)):
            if t in aff and rep0 <= rep < rep0 + R:
                runs[t][rep - rep0] = (w, c)
        complete = all(len(runs[t]) == R for t in aff)
        data[ch.text] = (aff, runs, complete)
    return Y, data


def estimate(Y, aff, runs, reps):
    """Deployment-basis effect and SE using only the given reps; also model calls spent."""
    N = len(Y)
    D, calls = [], 0
    for t in Y:
        if t in aff:
            w = [runs[t][r][0] for r in reps]
            calls += sum(runs[t][r][1] for r in reps)
            D.append(sum(w) / len(w) - Y[t])
        else:
            D.append(0.0)
    m = sum(D) / N
    se = math.sqrt(sum((d - m) ** 2 for d in D) / (N - 1) / N)
    return m, se, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--cands", required=True)
    ap.add_argument("--half", default="B")
    ap.add_argument("--rep0", type=int, default=100)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--z", type=float, default=1.0)
    a = ap.parse_args()
    cands = [l.strip() for l in open(a.cands) if l.startswith("UPDATE")]
    Y, data = load(a.db, cands, a.half, a.rep0, a.R)
    ok = {c: v for c, v in data.items() if v[2]}
    print(f"{len(ok)}/{len(data)} candidates have all {a.R} reps; using those", file=sys.stderr)

    # full-data view, for the record
    full = {c: estimate(Y, aff, runs, list(range(a.R)))[:2] for c, (aff, runs, _) in ok.items()}
    print("all-rep estimates (top 8):")
    for c, (m, s) in sorted(full.items(), key=lambda x: -x[1][0])[:8]:
        print(f"  {100 * m:+5.1f} (SE {100 * s:.1f})  {c.replace('UPDATE events SET action = ', '')[:120]}")

    res = defaultdict(lambda: {"calls": [], "held": [], "hit": []})
    splits = [s for s in itertools.combinations(range(a.R), 2)]
    for sel in splits:
        held = [r for r in range(a.R) if r not in sel]
        hv = {c: estimate(Y, aff, runs, held)[0] for c, (aff, runs, _) in ok.items()}
        best_h = max(hv.values())

        def record(name, pick, calls):
            res[name]["calls"].append(calls)
            res[name]["held"].append(hv[pick])
            res[name]["hit"].append(hv[pick] >= best_h - 1e-12)

        for n, name in [(1, "uniform-1"), (2, "uniform-2")]:
            est = {c: estimate(Y, aff, runs, list(sel[:n])) for c, (aff, runs, _) in ok.items()}
            record(name, max(est, key=lambda c: est[c][0]), sum(e[2] for e in est.values()))

        r1 = {c: estimate(Y, aff, runs, [sel[0]]) for c, (aff, runs, _) in ok.items()}
        best_lcb = max(m - a.z * s for m, s, _ in r1.values())
        alive = [c for c, (m, s, _) in r1.items() if m + a.z * s >= best_lcb]
        r2 = {c: estimate(Y, *ok[c][:2], list(sel)) for c in alive}
        record("elim", max(r2, key=lambda c: r2[c][0]),
               sum(e[2] for e in r1.values()) + sum(r2[c][2] - r1[c][2] for c in alive))
        res["elim"].setdefault("alive", []).append(len(alive))

    print(f"\n{len(splits)} selection/held-out splits, {len(ok)} candidates, N = {len(Y)} games")
    print("strategy     model calls   held-out effect of pick   pick = held-out best")
    for name in ["uniform-1", "uniform-2", "elim"]:
        r = res[name]
        extra = f"   (survivors after round 1: {sum(r['alive']) / len(r['alive']):.1f})" if "alive" in r else ""
        print(f"  {name:10s} {sum(r['calls']) / len(r['calls']):10.0f}   {100 * sum(r['held']) / len(r['held']):+8.1f} pts"
              f"              {sum(r['hit'])}/{len(r['hit'])}{extra}")


if __name__ == "__main__":
    main()
