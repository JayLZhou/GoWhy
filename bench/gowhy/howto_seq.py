"""Sequential HOWTO with ARP: does each new candidate get cheaper as more are evaluated?

Candidates are evaluated one after another in a random order. Every replay drawn
for an earlier candidate is added to the per-task pool of "other runs"; ARP uses
the task success rates from that pool (plus extra base runs) to allocate the
budget of the next candidate. Uniform sampling spends the same budget evenly.

Offline: replays are drawn from the cached seeded replays (reps 100..103), and each
candidate's truth is the mean over all four.

  python gowhy/howto_seq.py --db store/fresh.db --cands logs/opt_cands.txt --half B \
      --extra-runs data/fresh/Qwen3-8B-fresh.rep2.jsonl --budget 0.3 --trials 40
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arp  # noqa: E402
from changes import Change  # noqa: E402
from engine import half_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--cands", required=True)
    ap.add_argument("--half", default="B")
    ap.add_argument("--extra-runs", default=None)
    ap.add_argument("--budget", type=float, default=0.3)
    ap.add_argument("--trials", type=int, default=40)
    a = ap.parse_args()

    st = arp.Store(a.db)
    keep = {t for t in st.tr if half_of(t) == a.half}
    st.tr = {t: v for t, v in st.tr.items() if t in keep}
    tids = sorted(st.tr)
    rng0 = random.Random(0)
    sh = tids[:]
    rng0.shuffle(sh)
    folds = {t: i % 2 for i, t in enumerate(sh)}
    models = arp.base_models(st, folds, "won")

    cands = []
    for text in [l.strip() for l in open(a.cands) if l.startswith("UPDATE")]:
        ch = Change.parse(text)
        k0 = {t: k for t, k in st.con.execute(ch.first_match_sql()) if t in keep}
        cache = collections.defaultdict(list)
        for t, w, p, c in st.con.execute(
                "SELECT tid, won, prompt_tokens, completion_tokens FROM runs "
                "WHERE change=? AND k0>=0 AND rep BETWEEN 100 AND 103", (ch.text,)):
            if t in k0:
                cache[t].append((float(w), p + c))
        if not k0 or any(len(cache[t]) < 4 for t in k0):
            continue
        truth = sum(sum(v for v, _ in cache[t]) / 4 - st.tr[t]["won"] for t in k0) / len(tids)
        full = sum(sum(c for _, c in cache[t]) / 4 for t in k0)
        steps = sum(st.H - k0[t] for t in k0)
        cands.append({"text": ch.text, "k0": k0, "cache": cache, "truth": truth, "full": full,
                      "cps": full / steps})
    print(f"{len(cands)} complete candidates, N = {len(tids)} games, budget {a.budget} of each candidate's v1 cost",
          flush=True)

    base_extra = collections.defaultdict(list)
    if a.extra_runs:
        for line in open(a.extra_runs):
            d = json.loads(line)
            if d["tid"] in keep:
                base_extra[d["tid"]].append(float(d["won"]))

    by_pos = collections.defaultdict(lambda: collections.defaultdict(list))   # method -> position -> sq err
    cost = collections.defaultdict(float)
    picks = collections.defaultdict(int)
    best_truth = max(c["truth"] for c in cands)
    for trial in range(a.trials):
        rng = random.Random(100 + trial)
        order = cands[:]
        rng.shuffle(order)
        for method in ("uniform", "arpdr"):
            pool = collections.defaultdict(list, {t: v[:] for t, v in base_extra.items()})
            est = {}
            for pos, c in enumerate(order):
                rates = {t: (sum(v) / len(v), len(v)) for t, v in pool.items() if v}
                drawn = []
                r2 = random.Random(hash((trial, method, pos)) & 0xffffffff)

                def replay(t):
                    v, cc = r2.choice(c["cache"][t])
                    drawn.append((t, v))
                    return v, cc

                m, se, spent, _ = arp.plan_and_estimate(
                    st, set(c["k0"]), c["k0"], replay, a.budget * c["full"], method, 0.0, r2, "won",
                    c["cps"], models, folds, rates=rates)
                by_pos[method][pos].append((m - c["truth"]) ** 2)
                cost[method] += spent / c["full"]
                est[c["text"]] = m
                for t, v in drawn:
                    pool[t].append(v)
            pick = max(est, key=est.get)
            picks[method] += int(next(x["truth"] for x in cands if x["text"] == pick) >= best_truth - 1e-12)

    K = len(cands)
    print("position of candidate in evaluation order -> RMSE (points)")
    for lo, hi in [(0, 5), (5, 15), (15, K)]:
        line = f"  positions {lo + 1:2d}-{hi:2d}:"
        for method in ("uniform", "arpdr"):
            errs = [e for p in range(lo, hi) for e in by_pos[method][p]]
            line += f"  {method} {100 * math.sqrt(sum(errs) / len(errs)):.2f}"
        print(line)
    for method in ("uniform", "arpdr"):
        allerr = [e for p in by_pos[method] for e in by_pos[method][p]]
        print(f"  {method:8s} overall RMSE {100 * math.sqrt(sum(allerr) / len(allerr)):.2f}  "
              f"mean spend {cost[method] / (a.trials * K):.2f}  picks the best {picks[method]}/{a.trials}")


if __name__ == "__main__":
    main()
