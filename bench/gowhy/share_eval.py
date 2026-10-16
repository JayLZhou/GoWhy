"""Lattice-Share check: evaluate many candidates with shared replays, measure cost and bias.

Candidates whose first firing step on an episode coincides and whose effect is the same
fall into one cell (episode, fork step, effect). Within a cell, candidates are replayed one
after another; each picks as donor the earlier replay in the cell whose logged firing
decisions agree with its own predicate for the longest prefix, and replays by walking that
donor (engine.replay_one with donor=...), paying model calls only after a divergence.

Two numbers decide whether sharing is worth it:
  cost   model calls with sharing vs. the same (candidate, episode) pulls replayed
         independently (the cached search-seed replays, rep 600);
  bias   per-candidate effect from shared replays vs. from independent replays: both are
         unbiased for the same effect, so their differences should average to zero.

  python gowhy/share_eval.py --db store/fresh_q25.db --half B --universe store/universe500.json \
      --K 100 --rep 900 --out store/share_eval.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from changes import Change  # noqa: E402
from engine import Engine, replay_one, unpack_steps  # noqa: E402
from lattice import Lattice  # noqa: E402

INDEP_REP = 600


def prefix_agree(ch: Change, steps: list) -> int:
    """Number of leading donor steps on which ch's firing decision equals the donor's."""
    n = 0
    for st in steps:
        if bool(ch.matches(st["attrs"])) != bool(st["fired"]):
            break
        n += 1
    return n


def run_cell(base, trace, events, k0, rep, texts):
    """Replay every change of one cell on one episode, sharing along a donor chain."""
    out = []
    logs = []
    for text in texts:
        ch = Change.parse(text)
        donor = None
        if logs:
            donor = max(logs, key=lambda st: prefix_agree(ch, st))
        r = replay_one(text, base, trace, events, k0, rep, donor=donor, log_steps=True)
        logs.append(unpack_steps(r["steps"]))
        r["shared"] = donor is not None
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--half", default="B")
    ap.add_argument("--base", default="qwen2.5-7b")
    ap.add_argument("--universe", required=True)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rep", type=int, default=900)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    L = Lattice(a.db, a.half, 3, "won")
    with open(a.universe) as f:
        arms = json.load(f)["arms"]
    arms = random.Random(a.seed).sample(arms, min(a.K, len(arms)))
    texts = [Change.parse(f"UPDATE events SET action = {x['effect']} WHERE "
                          f"{L.predicate(tuple(x['combo']))} SCOPE ALL").text for x in arms]
    eng = Engine(a.db, a.base, a.workers)
    traces = eng.traces()
    half = set(L.tids)
    cells = collections.defaultdict(list)          # (tid, k0, effect) -> [change text]
    members = {}
    for text in texts:
        k0 = {t: k for t, k in eng.con.execute(Change.parse(text).first_match_sql()) if t in half}
        members[text] = k0
        eff = Change.parse(text).effect + ":" + Change.parse(text).arg
        for t, k in k0.items():
            cells[(t, k, eff)].append(text)
    n_pulls = sum(len(v) for v in members.values())
    print(f"{len(texts)} candidates, {n_pulls} pulls, {len(cells)} cells "
          f"({n_pulls / len(cells):.2f} pulls per cell)", flush=True)

    # independent baseline: cached rep-600 replays of the same pulls
    indep = {}
    for text in texts:
        for t, won, calls in eng.con.execute("SELECT tid, won, calls FROM runs WHERE change=? AND rep=? AND k0>=0",
                                              (text, INDEP_REP)):
            indep[(text, t)] = (float(won), calls)
    missing = sum(1 for text in texts for t in members[text] if (text, t) not in indep)
    print(f"independent baseline: {n_pulls - missing} of {n_pulls} pulls cached", flush=True)

    t0 = time.time()
    res = {}
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {}
        for (t, k, eff), tx in cells.items():
            futs[ex.submit(run_cell, a.base, traces[t], eng.events(t), k, a.rep, sorted(tx))] = (t, k, eff)
        for fu in as_completed(futs):
            try:
                rows = fu.result()
            except Exception as e:
                print("cell failed:", repr(e)[:200], flush=True)
                continue
            for r in rows:
                eng._insert_run(r)
                res[(r["change"], r["tid"])] = r
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(cells)} cells {int(time.time() - t0)}s", flush=True)

    # cost
    calls_share = sum(r["calls"] for r in res.values())
    keys = [k for k in res if k in indep]
    calls_indep = sum(indep[k][1] for k in keys)
    calls_share_k = sum(res[k]["calls"] for k in keys)
    free = sum(1 for r in res.values() if r["calls"] == 0)
    # bias: per candidate, effect from shared replays minus effect from independent replays
    N = L.N
    Y = dict(zip(L.tids, L.Y))
    diffs = []
    for text in texts:
        ks = [(text, t) for t in members[text] if (text, t) in res and (text, t) in indep]
        if len(ks) < len(members[text]):
            continue
        d_share = sum(res[k]["won"] - Y[k[1]] for k in ks) / N
        d_ind = sum(indep[k][0] - Y[k[1]] for k in ks) / N
        diffs.append(d_share - d_ind)
    m = sum(diffs) / len(diffs)
    se = math.sqrt(sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1) / len(diffs))
    pw_share = sum(res[k]["won"] for k in keys) / len(keys)
    pw_ind = sum(indep[k][0] for k in keys) / len(keys)
    summ = {"candidates": len(texts), "pulls": n_pulls, "cells": len(cells), "replayed": len(res),
            "calls_shared": calls_share, "calls_shared_on_matched": calls_share_k,
            "calls_independent_on_matched": calls_indep,
            "saving": calls_indep / max(1, calls_share_k), "free_replays": free,
            "won_rate_shared": pw_share, "won_rate_independent": pw_ind,
            "effect_diff_mean": m, "effect_diff_se": se, "n_candidates_compared": len(diffs),
            "secs": int(time.time() - t0)}
    print(json.dumps(summ), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(summ) + "\n")


if __name__ == "__main__":
    main()
