"""Build an unbiased offline benchmark: sample K candidates uniformly from the lattice and
replay every affected episode of each on the search seed (rep 600), reusing the cache.

Writes the sampled candidates to --out (JSON); offline_bench.py --universe reads it.

  python gowhy/fill_universe.py --db store/fresh_q25.db --half B --K 500 --out store/universe500.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from changes import Change  # noqa: E402
from engine import Engine, replay_one  # noqa: E402
from lattice import Lattice  # noqa: E402

REP = 600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--half", default="B")
    ap.add_argument("--base", default="qwen2.5-7b")
    ap.add_argument("--effects", nargs="*", default=["hint:2", "replan:2", "reject:3"])
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    L = Lattice(a.db, a.half, 3, "won")
    best = {}
    for cb, F in L.enumerate().items():
        k = F.tobytes()
        if k not in best or len(cb) < len(best[k][0]):
            best[k] = (cb, F)
    arms = [(cb, e) for cb, _ in sorted(best.values(), key=lambda v: v[0]) for e in a.effects]
    sample = random.Random(a.seed).sample(arms, min(a.K, len(arms)))
    with open(a.out, "w") as f:
        json.dump({"db": a.db, "half": a.half, "rep": REP, "effects": a.effects,
                   "arms": [{"combo": list(cb), "effect": e} for cb, e in sample]}, f)

    eng = Engine(a.db, a.base, a.workers)
    traces = eng.traces()
    todo = []
    for cb, e in sample:
        text = Change.parse(f"UPDATE events SET action = {e} WHERE {L.predicate(cb)} SCOPE ALL").text
        have = {t for (t,) in eng.con.execute("SELECT tid FROM runs WHERE change=? AND rep=? AND k0>=0",
                                               (text, REP))}
        k0 = dict(eng.con.execute(Change.parse(text).first_match_sql()).fetchall())
        todo += [(text, t, k0[t]) for t in L.tids if t in k0 and t not in have]
    print(f"{len(sample)} candidates, {len(todo)} replays to run", flush=True)
    evs = {}
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(a.workers) as pool:
        futs = {}
        for text, t, k in todo:
            if t not in evs:
                evs[t] = eng.events(t)
            futs[pool.submit(replay_one, text, a.base, traces[t], evs[t], k, REP)] = (text, t)
        for fu in as_completed(futs):
            try:
                eng._insert_run(fu.result())
            except Exception as ex:
                print("replay failed:", repr(ex)[:200], flush=True)
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(todo)} {int(time.time() - t0)}s", flush=True)
    print("FILL_DONE", flush=True)


if __name__ == "__main__":
    main()
