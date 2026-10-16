"""Lattice-LUCB: best-arm identification over the predicate lattice (HOWTO at scale).

Arm      a candidate change (predicate phi, effect e), one per distinct affected set F
         (shortest predicate kept) and effect.
Pull     replay one episode of F, drawn without replacement, under the change, with the
         common seed rep (same seed for every arm on the same episode); observe
         x = Vhat - Y in [-1, 1].
Mean     Delta(arm) = (|F|/N) * mean_{i in F}(V_i - Y_i); estimated by (|F|/N) * mean(x).
Range    every arm's mean lies in [-|F & succ|/N, |F & fail|/N] (lattice bounds), known
         before any pull; unpulled arms use the upper end as their UCB, so the search
         expands the lattice best-first by bound.
Radius   Hoeffding-Serfling for sampling without replacement, scaled by |F|/N:
         rad = (|F|/N) * 2 * sqrt((1 - (t-1)/|F|) * log(c K t^2 / delta) / (2 t)),
         intersected with the lattice range.
Rounds   batched LUCB: each round pulls the empirical best h and the challengers with the
         highest UCB, in one parallel batch; stop when LCB_h >= max_{j != h} UCB_j - eps
         or the budget is spent.

Baselines (same pull cache, same seeds): "ub-order" evaluates arms fully (every affected
episode once) in decreasing upper bound until the budget is spent, and picks the best.

  python gowhy/lattice_bai.py --db store/fresh_q25.db --half B --budget 3000 --batch 96
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import Engine, replay_one  # noqa: E402
from lattice import Lattice  # noqa: E402

REP = 600          # common seed for every first pull of an (arm, episode)


class Arm:
    def __init__(self, idx, pred, effect, F, N, Y, tids):
        self.idx, self.pred, self.effect = idx, pred, effect
        self.text = f"UPDATE events SET action = {effect} WHERE {pred} SCOPE ALL"
        self.members = [tids[i] for i in range(len(F)) if F[i]]
        self.n = len(self.members)
        self.w = self.n / N
        self.lo = -float(sum(Y[i] for i in range(len(F)) if F[i])) / N        # outcomes in [0, 1]
        self.hi = float(sum(1.0 - Y[i] for i in range(len(F)) if F[i])) / N
        self.x = {}                                    # tid -> Vhat - Y
        self.order = None

    def mean(self):
        return self.w * sum(self.x.values()) / len(self.x) if self.x else None

    def rad(self, K, delta, bound="hoeffding"):
        t = len(self.x)
        if t == 0:
            return float("inf")
        fpc = 1.0 if t >= self.n else (1 - (t - 1) / self.n)   # finite-population correction
        lg = math.log(4 * K * t * t / delta)
        if bound == "bernstein":                       # empirical Bernstein (Maurer and Pontil), range 2
            xs = list(self.x.values())
            m = sum(xs) / t
            var = sum((v - m) ** 2 for v in xs) / max(1, t - 1)
            r = math.sqrt(2 * var * lg / t) + 7 * 2 * lg / (3 * max(1, t - 1))
        else:                                          # Hoeffding, range 2
            r = 2 * math.sqrt(lg / (2 * t))
        if t >= self.n:                                # all episodes pulled: replay noise only (range 1)
            r = min(r, math.sqrt(math.log(4 * K / delta) / (2 * t)))
        return self.w * r * math.sqrt(fpc)

    BOUND = "hoeffding"

    def ucb(self, K, delta):
        m = self.mean()
        return self.hi if m is None else min(self.hi, m + self.rad(K, delta, Arm.BOUND))

    def lcb(self, K, delta):
        m = self.mean()
        return self.lo if m is None else max(self.lo, m - self.rad(K, delta, Arm.BOUND))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--half", default="B")
    ap.add_argument("--base", default=None)
    ap.add_argument("--effects", nargs="*", default=["hint:2", "replan:2", "reject:3"])
    ap.add_argument("--max-atoms", type=int, default=3)
    ap.add_argument("--budget", type=int, default=3000, help="max pulls (replays)")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.005)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--method", choices=["lucb", "ub-order"], default="lucb")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bound", choices=["hoeffding", "bernstein"], default="hoeffding")
    a = ap.parse_args()
    Arm.BOUND = a.bound
    base = a.base or {"fresh.db": "qwen3-8b", "fresh_q25.db": "qwen2.5-7b", "ws_q3.db": "qwen3-8b"}[
        os.path.basename(a.db)]
    out = a.out or os.path.join(os.path.dirname(a.db), f"lattice_{a.method}_{a.bound}_{os.path.basename(a.db)}.jsonl")

    L = Lattice(a.db, a.half, a.max_atoms)
    preds = L.enumerate()
    best_pred = {}
    for c, F in preds.items():                         # one predicate per distinct affected set
        key = F.tobytes()
        if key not in best_pred or len(c) < len(best_pred[key][0]):
            best_pred[key] = (c, F)
    arms = []
    for c, F in best_pred.values():
        for e in a.effects:
            arms.append(Arm(len(arms), L.predicate(c), e, F, L.N, L.Y, L.tids))
    K = len(arms)
    rng = random.Random(0)
    for arm in arms:
        arm.order = arm.members[:]
        rng.shuffle(arm.order)

    eng = Engine(a.db, base, a.workers)
    traces = eng.traces()
    events = {}

    def ev(tid):
        if tid not in events:
            events[tid] = eng.events(tid)
        return events[tid]

    from changes import Change
    k0_cache = {}

    def k0_of(arm, tid):
        key = (arm.idx, tid)
        if key not in k0_cache:
            ch = Change.parse(arm.text)
            for t, k in eng.con.execute(ch.first_match_sql() + "", ()):
                k0_cache[(arm.idx, t)] = k
        return k0_cache[key]

    def cached(arm, tid):
        rows = eng.con.execute("SELECT won FROM runs WHERE change=? AND tid=? AND rep=? AND k0>=0",
                               (Change.parse(arm.text).text, tid, REP)).fetchall()
        return None if not rows else rows[0][0]

    Y = dict(zip(L.tids, L.Y))
    pulls = 0
    t0 = time.time()
    pool = ProcessPoolExecutor(a.workers)
    from concurrent.futures import FIRST_COMPLETED, wait
    inflight = {}                                      # future -> (arm, tid)
    busy = {}                                          # arm idx -> set of tids in flight

    def next_free(arm):
        b = busy.get(arm.idx, set())
        for t in arm.order:
            if t not in arm.x and t not in b:
                return t
        return None

    def take_cached(arm, tid):
        w = cached(arm, tid)
        if w is None:
            return False
        arm.x[tid] = float(w) - Y[tid]
        return True

    def submit(arm, tid):
        fu = pool.submit(replay_one, Change.parse(arm.text).text, base, traces[tid], ev(tid),
                         k0_of(arm, tid), REP)
        inflight[fu] = (arm, tid)
        busy.setdefault(arm.idx, set()).add(tid)

    def ranking():
        if a.method == "ub-order":
            return sorted((x for x in arms if len(x.x) < x.n), key=lambda x: -x.hi)
        pulled = [x for x in arms if x.x]
        h = max(pulled, key=lambda x: x.mean()) if pulled else max(arms, key=lambda x: x.hi)
        others = sorted((x for x in arms if x is not h), key=lambda x: -x.ucb(K, a.delta))
        return [h] + others

    def stop_now():
        if a.method != "lucb":
            return False
        r = ranking()
        return r[0].lcb(K, a.delta) >= r[1].ucb(K, a.delta) - a.eps

    def fill():
        nonlocal pulls
        share = max(4, a.batch // 8) if a.method == "lucb" else 10 ** 9
        while len(inflight) < a.batch and pulls + len(inflight) < a.budget:
            progressed = False
            for arm in ranking()[: (8 if a.method == "lucb" else 50)]:
                if len(inflight) >= a.batch or pulls + len(inflight) >= a.budget:
                    break
                if len(busy.get(arm.idx, ())) >= share:
                    continue
                t = next_free(arm)
                while t is not None and take_cached(arm, t):
                    t = next_free(arm)
                if t is None:
                    continue
                submit(arm, t)
                progressed = True
            if not progressed:
                break

    log = open(out, "a")
    done_since_log = 0
    fill()
    while inflight:
        fin, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
        for fu in fin:
            arm, tid = inflight.pop(fu)
            busy[arm.idx].discard(tid)
            try:
                r = fu.result()
                eng._insert_run(r)
                arm.x[tid] = float(r["won"]) - Y[tid]
                pulls += 1
                done_since_log += 1
            except Exception as e:
                print("pull failed:", repr(e)[:200], flush=True)
        if done_since_log >= 50:
            done_since_log = 0
            pulled = [x for x in arms if x.x]
            h = max(pulled, key=lambda x: x.mean())
            msg = {"pulls": pulls, "secs": int(time.time() - t0), "arms_touched": len(pulled),
                   "best": h.text, "best_mean": h.mean(), "best_n": len(h.x), "best_size": h.n,
                   "best_lcb": h.lcb(K, a.delta), "max_other_ucb": ranking()[1].ucb(K, a.delta)}
            print(json.dumps(msg), flush=True)
            log.write(json.dumps(msg) + "\n")
            log.flush()
        if stop_now():
            print("stopped by the LUCB rule", flush=True)
            for fu in inflight:
                fu.cancel()
            break
        fill()
    pool.shutdown()
    pulled = [x for x in arms if x.x]
    top = sorted(pulled, key=lambda x: -x.mean())[:10]
    print(f"\n{a.method}: K = {K} arms, pulls {pulls}, arms touched {len(pulled)}")
    for x in top:
        print(f"  {100 * x.mean():+6.2f} pts  pulled {len(x.x)}/{x.n}  [{100 * x.lcb(K, a.delta):+.1f}, "
              f"{100 * x.ucb(K, a.delta):+.1f}]  {x.text[27:]}")
    log.write(json.dumps({"final": True, "method": a.method, "pulls": pulls, "K": K,
                          "top": [(x.text, x.mean(), len(x.x), x.n) for x in top]}) + "\n")


if __name__ == "__main__":
    main()
