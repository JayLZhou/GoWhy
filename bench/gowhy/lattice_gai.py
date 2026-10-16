"""Good-arm identification over the predicate lattice, with independent certification.

Goal: return a change whose effect is certified to exceed tau, i.e.
    P(returned change has Delta < tau) <= delta,
using as few replays as possible, and report the best change found so far.

Search (uses seed rep 600 for every first pull, shared with earlier runs):
  sh        sequential halving on the M arms with the highest lattice upper bound:
            pull a few episodes of each, keep the best quarter, pull more, keep the best
            quarter, then pull every episode of the survivors
  ub-order  evaluate arms fully, one after another, in decreasing upper bound
  random    evaluate arms fully in random order (among arms whose upper bound >= tau)
Certification (fresh seeds rep 700 + attempt): replay every affected episode of the
search winner once, deployment-basis estimate and SE over all N episodes, one-sided
lower bound at level delta / m. If it clears tau, return it; otherwise certify the next
best, at most m attempts. Only the m certifications need a union bound, not all arms.

  python gowhy/lattice_gai.py --db store/fresh_q25.db --half B --method sh --search-budget 1950
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from statistics import NormalDist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from changes import Change  # noqa: E402
from engine import Engine, replay_one  # noqa: E402
from lattice import Lattice  # noqa: E402
from lattice_bai import Arm  # noqa: E402

SEARCH_REP = 600
CERT_REP = 700


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--half", default="B")
    ap.add_argument("--base", default=None)
    ap.add_argument("--effects", nargs="*", default=["hint:2", "replan:2", "reject:3"])
    ap.add_argument("--method", choices=["sh", "ub-order", "random", "lin"], required=True)
    ap.add_argument("--search-budget", type=int, default=1950)
    ap.add_argument("--M", type=int, default=96)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--m", type=int, default=3, help="max certification attempts")
    ap.add_argument("--workers", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--metric", choices=["won", "score"], default="won")
    ap.add_argument("--prefix", default="", help="patch already applied to the store (Counterfactual Store "
                                                 "Iteration): candidates are appended to it")
    ap.add_argument("--search-rep", type=int, default=SEARCH_REP)
    ap.add_argument("--extra", default=None, help="file of extra candidate changes (e.g. LLM-proposed), one per "
                                                  "line; they always enter the search set")
    ap.add_argument("--cert-rep", type=int, default=CERT_REP)
    a = ap.parse_args()
    base = a.base or {"fresh.db": "qwen3-8b", "fresh_q25.db": "qwen2.5-7b", "ws_q3.db": "qwen3-8b",
                      "sw_q25.db": "qwen2.5-7b"}[os.path.basename(a.db)]
    out = a.out or os.path.join(os.path.dirname(a.db), f"gai_{a.method}_{os.path.basename(a.db)}.jsonl")

    L = Lattice(a.db, a.half, 3, a.metric)
    preds = L.enumerate()
    best_pred = {}
    for c, F in preds.items():
        key = F.tobytes()
        if key not in best_pred or len(c) < len(best_pred[key][0]):
            best_pred[key] = (c, F)
    arms = [Arm(0, L.predicate(c), e, F, L.N, L.Y, L.tids) for c, F in best_pred.values() for e in a.effects]
    if a.extra:
        import numpy as np
        import sqlite3 as _sq
        _con = _sq.connect(a.db)
        idx_of = {t: i for i, t in enumerate(L.tids)}
        for line in open(a.extra):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                ch = Change.parse(line)
                hit = {t for t, _k in _con.execute(ch.first_match_sql()) if t in idx_of}
            except Exception as e:
                print("extra candidate skipped:", repr(e)[:120], flush=True)
                continue
            if not hit:
                continue
            F = np.zeros(L.N, dtype=bool)
            for t in hit:
                F[idx_of[t]] = True
            arm = Arm(0, ch.pred, ch.effect + (":" + ch.arg if ch.arg else ""), F, L.N, L.Y, L.tids)
            arm.extra = True
            arms.append(arm)
    for i, arm in enumerate(arms):
        arm.idx = i
    rng = random.Random(a.seed)
    for arm in arms:
        arm.order = arm.members[:]
        rng.shuffle(arm.order)
    Y = dict(zip(L.tids, L.Y))
    N = L.N

    def full(arm):
        t = Change.parse(arm.text).text
        return (a.prefix + " ;; " + t) if a.prefix else t

    eng = Engine(a.db, base, a.workers)
    traces = eng.traces()
    evs, k0s = {}, {}

    def k0_of(arm, tid):
        if (arm.idx, tid) not in k0s:
            for t, k in eng.con.execute(Change.parse(arm.text).first_match_sql()):
                k0s[(arm.idx, t)] = k
        return k0s[(arm.idx, tid)]

    def cached(arm, tid, rep):
        col = "won" if a.metric == "won" else "COALESCE(score, won)"
        r = eng.con.execute(f"SELECT {col} FROM runs WHERE change=? AND tid=? AND rep=? AND k0>=0",
                            (full(arm), tid, rep)).fetchone()
        return None if r is None else float(r[0])

    pool = ProcessPoolExecutor(a.workers)
    stats = {"requested": 0, "executed": 0}

    def run(reqs):
        """reqs: list of (arm, tid, rep). Returns {(arm.idx, tid, rep): won}."""
        res, todo = {}, []
        for arm, tid, rep in reqs:
            w = cached(arm, tid, rep)
            if w is None:
                todo.append((arm, tid, rep))
            else:
                res[(arm.idx, tid, rep)] = w
        futs = {}
        for arm, tid, rep in todo:
            if tid not in evs:
                evs[tid] = eng.events(tid)
            futs[pool.submit(replay_one, full(arm), base, traces[tid], evs[tid],
                             k0_of(arm, tid), rep, None, True)] = (arm, tid, rep)
        for fu in as_completed(futs):
            arm, tid, rep = futs[fu]
            try:
                r = fu.result()
                eng._insert_run(r)
                res[(arm.idx, tid, rep)] = float(r["won"] if a.metric == "won" else r["score"])
            except Exception as e:
                print("replay failed:", repr(e)[:200], flush=True)
        stats["requested"] += len(reqs)
        stats["executed"] += len(todo)
        return res

    def pull(arm_list, k):
        """k more search pulls (or all remaining when k is None) for each arm."""
        reqs = []
        for arm in arm_list:
            rest = [t for t in arm.order if t not in arm.x]
            for t in (rest if k is None else rest[:k]):
                reqs.append((arm, t, a.search_rep))
        res = run(reqs)
        for arm, t, rep in reqs:
            if (arm.idx, t, rep) in res:
                arm.x[t] = res[(arm.idx, t, rep)] - Y[t]

    t0 = time.time()
    eligible = [x for x in arms if x.hi >= a.tau and x.n > 0]
    if a.method == "sh":
        # canonical eta-ary sequential halving (Theorem B): R = ceil(log_eta M) rounds, budget B/R per
        # round split evenly over the surviving arms; never exceeds the search budget
        eta = 4
        ext = [x for x in eligible if getattr(x, "extra", False)]
        S = ext + sorted((x for x in eligible if not getattr(x, "extra", False)), key=lambda x: -x.hi)[
            : max(0, a.M - len(ext))]
        R = max(1, math.ceil(math.log(len(S), eta)))
        per_round = a.search_budget / R
        for r in range(R):
            t_r = max(1, int(per_round // len(S)))
            pull(S, t_r)
            if len(S) == 1:
                break
            S = sorted(S, key=lambda x: -x.mean())[: max(1, math.ceil(len(S) / eta))]
        finalists = sorted(S, key=lambda x: -x.mean()) + sorted(
            (x for x in eligible if x.x and x not in S), key=lambda x: -x.mean())
    elif a.method == "lin":
        # preregistered Lattice-Lin (PLAN.md): shared ridge model + Thompson sampling over all arms
        import numpy as np
        from lattice_lin import LinModel
        M = LinModel(L, a.effects)
        atom_idx = {at[0]: j for j, at in enumerate(L.atoms)}
        idx = {t: i for i, t in enumerate(L.tids)}

        def combo(arm):
            return tuple(sorted(atom_idx[p.strip()] for p in arm.pred.split(" AND ")))

        feats = {}
        for arm in eligible:
            f = M.arm_features(combo(arm), arm.effect)
            if f is not None:
                feats[arm.idx] = f
        cands = [x for x in eligible if x.idx in feats]
        rs = random.Random(a.seed + 7)
        pull(rs.sample(cands, min(40, len(cands))), 4)
        while stats["requested"] < a.search_budget:
            X, yv = [], []
            for arm in cands:
                cb = combo(arm)
                for t, xv in arm.x.items():
                    f = M.feat(cb, arm.effect, idx[t])
                    if f is not None:
                        X.append(f)
                        yv.append(xv)
            M.fit(X, yv, lam=3.0)
            theta = np.random.default_rng(a.seed * 1000 + stats["requested"]).multivariate_normal(
                M.theta, M.sigma2 * M.Pinv)
            cand = [x for x in cands if len(x.x) < x.n]
            if not cand:
                break
            top = sorted(cand, key=lambda x: -float(feats[x.idx] @ theta))[:8]
            left = a.search_budget - stats["requested"]
            pull(top, max(1, min(8, left // len(top))))
        post = {x.idx: float(feats[x.idx] @ M.theta) for x in cands}
        finalists = sorted((x for x in cands if len(x.x) >= 10), key=lambda x: -post[x.idx])
    else:
        order = sorted(eligible, key=lambda x: -x.hi) if a.method == "ub-order" else \
            random.Random(a.seed + 1).sample(eligible, len(eligible))
        done = []
        i = 0
        while i < len(order) and stats["requested"] < a.search_budget:
            batch = []
            need = 0
            while i < len(order) and stats["requested"] + need < a.search_budget and need < 256:
                batch.append(order[i])
                need += order[i].n - len(order[i].x)
                i += 1
            pull(batch, None)
            done += batch
        finalists = sorted(done, key=lambda x: -x.mean())
    search_requested, search_executed = stats["requested"], stats["executed"]
    print(f"[{a.method}] search: {search_requested} pulls requested ({search_executed} new), "
          f"{sum(1 for x in arms if x.x)} arms touched, {int(time.time() - t0)}s", flush=True)
    for x in finalists[:5]:
        print(f"  search {100 * x.mean():+6.2f} pts  pulled {len(x.x)}/{x.n}  {x.text[27:]}", flush=True)

    z = NormalDist().inv_cdf(1 - a.delta / a.m)
    certified = None
    attempts = []
    for att, arm in enumerate(finalists[: a.m]):
        rep = a.cert_rep + att
        res = run([(arm, t, rep) for t in arm.members])
        D = [(res[(arm.idx, t, rep)] - Y[t]) if t in arm.members and (arm.idx, t, rep) in res else 0.0
             for t in L.tids]
        mu = sum(D) / N
        se = math.sqrt(sum((d - mu) ** 2 for d in D) / (N - 1) / N)
        lcb = mu - z * se
        attempts.append({"arm": arm.text, "full": full(arm), "rep": rep, "search": arm.mean(), "cert": mu,
                         "se": se, "lcb": lcb})
        print(f"  certify #{att + 1}: {100 * mu:+.2f} pts (SE {100 * se:.2f}), one-sided LCB {100 * lcb:+.2f} "
              f"vs tau {100 * a.tau:+.2f}  {arm.text[27:]}", flush=True)
        if lcb > a.tau and mu > a.tau:       # strict: a change with zero effect and zero variance is not certified
            certified = attempts[-1]
            break
    pool.shutdown()
    summary = {"method": a.method, "prefix": a.prefix, "db": a.db, "search_requested": search_requested, "search_executed": search_executed,
               "total_requested": stats["requested"], "total_executed": stats["executed"],
               "certified": certified, "attempts": attempts, "secs": int(time.time() - t0)}
    print(json.dumps(summary), flush=True)
    with open(out, "a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
