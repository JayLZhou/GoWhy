"""Offline benchmark for how-to search, built from cached real replays (no GPU).

Universe: the candidates (predicate, effect) of the lattice whose every affected
episode already has a cached replay on the search seed (rep 600). A pull returns the
cached outcome, so a search algorithm can be run for hundreds of seeds at no cost,
and each candidate's true effect on this seed is known exactly:
    Delta(C) = (1/N) sum_{i in F(C)} (Y'_i - Y_i).
Randomness across runs comes from the algorithms (orderings, Thompson draws), not
from replays, so this benchmark measures search quality, not certification.

Metrics per run: true effect of the top-1 and best of the top-3 candidates the
algorithm would send to certification, and simple regret against the best candidate.

  python gowhy/offline_bench.py --db store/fresh_q25.db --half B --budget 300 600 --runs 200
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import sqlite3
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from changes import Change  # noqa: E402
from lattice import Lattice  # noqa: E402
from lattice_lin import LinModel  # noqa: E402

REP = 600


class Cand:
    def __init__(self, idx, combo, effect, members, x, N):
        self.idx, self.combo, self.effect = idx, combo, effect
        self.members = members            # episode indices
        self.n = len(members)
        self.w = self.n / N
        self.xall = x                     # episode index -> cached gain Y' - Y
        self.truth = sum(x.values()) / N


def load(db, half, effects, metric, universe=None, exclude=None):
    L = Lattice(db, half, 3, metric)
    best = {}
    for cb, F in L.enumerate().items():
        k = F.tobytes()
        if k not in best or len(cb) < len(best[k][0]):
            best[k] = (cb, F)
    con = sqlite3.connect(db)
    col = "won" if metric == "won" else "COALESCE(score, won)"
    cache = collections.defaultdict(dict)
    for ch, tid, v in con.execute(f"SELECT change, tid, {col} FROM runs WHERE rep=? AND k0>=0", (REP,)):
        cache[ch][tid] = float(v)
    keep = None
    if universe:
        with open(universe) as f:
            keep = {(tuple(x["combo"]), x["effect"]) for x in json.load(f)["arms"]}
    drop = set()
    if exclude:
        with open(exclude) as f:
            drop = {(tuple(x["combo"]), x["effect"]) for x in json.load(f)["arms"]}
    cands = []
    for cb, F in best.values():
        mem = [i for i in range(L.N) if F[i]]
        for e in effects:
            if keep is not None and (tuple(cb), e) not in keep:
                continue
            if (tuple(cb), e) in drop:
                continue
            txt = Change.parse(f"UPDATE events SET action = {e} WHERE {L.predicate(cb)} SCOPE ALL").text
            c = cache.get(txt, {})
            if all(L.tids[i] in c for i in mem):
                x = {i: c[L.tids[i]] - L.Y[i] for i in mem}
                cands.append(Cand(len(cands), cb, e, mem, x, L.N))
    return L, cands


class Run:
    """One search run: pulls without replacement within a candidate, counted against a budget."""

    def __init__(self, cands, budget, rng):
        self.cands, self.budget, self.used = cands, budget, 0
        self.order = {c.idx: rng.sample(c.members, c.n) for c in cands}
        self.x = {c.idx: {} for c in cands}

    def pull(self, c, k=None):
        rest = [i for i in self.order[c.idx] if i not in self.x[c.idx]]
        take = rest if k is None else rest[:k]
        take = take[: max(0, self.budget - self.used)]
        for i in take:
            self.x[c.idx][i] = c.xall[i]
        self.used += len(take)
        return len(take)

    def emp(self, c):
        xs = self.x[c.idx]
        return c.w * sum(xs.values()) / len(xs) if xs else None

    def left(self):
        return self.budget - self.used


def m_random(R, cands, rng, ub=False):
    order = sorted(cands, key=lambda c: -c.ub) if ub else rng.sample(cands, len(cands))
    done = []
    for c in order:
        if R.left() <= 0:
            break
        R.pull(c)
        if len(R.x[c.idx]) == c.n:
            done.append(c)
    return sorted(done, key=lambda c: -R.emp(c))


def m_sh(R, cands, rng, M=96, eta=4):
    S = sorted(cands, key=lambda c: (-c.ub, rng.random()))[:M]
    rounds = max(1, math.ceil(math.log(len(S), eta)))
    per = R.budget / rounds
    for _ in range(rounds):
        t = max(1, int(per // len(S)))
        for c in S:
            R.pull(c, t)
        if len(S) == 1:
            break
        S = sorted(S, key=lambda c: -(R.emp(c) or -9))[: max(1, math.ceil(len(S) / eta))]
    rest = sorted((c for c in cands if R.x[c.idx] and c not in S), key=lambda c: -R.emp(c))
    return sorted(S, key=lambda c: -(R.emp(c) or -9)) + rest


def m_sh_lin(R, cands, rng, M, kappa, M0=96, eta=4, init="ub"):
    """Sequential halving whose eliminations rank survivors by the empirical-Bayes blend of their
    own pulls and a shared model refit on every pull so far (all candidates, not only survivors).
    init='ub': the M0 candidates with the largest log upper bound; init='model': spend a warm-up
    (40 random candidates x 4 pulls), fit the model, start from the M0 best predicted candidates."""
    if init == "model":
        for c in rng.sample(cands, min(40, len(cands))):
            R.pull(c, 4)
        lin_fit(R, cands, M)
        S = sorted(cands, key=lambda c: -float(c.f_arm @ M.theta))[:M0]
        R0 = R.left()
    else:
        S = sorted(cands, key=lambda c: (-c.ub, rng.random()))[:M0]
        R0 = R.budget
    rounds = max(1, math.ceil(math.log(len(S), eta)))
    per = R0 / rounds
    for r in range(rounds):
        t = max(1, int(per // len(S)))
        for c in S:
            R.pull(c, t)
        lin_fit(R, cands, M)
        score = {c.idx: eb(R, c, M, M.theta, kappa) for c in S}
        if len(S) == 1:
            break
        S = sorted(S, key=lambda c: -score[c.idx])[: max(1, math.ceil(len(S) / eta))]
    lin_fit(R, cands, M)
    pulled = [c for c in cands if R.x[c.idx]]
    score = {c.idx: eb(R, c, M, M.theta, kappa) for c in pulled}
    return sorted(S, key=lambda c: -score[c.idx]) + sorted((c for c in pulled if c not in S),
                                                           key=lambda c: -score[c.idx])


def lin_fit(R, cands, M, lam=3.0):
    X, y, arm = [], [], []
    for c in cands:
        for i, v in R.x[c.idx].items():
            X.append(c.f_ep[i])
            y.append(v)
            arm.append(c.idx)
    M.fit(X, y, lam=lam)
    # empirical-Bayes shrinkage strength kappa = within-candidate noise / between-candidate residual
    # variance (method of moments over candidates with >= 2 pulls)
    res = np.asarray(y) - np.asarray(X) @ M.theta
    by = collections.defaultdict(list)
    for r_, k in zip(res, arm):
        by[k].append(r_)
    g = [v for v in by.values() if len(v) >= 2]
    if g:
        within = float(np.mean([np.var(v, ddof=1) for v in g]))
        means = np.array([np.mean(v) for v in g])
        tbar = float(np.mean([len(v) for v in g]))
        between = max(float(np.var(means, ddof=1)) - within / tbar, 1e-4) if len(g) > 1 else 1e-4
        M.kappa_hat = within / between
    else:
        M.kappa_hat = 4.0


def m_lin(R, cands, rng, M, final="post", min_pulls=10, kappa=None, seed=0):
    """Preregistered Lattice-Lin (final='post'); final='eb' ranks by an empirical-Bayes blend of
    each candidate's own mean and the model's prediction, with min_pulls capped at |F|."""
    nprng = np.random.default_rng(seed)
    for c in rng.sample(cands, min(40, len(cands))):
        R.pull(c, 4)
    while R.left() > 0:
        lin_fit(R, cands, M)
        th = nprng.multivariate_normal(M.theta, M.sigma2 * M.Pinv)
        open_ = [c for c in cands if len(R.x[c.idx]) < c.n]
        if not open_:
            break
        if final == "eb":
            score = {c.idx: eb(R, c, M, th, kappa) for c in open_}
            top = sorted(open_, key=lambda c: -score[c.idx])[:8]
        else:
            top = sorted(open_, key=lambda c: -float(c.f_arm @ th))[:8]
        k = max(1, min(8, R.left() // len(top)))
        if sum(R.pull(c, k) for c in top) == 0:
            break
    lin_fit(R, cands, M)
    if final == "eb":
        pool = [c for c in cands if len(R.x[c.idx]) >= min(min_pulls, c.n)]
        return sorted(pool, key=lambda c: -eb(R, c, M, M.theta, kappa))
    pool = [c for c in cands if len(R.x[c.idx]) >= min_pulls]
    return sorted(pool, key=lambda c: -float(c.f_arm @ M.theta))


def eb(R, c, M, th, kappa):
    """Shrink the candidate's own estimate toward the model: (t * emp + kappa * pred) / (t + kappa),
    on the per-episode scale; t = pulls so far. With no pulls, the prediction alone."""
    pred = float(c.f_arm @ th)
    xs = R.x[c.idx]
    t = len(xs)
    if t == 0:
        return pred
    # residual of own episodes against the model, scaled to the candidate's effect
    resid = sum(v - float(c.f_ep[i] @ th) for i, v in xs.items()) / t
    fpc = 1.0 - (t - 1) / c.n if c.n > 1 else 0.0   # finite population: exhausted -> trust data fully
    k = (M.kappa_hat if kappa is None else kappa) * fpc
    return pred + c.w * resid * t / (t + k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--half", default="B")
    ap.add_argument("--effects", nargs="*", default=["hint:2", "replan:2", "reject:3"])
    ap.add_argument("--metric", default="won")
    ap.add_argument("--budget", type=int, nargs="*", default=[300, 600])
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--methods", nargs="*", default=["random", "ub-order", "sh", "lin", "lin-eb", "sh-lin"])
    ap.add_argument("--kappa", type=float, default=4.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--universe", default=None, help="JSON from fill_universe.py: benchmark on these candidates")
    ap.add_argument("--exclude", default=None, help="JSON from fill_universe.py: leave these candidates out (dev set)")
    a = ap.parse_args()

    L, cands = load(a.db, a.half, a.effects, a.metric, a.universe, a.exclude)
    M = LinModel(L, a.effects)
    for c in cands:
        c.ub = float(sum(1.0 - L.Y[i] for i in c.members)) / L.N
        c.f_ep = {i: M.feat(c.combo, c.effect, i) for i in c.members}
        c.f_arm = sum(c.f_ep.values()) / L.N
    truths = sorted((c.truth for c in cands), reverse=True)
    total = sum(c.n for c in cands)
    print(f"{a.db} half {a.half}: {len(cands)} fully cached candidates, {total} pulls to exhaust all; "
          f"true effect best {100 * truths[0]:+.2f}, 10th {100 * truths[min(9, len(truths) - 1)]:+.2f}, median "
          f"{100 * truths[len(truths) // 2]:+.2f} pts; candidates >= +4 pts: {sum(t >= 0.04 for t in truths)}",
          flush=True)
    res = []
    for B in a.budget:
        for meth in a.methods:
            top1, top3 = [], []
            for s in range(a.runs):
                rng = random.Random(1000 * s + 17)
                R = Run(cands, B, rng)
                if meth in ("random", "ub-order"):
                    fin = m_random(R, cands, rng, ub=meth == "ub-order")
                elif meth == "sh":
                    fin = m_sh(R, cands, rng)
                elif meth == "sh-lin":
                    fin = m_sh_lin(R, cands, rng, M, a.kappa)
                elif meth == "sh-lin-m":
                    fin = m_sh_lin(R, cands, rng, M, a.kappa, init="model")
                elif meth == "sh-lin-k":
                    fin = m_sh_lin(R, cands, rng, M, None)
                elif meth == "sh-lin-mk":
                    fin = m_sh_lin(R, cands, rng, M, None, init="model")
                else:
                    fin = m_lin(R, cands, rng, M, final="eb" if meth == "lin-eb" else "post",
                                kappa=a.kappa, seed=s)
                t1 = fin[0].truth if fin else 0.0
                t3 = max((c.truth for c in fin[:3]), default=0.0)
                top1.append(t1)
                top3.append(t3)
            r = {"budget": B, "method": meth, "runs": a.runs,
                 "top1": float(np.mean(top1)), "top1_se": float(np.std(top1) / math.sqrt(a.runs)),
                 "top3": float(np.mean(top3)), "regret1": float(truths[0] - np.mean(top1)),
                 "p_top1_ge4": float(np.mean([t >= 0.04 for t in top1])),
                 "p_top3_ge4": float(np.mean([t >= 0.04 for t in top3]))}
            res.append(r)
            print(f"  B={B:5d} {meth:9s} top1 {100 * r['top1']:+.2f} (SE {100 * r['top1_se']:.2f})  "
                  f"best-of-3 {100 * r['top3']:+.2f}  P(top1>=4) {r['p_top1_ge4']:.2f}  "
                  f"P(top3>=4) {r['p_top3_ge4']:.2f}", flush=True)
    if a.out:
        with open(a.out, "a") as f:
            for r in res:
                f.write(json.dumps({"db": a.db, **r}) + "\n")


if __name__ == "__main__":
    main()
