"""ARP: adaptive replay planning for what-if queries under a budget (DESIGN_V2.md).

Estimator (doubly robust fork-point replay). For each affected episode i with
prediction mu_i and replay probability q_i:
    D_i = (mu_i - Y_i) + (S_i / q_i) * (Vhat_i - mu_i),   Delta = mean over all N episodes
Unbiased for any predictor that does not use episode i's own replay, and any q_i > 0.

Allocation: q_i = min(1, lam * sqrt(r_i / c_i)), r_i = predicted E[(Vhat - mu)^2],
c_i = predicted replay cost; lam spends the budget (water filling).

Predictor, in two layers:
  base   h(x): P(success | prefix features), trained on the LOGGED trajectories of
         other episodes (every logged step k of episode j is a sample of the base
         policy's continuation value with label Y_j): free, no replays.
  calib  mu_i = sigmoid(a * logit h(x_i) + b * Y_i + c), fitted on the pilot replays
         of this change (cross-fitted across two pilot folds).

Two phases: a uniform pilot (fraction rho of affected episodes, replayed once),
then the water-filled second phase on the remaining affected episodes, with q
computed from the pilot only, so each phase is unbiased conditionally.

Modes
  offline  draw replays from cached runs (cheap; for tuning and budget curves)
  online   execute new replays through the engine (GPU), rep ids >= --rep-base

  python gowhy/arp.py offline --db store/fresh_q25.db --change "<UPDATE ...>" --budget 0.4 --trials 200
  python gowhy/arp.py online  --db store/fresh_q25.db --change "<UPDATE ...>" --budget 0.4 --method arp --run 0
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


# --------------------------------------------------------------------------- #
# features of a prefix (ALFWorld progress features; generic ones for WebShop)
# --------------------------------------------------------------------------- #
class Store:
    def __init__(self, db):
        import store as _store
        _store.connect(db).close()                        # adds columns missing in older stores
        self.con = sqlite3.connect(db)
        self.con.row_factory = sqlite3.Row
        self.tr = {r["tid"]: dict(r) for r in self.con.execute("SELECT * FROM traces")}
        self.ev = collections.defaultdict(list)
        for r in self.con.execute("SELECT tid, k, action, obs_after, valid FROM events ORDER BY tid, k"):
            self.ev[r["tid"]].append((r["action"], (r["obs_after"] or "").lower().replace(" ", ""), r["valid"]))
        self.types = sorted({t["task_type"] for t in self.tr.values()})
        self.webshop = self.types == ["webshop"]
        self.H = 30

    def feats(self, tid, k):
        t, pre = self.tr[tid], self.ev[tid][:k]
        base = [k, self.H - k, sum(1 - v for *_, v in pre)]
        if self.webshop:
            return base + [sum(a.startswith("search") for a, *_ in pre),
                           sum(a in ("click[next >]", "click[< prev]") for a, *_ in pre),
                           int(any(a.startswith("click[b0") for a, *_ in pre)),
                           int(bool(pre) and pre[-1][0].startswith("click[b0"))]
        obj, rec = t["obj"] or "", t["recep"] or ""
        return base + [int(any(obj in o for _, o, _ in pre)),
                       int(any(a.startswith("take") and obj in a.replace(" ", "") for a, *_ in pre)),
                       int(any((a.startswith("move") or a.startswith("put")) and obj in a.replace(" ", "")
                               for a, *_ in pre)),
                       int(any(a.startswith("go to") and rec in a.replace(" ", "") for a, *_ in pre)),
                       len({a for a, *_ in pre if a.startswith("go to")})] + \
                      [int(t["task_type"] == x) for x in self.types]

    def outcome(self, tid, metric):
        return float(self.tr[tid][metric])


def base_models(st: Store, folds: dict, metric: str):
    """h_f trained on logged steps of episodes NOT in fold f (two folds)."""
    from sklearn.ensemble import GradientBoostingRegressor
    models = {}
    for f in (0, 1):
        X, y = [], []
        for tid in st.tr:
            if folds.get(tid, 1 - f) == f:
                continue
            Y = st.outcome(tid, metric)
            for k in range(0, len(st.ev[tid]), 2):
                X.append(st.feats(tid, k))
                y.append(Y)
        models[f] = GradientBoostingRegressor(n_estimators=150, max_depth=3, subsample=0.8,
                                              random_state=0).fit(np.array(X), np.array(y))
    return models


def logit(p):
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def sigm(z):
    return 1 / (1 + math.exp(-z))


def fit_calib(rows):
    """rows: (logit h, Y, Vhat). Least squares on the probability scale, 3 params, few points."""
    if len(rows) < 6:
        return (1.0, 0.0, 0.0)
    best, bp = None, (1.0, 0.0, 0.0)
    for a in (0.5, 0.75, 1.0, 1.25):
        for b in (-1.0, 0.0, 1.0, 2.0, 3.0):
            for c in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0):
                err = sum((sigm(a * z + b * y + c) - v) ** 2 for z, y, v in rows)
                if best is None or err < best:
                    best, bp = err, (a, b, c)
    return bp


# --------------------------------------------------------------------------- #
# one ARP / uniform run
# --------------------------------------------------------------------------- #
def plan_and_estimate(st, affected, k0, replay, budget, method, rho, rng, metric, cost_per_step, models, folds,
                      qmin=0.1, oracle=None, rates=None):
    """replay(tid) -> (Vhat, cost). Returns estimate, SE, spent, n_replays.

    all      v1: every affected episode once
    uniform  same replay probability q for every affected episode (Horvitz-Thompson)
    arp      q_i = clip(lam * sqrt(m_i / c_i), qmin, 1), m_i = predicted E[(Vhat - Y_i)^2]:
             h_i if Y_i = 0 else 1 - h_i, h = log-trained base model; estimator HT with mu = Y
    arpdr    same allocation, doubly robust estimator with mu_i = h_i
    The predictor is used for allocation; with HT it cannot add variance to the estimate.
    """
    N = len(st.tr)
    Y = {t: st.outcome(t, metric) for t in st.tr}
    cst = {t: cost_per_step * max(1, st.H - k0[t]) for t in affected}
    A = list(affected)
    rng.shuffle(A)
    if method == "oracle":                                # true V from the cache: upper bound on what allocation can give
        m = {t: (oracle[t] if Y[t] < 0.5 else 1 - oracle[t]) + 0.02 for t in A}
        mu = {t: Y[t] for t in A}
    elif method in ("arp", "arpdr"):
        h = {}
        for t in A:
            hl = float(min(1.0, max(0.0, models[folds[t]].predict(np.array([st.feats(t, k0[t])]))[0])))
            if rates and t in rates:                      # shrink the task rate toward the log model
                r_, n_ = rates[t]
                h[t] = (r_ * n_ + hl * 1.0) / (n_ + 1.0)
            else:
                h[t] = hl
        m = {t: (h[t] if Y[t] < 0.5 else 1 - h[t]) + 0.02 for t in A}
        mu = {t: (h[t] if method == "arpdr" else Y[t]) for t in A}
    else:
        m = {t: 1.0 for t in A}
        mu = {t: Y[t] for t in A}
    D, spent, nrep = {}, 0.0, 0
    if method == "all":
        q = {t: 1.0 for t in A}
    else:
        def total(lam):
            return sum(min(1.0, max(qmin, lam * math.sqrt(m[t] / cst[t]))) * cst[t] for t in A)
        lo, hi = 0.0, 1e6
        for _ in range(80):                               # water filling
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if total(mid) < budget else (lo, mid)
        q = {t: min(1.0, max(qmin, lo * math.sqrt(m[t] / cst[t]))) for t in A}
    for t in A:
        if q[t] >= 1.0 or rng.random() < q[t]:
            v, c = replay(t)
            spent += c
            nrep += 1
            D[t] = (mu[t] - Y[t]) + (v - mu[t]) / q[t]
        else:
            D[t] = mu[t] - Y[t]
    vals = [D.get(t, 0.0) for t in st.tr]
    mm = sum(vals) / N
    se = math.sqrt(sum((x - mm) ** 2 for x in vals) / (N - 1) / N)
    return mm, se, spent, nrep


def task_rates(st, ch, metric, extra=None):
    """Per-task success rate from runs that are not replays of the queried change:
    from-scratch runs and fork replays of other changes, and optional extra base runs (jsonl)."""
    vals = collections.defaultdict(list)
    for t, w, sc in st.con.execute("SELECT tid, won, score FROM runs WHERE change != ?", (ch.text,)):
        vals[t].append(float(w if metric == "won" else (sc if sc is not None else w)))
    if extra:
        for line in open(extra):
            d = json.loads(line)
            vals[d["tid"]].append(float(d["won"] if metric == "won" else d.get("score", d["won"])))
    return {t: (sum(v) / len(v), len(v)) for t, v in vals.items() if v}


def setup(db, change_text, metric):
    st = Store(db)
    ch = Change.parse(change_text)
    k0 = {t: k for t, k in st.con.execute(ch.first_match_sql())}
    affected = set(k0)
    rng = random.Random(0)
    tids = sorted(st.tr)
    rng.shuffle(tids)
    folds = {t: i % 2 for i, t in enumerate(tids)}
    models = base_models(st, folds, metric)
    return st, ch, k0, affected, folds, models


def offline(a):
    st, ch, k0, affected, folds, models = setup(a.db, a.change, a.metric)
    cache = collections.defaultdict(list)
    for t, w, s, p, c in st.con.execute(
            "SELECT tid, won, score, prompt_tokens, completion_tokens FROM runs WHERE change=? AND k0>=0",
            (ch.text,)):
        cache[t].append((float(w if a.metric == "won" else (s if s is not None else w)), p + c))
    affected = {t for t in affected if cache[t]}
    truth = sum(sum(v for v, _ in cache[t]) / len(cache[t]) - st.outcome(t, a.metric) for t in affected) / len(st.tr)
    full_cost = sum(sum(c for _, c in cache[t]) / len(cache[t]) for t in affected)
    Vbar = {t: sum(v for v, _ in cache[t]) / len(cache[t]) for t in affected}
    rates = task_rates(st, ch, a.metric, a.extra_runs)
    steps = sum(st.H - k0[t] for t in affected)
    cps = full_cost / steps
    print(f"{len(affected)} affected of {len(st.tr)}; truth (all cached reps) {truth:+.4f}; "
          f"v1 cost {full_cost / 1e6:.1f}M tokens", flush=True)
    for budget in a.budgets:
        for method in (["all"] if budget >= 1 else []) + ["uniform", "arp", "arpdr", "oracle"]:
            errs, ses, costs = [], [], []
            for trial in range(a.trials):
                rng = random.Random(1000 + trial)

                def replay(t):
                    return rng.choice(cache[t])

                m, se, spent, _ = plan_and_estimate(st, affected, k0, replay, budget * full_cost, method,
                                                    a.rho, rng, a.metric, cps, models, folds, oracle=Vbar,
                                                    rates=rates)
                errs.append(m - truth)
                ses.append(se)
                costs.append(spent / full_cost)
            rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
            bias = sum(errs) / len(errs)
            print(f"  budget {budget:4.2f} {method:8s} spent {sum(costs) / len(costs):.2f}  "
                  f"bias {bias:+.4f}  RMSE {rmse:.4f}  mean SE {sum(ses) / len(ses):.4f}", flush=True)


def online(a):
    """New replays on the GPU (rep id = rep_base + run). Pilot first, then the planned
    second phase, both executed in parallel by the engine; decisions use the same RNG
    stream as the offline simulation, so the plan is exactly the simulated one."""
    from engine import Engine
    st, ch, k0, affected, folds, models = setup(a.db, a.change, a.metric)
    eng = Engine(a.db, a.base, a.workers)
    rep = a.rep_base + a.run
    cached = [p + c for p, c in st.con.execute(
        "SELECT prompt_tokens, completion_tokens FROM runs WHERE change=? AND k0>=0 AND rep<100", (ch.text,))]
    steps = sum(st.H - k0[t] for t in affected)
    cps = (sum(cached) / len(cached)) / (steps / len(affected)) if cached else 2000.0
    full_cost = cps * steps

    def have(t):
        rows = [r for r in eng.cached(ch, t, k0[t]) if r["rep"] == rep]
        if not rows:
            return None
        r = rows[0]
        v = float(r["won"] if a.metric == "won" else (r["score"] if r["score"] is not None else r["won"]))
        return v, (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0)

    def run_tids(tids):
        tids = {t for t in tids if have(t) is None}
        if tids:
            eng.run(ch, 1, reps_offset=rep, tids=tids, verbose=True)

    A = list(affected)
    seed = 5000 + a.run
    if a.method == "all":
        run_tids(A)
    need = []

    def plan_replay(t):
        h = have(t)
        if h is not None:
            return h
        need.append(t)
        return 0.0, cps * max(1, st.H - k0[t])

    rates = task_rates(st, ch, a.metric, a.extra_runs)
    plan_and_estimate(st, affected, k0, plan_replay, a.budget * full_cost, a.method, a.rho,
                      random.Random(seed), a.metric, cps, models, folds, rates=rates)
    run_tids(need)
    missing = []

    def final_replay(t):
        h = have(t)
        if h is None:
            missing.append(t)
            return 0.0, 0.0
        return h

    m, se, spent, nrep = plan_and_estimate(st, affected, k0, final_replay, a.budget * full_cost, a.method,
                                           a.rho, random.Random(seed), a.metric, cps, models, folds, rates=rates)
    out = {"change": ch.text, "method": a.method, "budget": a.budget, "run": a.run, "rep": rep,
           "estimate": m, "se": se, "spent_tokens": spent, "v1_cost_est": full_cost, "replays": nrep,
           "missing": len(missing)}
    print(json.dumps(out), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["offline", "online"])
    ap.add_argument("--db", required=True)
    ap.add_argument("--change", required=True)
    ap.add_argument("--metric", choices=["won", "score"], default="won")
    ap.add_argument("--rho", type=float, default=0.25)
    ap.add_argument("--budgets", type=float, nargs="*", default=[1.0, 0.6, 0.4, 0.25])
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--budget", type=float, default=0.4)
    ap.add_argument("--method", choices=["arp", "arpdr", "uniform", "all"], default="arp")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--rep-base", type=int, default=200)
    ap.add_argument("--base", default=None)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", default="store/arp_online.jsonl")
    ap.add_argument("--extra-runs", default=None, help="jsonl of extra base-policy runs (tid, won)")
    a = ap.parse_args()
    if a.change.startswith("@"):
        a.change = open(a.change[1:]).read().strip()
    if a.base is None:
        a.base = {"fresh.db": "qwen3-8b", "fresh_q25.db": "qwen2.5-7b", "ws_q3.db": "qwen3-8b",
                  "fresh_llama.db": "llama-3.1-8b"}[os.path.basename(a.db)]
    offline(a) if a.mode == "offline" else online(a)


if __name__ == "__main__":
    main()
