"""WHATIF / HOWTO over a trajectory store.

  python gowhy/query.py "WHATIF UPDATE events SET action = reject:3 WHERE valid = 0 SCOPE ALL GROUP BY task_type WITH n = 8"
  python gowhy/query.py --db store/fresh_q25.db "WHATIF ... WITH n = 8 DEPLOY m = 2"
  python gowhy/query.py "HOWTO candidates.txt WITH n = 4 CERTIFY n = 8"

Estimator (deployment basis, README section 6 / QGATE C1). Over the N stored
games, D_i = mean_r(won of replay r) - Y_i for affected games and D_i = 0 for
unaffected ones. Delta = mean(D), SE = sd(D) / sqrt(N). Do not use the
spread of the replays alone: it ignores the unaffected games and is too narrow.

DEPLOY m: also run the change from scratch m times on all N games and report
the deployed effect with its paired SE and z = (whatif - deploy) / sqrt(SE1^2 + SE2^2).
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from changes import Change  # noqa: E402
from engine import Engine  # noqa: E402

ROOT = os.environ.get("GOWHY_ROOT", "/data2/yujia/Yingli/gowhy")
BASE_OF = {"fresh.db": "qwen3-8b", "fresh_q25.db": "qwen2.5-7b", "ws_q3.db": "qwen3-8b",
           "fresh_llama.db": "llama-3.1-8b"}

_Q = re.compile(r"^\s*(?P<kind>WHATIF|HOWTO)\s+(?P<body>.+?)(?:\s+GROUP\s+BY\s+(?P<group>\w+))?"
                r"\s+WITH\s+n\s*=\s*(?P<n>\d+)(?:\s+DEPLOY\s+m\s*=\s*(?P<m>\d+))?"
                r"(?:\s+CERTIFY\s+n\s*=\s*(?P<c>\d+))?\s*$", re.I | re.S)


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def se(x):
    if len(x) < 2:
        return float("nan")
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1) / len(x))


def estimate(traces: dict, rows: list, affected: set, group: str | None = None) -> dict:
    """Deployment-basis estimate from what-if replay rows.
    `affected` comes from the fork points, not from the rows: an affected game
    whose replays are missing must not be counted as unaffected (D = 0)."""
    by = defaultdict(list)
    for r in rows:
        by[r["tid"]].append(r[METRIC])
    missing = sorted(t for t in affected if t in traces and not by[t])
    if missing:
        raise RuntimeError(f"{len(missing)} affected games have no replays (e.g. {missing[:3]}); rerun to fill them")
    groups = defaultdict(list)
    for tid, t in traces.items():
        d = mean(by[tid]) - t[METRIC] if tid in affected else 0.0
        for g in (["all", t[group]] if group else ["all"]):
            groups[g].append((d, tid in affected))
    out = {}
    for g, ds in sorted(groups.items()):
        D = [d for d, _ in ds]
        out[g] = {"N": len(D), "affected": sum(a for _, a in ds), "delta": mean(D), "se": se(D)}
    return out


METRIC = "won"  # or "score" (WebShop's 0-1 reward); set by --metric
REP0 = 0  # first rep id; 100 with --crn (seeded replays, see engine.SEEDED_REP0)
BASE_EXTRA: dict = {}  # tid -> [won, ...] from independent base-policy runs (--base-extra)


def deploy_estimate(traces: dict, rows: list, group: str | None = None) -> dict:
    """Deployed effect: changed policy from scratch minus the base policy's success.
    The base is the logged Y, averaged with any extra independent base runs, which
    halves the noise a single logged outcome puts into the comparison."""
    by = defaultdict(list)
    for r in rows:
        by[r["tid"]].append(r[METRIC])
    groups = defaultdict(list)
    for tid, t in traces.items():
        if tid in by:
            base = mean([t[METRIC]] + (BASE_EXTRA.get(tid, []) if METRIC == "won" else []))
            for g in (["all", t[group]] if group else ["all"]):
                groups[g].append(mean(by[tid]) - base)
    return {g: {"N": len(D), "delta": mean(D), "se": se(D)} for g, D in sorted(groups.items())}


def cost(rows: list) -> dict:
    return {k: sum(r[k] or 0 for r in rows) for k in ["calls", "prompt_tokens", "completion_tokens"]}


def scratch_cost(traces: dict, n: int) -> dict:
    """Cost of rerunning every stored game n times, from the logged per-game usage."""
    return {k: n * sum(t[k] or 0 for t in traces.values())
            for k in ["calls", "prompt_tokens", "completion_tokens"]}


def fmt(e):
    lo, hi = e["delta"] - 1.96 * e["se"], e["delta"] + 1.96 * e["se"]
    return f"{100 * e['delta']:+.1f} pts (SE {100 * e['se']:.1f}, 95% CI {100 * lo:+.1f}..{100 * hi:+.1f})"


def whatif(eng: Engine, ch: Change, n: int, group=None, m=0, quiet=False) -> dict:
    traces = eng.traces()
    rows = eng.run(ch, n, reps_offset=REP0, verbose=not quiet)
    affected = set(eng.fork_points(ch)) & set(traces)
    est = estimate(traces, rows, affected, group)
    c, s = cost(rows), scratch_cost(traces, n)
    if not quiet:
        base_sr = mean([t[METRIC] for t in traces.values()])
        print(f"\nWHATIF {ch.text}\n  base {'success' if METRIC == 'won' else 'mean score'} "
              f"{100 * base_sr:.1f}% over N = {len(traces)} games")
        for g, e in est.items():
            print(f"  [{g}] affected {e['affected']}/{e['N']} ({100 * e['affected'] / e['N']:.0f}%): {fmt(e)}")
        tok = c["prompt_tokens"] + c["completion_tokens"]
        stok = s["prompt_tokens"] + s["completion_tokens"]
        print(f"  cost: {c['calls']} calls, {tok:,} tokens; from-scratch with the same n: "
              f"{s['calls']} calls, {stok:,} tokens; ratio {tok / max(stok, 1):.2f}")
    res = {"est": est, "cost": c, "scratch": s}
    if m:
        drows = eng.run(ch, m, deploy=True, reps_offset=REP0, verbose=not quiet)
        dep = deploy_estimate(traces, drows, group)
        res["deploy"] = dep
        print(f"\nDEPLOY (from scratch, m = {m})")
        for g, d in dep.items():
            w = est[g]
            z = (w["delta"] - d["delta"]) / math.sqrt(w["se"] ** 2 + d["se"] ** 2)
            print(f"  [{g}] deployed {fmt(d)}; what-if {100 * w['delta']:+.1f}; z = {z:.2f}")
        dc = cost(drows)
        print(f"  deploy cost: {dc['calls']} calls, {dc['prompt_tokens'] + dc['completion_tokens']:,} tokens")
    return res


def howto(eng: Engine, cands: list, n: int, n_cert: int):
    results = []
    for text in cands:
        ch = Change.parse(text)
        r = whatif(eng, ch, n, quiet=True)
        e = r["est"]["all"]
        results.append((e["delta"], e["se"], e["affected"], ch))
        print(f"  {100 * e['delta']:+6.1f} (SE {100 * e['se']:.1f}, affected {e['affected']})  {ch.text}", flush=True)
    results.sort(key=lambda x: -x[0])
    best = results[0][3]
    print(f"\nbest of {len(results)} on the selection replays: {best.text}  {100 * results[0][0]:+.1f}")
    if n_cert:
        traces = eng.traces()
        rows = eng.run(best, n_cert, reps_offset=REP0 + n, verbose=False)
        e = estimate(traces, rows, set(eng.fork_points(best)) & set(traces))["all"]
        print(f"certified on {n_cert} fresh replays per game (independent of selection): {fmt(e)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--db", default=os.path.join(ROOT, "store", "fresh.db"))
    ap.add_argument("--base", default=None, help="served name of the logged policy")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="first N stored games only (smoke tests)")
    ap.add_argument("--half", choices=["A", "B"], default="", help="only this half of the games (see engine.half_of)")
    ap.add_argument("--metric", choices=["won", "score"], default="won")
    ap.add_argument("--crn", action="store_true", help="seeded replays: common random numbers across changes")
    ap.add_argument("--base-extra", default="", help="jsonl of extra independent base-policy runs (tid, won) for DEPLOY")
    a = ap.parse_args()
    global REP0, METRIC
    METRIC = a.metric
    if a.crn:
        REP0 = 100
    if a.base_extra:
        import json
        for line in open(a.base_extra):
            d = json.loads(line)
            BASE_EXTRA.setdefault(d["tid"], []).append(d["won"])
    if a.query.startswith("@"):  # read the query from a file (avoids shell quoting)
        a.query = open(a.query[1:]).read().strip()
    m = _Q.match(a.query)
    if not m:
        sys.exit(f"cannot parse query: {a.query!r}")
    base = a.base or BASE_OF.get(os.path.basename(a.db))
    if base is None:
        sys.exit("pass --base <served model name>")
    eng = Engine(a.db, base, a.workers, a.limit, a.half)
    n = int(m.group("n"))
    if m.group("kind").upper() == "WHATIF":
        whatif(eng, Change.parse(m.group("body")), n, m.group("group"), int(m.group("m") or 0))
    else:
        path = m.group("body").strip()
        cands = [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
        howto(eng, cands, n, int(m.group("c") or 0))


if __name__ == "__main__":
    main()
