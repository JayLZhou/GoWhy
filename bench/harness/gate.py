"""Decisive-step gate on failed trajectories (README section 2, "failure structure").

V_k = success rate when the logged prefix of length k is kept and the policy
plays on from step k. V_n = Y = 0 for a failed trace of n steps. The drop
V_k - V_{k+1} is how much the logged action at step k hurt.

Phases (all replays cached in --out, keyed by (tid, k, rep)):
  1. capability: V_0 with --reps samples. V_0 = 0 means a rerun from scratch
     also fails: a capability failure, no single step to blame.
  2. grid:      V_k at --grid points for the other failures.
  3. bisect:    inside the steepest grid interval, halve until one step (a, a+1).
     This picks the largest of many noisy drops, so it is biased upward.
  4. certify:   re-measure V_a and V_{a+1} with --cert fresh replays (reps
     disjoint from selection). Decisive if the certified drop >= --drop and the
     one-sided two-proportion z >= 1.645.

The thresholds are assumptions of this rewrite; the original GATE.md is lost.
README result to compare with: 58% capability failures; after certification
2/43 (5%) single-step decisive events; with 8 replays per cell 12 of 14
"decisive" steps were false positives.

  python harness/gate.py --source data/fresh/Qwen3-8B-fresh.jsonl --served qwen3-8b
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aeb  # noqa: E402
from replay import continue_task  # noqa: E402

CERT_OFFSET = 1000  # rep ids for certification, disjoint from selection reps


class Cells:
    def __init__(self, out, traces, served, workers):
        self.out, self.served, self.workers = out, served, workers
        self.tr = {t.tid: t for t in traces}
        self.won = defaultdict(dict)  # (tid, k) -> {rep: won}
        if os.path.exists(out):
            for l in open(out):
                d = json.loads(l)
                self.won[(d["tid"], d["k"])][d["rep"]] = d["won"]

    def need(self, cells):
        """cells: [(tid, k, reps)] -> run whatever is missing."""
        jobs = []
        for tid, k, reps in cells:
            if k >= len(self.tr[tid].raws):
                continue  # V_n = Y, no replay needed
            have = self.won[(tid, k)]
            jobs += [(tid, k, r) for r in reps if r not in have]
        if not jobs:
            return
        print(f"  running {len(jobs)} replays", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(self.workers) as ex, open(self.out, "a") as fo:
            futs = {ex.submit(continue_task, self.tr[t], k, r, self.served): (t, k, r) for t, k, r in jobs}
            for i, fu in enumerate(as_completed(futs), 1):
                t, k, r = futs[fu]
                try:
                    d = fu.result()
                except Exception as e:
                    print("  replay failed:", repr(e)[:200], flush=True)
                    continue
                row = {"tid": t, "k": k, "rep": r, "won": d["won"], "n_steps": d["n_steps"],
                       "calls": d["calls"], "prompt_tokens": d["prompt_tokens"],
                       "completion_tokens": d["completion_tokens"]}
                fo.write(json.dumps(row) + "\n")
                fo.flush()
                self.won[(t, k)][r] = d["won"]
                if i % 200 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)

    def V(self, tid, k, reps):
        if k >= len(self.tr[tid].raws):
            return float(self.tr[tid].won), len(reps)
        w = [self.won[(tid, k)][r] for r in reps if r in self.won[(tid, k)]]
        return (sum(w) / len(w) if w else float("nan")), len(w)


def z_drop(v1, n1, v2, n2):
    p = (v1 * n1 + v2 * n2) / (n1 + n2)
    s = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else float("inf")
    return (v1 - v2) / s if s > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--served", required=True)
    ap.add_argument("--grid", default="0,5,10,15,20,25")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--cert", type=int, default=24)
    ap.add_argument("--drop", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    traces = [t for t in aeb.load_jsonl(os.path.join(aeb.ROOT, a.source)) if not t.won]
    out = a.out or os.path.join(aeb.ROOT, "store", f"gate_{a.served}.jsonl")
    C = Cells(out, traces, a.served, a.workers)
    sel = list(range(a.reps))
    grid = [int(x) for x in a.grid.split(",")]
    print(f"{len(traces)} failed traces -> {out}", flush=True)

    print("phase 1: V_0", flush=True)
    C.need([(t.tid, 0, sel) for t in traces])
    cap = [t for t in traces if C.V(t.tid, 0, sel)[0] == 0]
    rest = [t for t in traces if t not in cap]
    print(f"  capability failures (V_0 = 0/{a.reps}): {len(cap)}/{len(traces)} = {len(cap) / len(traces):.0%}")

    print("phase 2: grid", flush=True)
    C.need([(t.tid, k, sel) for t in rest for k in grid])

    print("phase 3: bisect", flush=True)
    iv = {}
    for t in rest:
        n = len(t.raws)
        pts = sorted({k for k in grid if k < n} | {n})
        best = max(zip(pts, pts[1:]), key=lambda ab: C.V(t.tid, ab[0], sel)[0] - C.V(t.tid, ab[1], sel)[0])
        iv[t.tid] = list(best)
    while any(b - a_ > 1 for a_, b in iv.values()):
        todo = {tid: (a_ + b) // 2 for tid, (a_, b) in iv.items() if b - a_ > 1}
        C.need([(tid, m, sel) for tid, m in todo.items()])
        for tid, m in todo.items():
            a_, b = iv[tid]
            va, vm, vb = (C.V(tid, x, sel)[0] for x in (a_, m, b))
            iv[tid] = [a_, m] if va - vm >= vm - vb else [m, b]

    sel_drops = {tid: C.V(tid, a_, sel)[0] - C.V(tid, a_ + 1, sel)[0] for tid, (a_, _) in iv.items()}
    flagged = [tid for tid, d in sel_drops.items() if d >= a.drop]
    print(f"  selection-stage single-step drops >= {a.drop}: {len(flagged)}/{len(rest)}")

    print("phase 4: certify", flush=True)
    cert = list(range(CERT_OFFSET, CERT_OFFSET + a.cert))
    C.need([(tid, k, cert) for tid, (a_, _) in iv.items() for k in (a_, a_ + 1)])
    decisive = []
    rows = []
    for tid, (a_, _) in sorted(iv.items()):
        v1, n1 = C.V(tid, a_, cert)
        v2, n2 = C.V(tid, a_ + 1, cert)
        z = z_drop(v1, n1, v2, n2) if a_ + 1 < len(C.tr[tid].raws) else z_drop(v1, n1, v2, max(n1, 1))
        ok = (v1 - v2) >= a.drop and z >= 1.645
        decisive += [tid] if ok else []
        rows.append((tid, a_, sel_drops[tid], v1 - v2, z, ok, C.tr[tid].raws and aeb.project(C.tr[tid].raws[a_])[0]))
    false_pos = [tid for tid in flagged if tid not in decisive]

    print("\nRESULT")
    print(f"  failed traces: {len(traces)}")
    print(f"  capability (V_0 = 0): {len(cap)} ({len(cap) / len(traces):.0%})")
    print(f"  non-capability: {len(rest)}")
    print(f"  flagged decisive with {a.reps} replays/cell: {len(flagged)}; "
          f"of these not certified (false positives): {len(false_pos)}")
    print(f"  certified decisive with {a.cert} fresh replays: {len(decisive)}/{len(rest)} "
          f"({len(decisive) / max(len(rest), 1):.0%} of non-capability, "
          f"{len(decisive) / len(traces):.0%} of all failures)")
    print("\n  tid  step  sel_drop  cert_drop  z  decisive  logged_action")
    for r in rows:
        if r[2] >= a.drop or r[5]:
            print(f"  {r[0]}  {r[1]}  {r[2]:+.2f}  {r[3]:+.2f}  {r[4]:.2f}  {r[5]}  {r[6]!r}")


if __name__ == "__main__":
    main()
