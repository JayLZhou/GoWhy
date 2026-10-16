"""Coupling experiment: disagreement mass of nearby changes under three replay schemes.

For pairs (C, C') with the same effect that share many cells (same task, same fork step), on
every task of F_C u F_C' we obtain one draw (Z_C, Z_C') under
  coupled      C replayed with a step log (seed family --rep), then C' replayed by walking C's
               replay (engine.replay_one donor=...): identical until their decisions differ;
  seeded       both replayed independently with the same seed family (rep 600, cached): common
               random numbers, but vLLM is not bit-reproducible, so the coupling is loose;
  independent  C from --rep and C' from rep 600: different seeds, independent draws.
On tasks outside the shared cells the pair is replayed independently in every scheme.

Disagreement mass d = (1/N) sum_i 1{Z_C != Z_C'} (an unbiased one-draw estimate), union coverage
w = |F_C u F_C'| / N, and the theory's comparison costs w d / Gamma^2 (coupled) vs w^2 / Gamma^2.
The model calls of C' under coupling are compared with its standalone replays.

  python gowhy/pair_eval.py --db store/fresh_q25.db --half B --universe store/universe500.json \
      --pairs 40 --rep 950 --out store/pair_eval.jsonl
"""
from __future__ import annotations

import argparse
import collections
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
from share_eval import run_cell  # noqa: E402

INDEP_REP = 600


def log(*a):
    """Print that survives a full disk (the shared /data2 fills up from time to time)."""
    try:
        print(*a, flush=True)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--half", default="B")
    ap.add_argument("--base", default="qwen2.5-7b")
    ap.add_argument("--universe", required=True)
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--min-shared", type=float, default=0.5, help="min fraction of F_u in shared cells")
    ap.add_argument("--rep", type=int, default=2000, help="pair p uses seed family rep + p")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    L = Lattice(a.db, a.half, 3, "won")
    N = L.N
    Y = dict(zip(L.tids, L.Y))
    half = set(L.tids)
    eng = Engine(a.db, a.base, a.workers)
    traces = eng.traces()
    arms = json.load(open(a.universe))["arms"]
    texts = [Change.parse(f"UPDATE events SET action = {x['effect']} WHERE "
                          f"{L.predicate(tuple(x['combo']))} SCOPE ALL").text for x in arms]
    k0 = {}
    for t in texts:
        k0[t] = {tid: k for tid, k in eng.con.execute(Change.parse(t).first_match_sql()) if tid in half}

    # candidate pairs: same effect, many shared cells, not identical
    by_eff = collections.defaultdict(list)
    for t in texts:
        c = Change.parse(t)
        by_eff[c.effect + ":" + c.arg].append(t)
    cand = []
    for eff, ts in by_eff.items():
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                A, B = k0[ts[i]], k0[ts[j]]
                U = set(A) | set(B)
                shared = [x for x in set(A) & set(B) if A[x] == B[x]]
                if not U or len(shared) / len(U) < a.min_shared or len(shared) == len(U) and A == B:
                    continue
                cand.append((ts[i], ts[j], len(shared) / len(U)))
    rng = random.Random(a.seed)
    rng.shuffle(cand)
    pairs = cand[: a.pairs]
    log(f"{len(cand)} eligible pairs, using {len(pairs)}")

    # cached independent replays (rep 600)
    cache = collections.defaultdict(dict)
    for t in texts:
        for tid, won, calls in eng.con.execute("SELECT tid, won, calls FROM runs WHERE change=? AND rep=? "
                                               "AND k0>=0", (t, INDEP_REP)):
            cache[t][tid] = (float(won), calls)

    # jobs: per pair, per task in the union
    jobs = []
    for pi, (c1, c2, frac) in enumerate(pairs):
        A, B = k0[c1], k0[c2]
        for tid in set(A) | set(B):
            if tid in A and tid in B and A[tid] == B[tid]:
                jobs.append(("cell", pi, tid, A[tid], [c1, c2]))
            else:
                if tid in A:
                    jobs.append(("solo", pi, tid, A[tid], [c1]))
                if tid in B:
                    jobs.append(("solo", pi, tid, B[tid], [c2]))
    log(f"{len(jobs)} jobs")
    res = collections.defaultdict(dict)       # (pi, tid) -> {text: run}

    def load_done():
        """Replays already in the store for each pair's own seed family (resume)."""
        for pi, (c1, c2, frac) in enumerate(pairs):
            for c in (c1, c2):
                for tid, won, calls in eng.con.execute("SELECT tid, won, calls FROM runs WHERE change=? AND rep=? "
                                                       "AND k0>=0", (c, a.rep + pi)):
                    res[(pi, tid)][c] = {"won": won, "calls": calls, "change": c}

    t0 = time.time()
    for attempt in range(3):
        load_done()
        todo = [j for j in jobs if not all(c in res[(j[1], j[2])] for c in j[4])]
        log(f"attempt {attempt + 1}: {len(todo)} jobs to run")
        if not todo:
            break
        done = 0
        with ProcessPoolExecutor(a.workers) as ex:
            futs = {}
            for kind, pi, tid, k, tx in todo:
                if kind == "cell":
                    fu = ex.submit(run_cell, a.base, traces[tid], eng.events(tid), k, a.rep + pi, tx)
                else:
                    fu = ex.submit(replay_one, tx[0], a.base, traces[tid], eng.events(tid), k, a.rep + pi)
                futs[fu] = (kind, pi, tid)
            for fu in as_completed(futs):
                kind, pi, tid = futs[fu]
                try:
                    rows = fu.result()
                    rows = rows if isinstance(rows, list) else [rows]
                    for r in rows:
                        eng._insert_run(r)
                        res[(pi, tid)][r["change"]] = r
                except Exception as e:
                    log("job failed:", repr(e)[:200])
                done += 1
                if done % 200 == 0:
                    log(f"  {done}/{len(todo)} {int(time.time() - t0)}s")

    out = open(a.out, "w")
    for pi, (c1, c2, frac) in enumerate(pairs):
        A, B = k0[c1], k0[c2]
        U = sorted(set(A) | set(B))
        d = {"coupled": 0.0, "seeded": 0.0, "independent": 0.0}
        gam = {"coupled": 0.0, "seeded": 0.0, "independent": 0.0}
        calls_c2_coupled = calls_c2_alone = 0
        ok = True
        for tid in U:
            r = res.get((pi, tid), {})
            z1 = r[c1]["won"] if c1 in r else (Y[tid] if tid not in A else None)
            z2 = r[c2]["won"] if c2 in r else (Y[tid] if tid not in B else None)
            s1 = cache[c1].get(tid, (Y[tid], 0))[0] if tid in A else Y[tid]
            s2 = cache[c2].get(tid, (Y[tid], 0))[0] if tid in B else Y[tid]
            if (tid in A and tid not in cache[c1]) or (tid in B and tid not in cache[c2]) or z1 is None or z2 is None:
                ok = False
                continue
            d["coupled"] += float(z1 != z2)
            d["seeded"] += float(s1 != s2)
            d["independent"] += float(z1 != s2)
            gam["coupled"] += z1 - z2
            gam["seeded"] += s1 - s2
            gam["independent"] += z1 - s2
            second = sorted([c1, c2])[1]          # run_cell replays the pair in sorted order
            if tid in A and tid in B and A[tid] == B[tid] and second in r:
                calls_c2_coupled += r[second]["calls"]
                calls_c2_alone += cache[second][tid][1]
        rec = {"pair": [c1, c2], "w_union": len(U) / N, "shared_frac": frac, "complete": ok,
               "d": {k: v / N for k, v in d.items()}, "gamma": {k: v / N for k, v in gam.items()},
               "calls_c2_coupled": calls_c2_coupled, "calls_c2_alone": calls_c2_alone}
        out.write(json.dumps(rec) + "\n")
    out.close()
    rows = [json.loads(l) for l in open(a.out)]
    import statistics as st
    for k in ["independent", "seeded", "coupled"]:
        log(f"  d {k:11s}: mean {100 * st.mean(r['d'][k] for r in rows):.2f} pts")
    rat = [r["d"]["independent"] / max(r["d"]["coupled"], 1 / N) for r in rows]
    log(f"  ratio d_indep / d_coupled: median {st.median(rat):.1f}, mean {st.mean(rat):.1f}")
    cc = sum(r["calls_c2_coupled"] for r in rows)
    ca = sum(r["calls_c2_alone"] for r in rows)
    log(f"  model calls of the second change on shared cells: coupled {cc} vs standalone {ca} "
        f"({ca / max(cc, 1):.1f}x fewer)")
    log("PAIR_DONE")


if __name__ == "__main__":
    main()
