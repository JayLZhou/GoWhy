"""Counterfactual Store Iteration (CSI): improve an agent by a chain of certified changes,
each found on a store materialized from the previous change's replays, never redeploying.

Round r:
  1. search the lattice on store S_{r-1} for a change C_r, every candidate replayed under the
     patch P_{r-1} ;; C (gowhy/lattice_gai.py --prefix P_{r-1});
  2. certify C_r on fresh seeds (one replay per affected episode, step logs kept);
  3. materialize S_r: every episode C_r affects is replaced by its certification replay
     (logged prefix up to the fork + the replayed suffix), every other episode is kept.

Store-update theorem (THEORY.md section 23): each replayed episode is an exact sample of the
patched policy P_r on that task, and unaffected episodes are identical under P_{r-1} and P_r,
so S_r has the law of a store collected by deploying P_r. Round r's certificate is therefore a
valid statement about P_r against P_{r-1}; with level delta per round, all R certificates hold
together with probability >= 1 - R * delta, and the certified gains add up.

  python gowhy/csi.py --db store/fresh_q25.db --half B --base qwen2.5-7b --rounds 3 --tag csi1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "harness"))


def materialize(src_db, dst_db, full_text, rep, half, base, workers=48):
    """S_r from S_{r-1}: replace each episode affected by the last change of full_text by its
    replay (change=full_text, rep) and recompute its events with the store's own pipeline."""
    import aeb
    import store
    from changes import parse_patch
    from engine import Engine, half_of, replay_one, unpack_steps

    last = parse_patch(full_text)[-1] if ";;" in full_text else __import__("changes").Change.parse(full_text)
    src = store.connect(src_db)
    src.row_factory = __import__("sqlite3").Row
    traces = {r["tid"]: dict(r) for r in src.execute("SELECT * FROM traces")
              if not half or half_of(r["tid"]) == half}
    raws = {t: [r[0] for r in src.execute("SELECT raw FROM events WHERE tid=? ORDER BY k", (t,))] for t in traces}
    k0 = {t: k for t, k in src.execute(last.first_match_sql()) if t in traces}
    runs = {}
    for t, kk, won, steps in src.execute("SELECT tid, k0, won, steps FROM runs WHERE change=? AND rep=? AND k0>=0",
                                         (full_text, rep)):
        if t in k0 and steps:
            runs[t] = (kk, won, unpack_steps(steps))
    missing = [t for t in k0 if t not in runs]
    if missing:                                  # replays that failed during certification: redo them
        print(f"  materialize: re-running {len(missing)} missing replays", flush=True)
        eng = Engine(src_db, base, workers)
        with ProcessPoolExecutor(workers) as ex:
            futs = {ex.submit(replay_one, full_text, base, traces[t], eng.events(t), k0[t], rep, None, True): t
                    for t in missing}
            for fu, t in futs.items():
                r = fu.result()
                eng._insert_run(r)
                runs[t] = (r["k0"], r["won"], unpack_steps(r["steps"]))
    trs = []
    for t, tr in traces.items():
        if t in runs:
            kk, won, steps = runs[t]
            rr = raws[t][:kk] + [s["final"] for s in steps]
        else:
            rr, won = raws[t], tr["won"]
        trs.append(aeb.Trace(t, tr["model"], tr["gamefile"], tr["task_type"], [], rr, bool(won)))
    if os.path.exists(dst_db):
        os.remove(dst_db)
    dst = store.connect(dst_db)
    for tr, t in zip(trs, traces):
        src_tr = traces[t]
        dst.execute("INSERT INTO traces(tid,model,gamefile,task_type,obj,recep,won,n_steps,calls,prompt_tokens,"
                    "completion_tokens) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (t, src_tr["model"], src_tr["gamefile"], src_tr["task_type"], src_tr["obj"], src_tr["recep"],
                     int(tr.won), len(tr.raws), None, None, None))
    mism = 0
    with ProcessPoolExecutor(workers) as ex:
        for tid, rows, won, task, score in ex.map(store.trace_events, trs, chunksize=2):
            dst.executemany("INSERT OR REPLACE INTO events(tid,k,action,raw,verb,valid,fmt_valid,quoted,revisit,"
                            "repeat,obj_match,n_invalid,obs_after,recep_match,obs) "
                            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            dst.execute("UPDATE traces SET task=?, score=? WHERE tid=?", (task, score, tid))
            logged = next(x.won for x in trs if x.tid == tid)
            if bool(won) != bool(logged):
                mism += 1
                dst.execute("UPDATE traces SET won=? WHERE tid=?", (int(won), tid))
    dst.commit()
    succ = dst.execute("SELECT AVG(won), COUNT(*) FROM traces").fetchone()
    print(f"  materialized {dst_db}: {succ[1]} episodes ({len(runs)} replaced), success {succ[0]:.3f}, "
          f"outcome re-check mismatches {mism}", flush=True)
    return {"episodes": succ[1], "replaced": len(runs), "success": succ[0], "mismatch": mism}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--half", default="B")
    ap.add_argument("--base", default="qwen2.5-7b")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--method", default="lin")
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--tag", default="csi1")
    ap.add_argument("--effects", nargs="*", default=None, help="effects searched each round (lattice_gai --effects)")
    ap.add_argument("--propose", default=None, help="proposer model: each round, it reads the current store's "
                                                    "failures and proposes targeted changes that join the search")
    ap.add_argument("--extra", default=None, help="fixed file of extra candidates added to every round's search")
    ap.add_argument("--start-round", type=int, default=1)
    ap.add_argument("--start-db", default=None)
    ap.add_argument("--start-prefix", default="")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(a.db))) if os.path.dirname(a.db) else "."
    py = sys.executable
    summ_path = os.path.join("store", f"{a.tag}_summary.jsonl")
    db, prefix = (a.start_db or a.db), a.start_prefix
    for r in range(a.start_round, a.rounds + 1):
        out = os.path.join("store", f"{a.tag}_r{r}.jsonl")
        if os.path.exists(out):
            os.remove(out)
        cmd = [py, "-u", os.path.join(HERE, "lattice_gai.py"), "--db", db, "--half", a.half, "--base", a.base,
               "--method", a.method, "--search-budget", str(a.budget), "--delta", str(a.delta),
               "--seed", str(a.seed), "--workers", str(a.workers), "--out", out,
               "--cert-rep", str(750 + 10 * r), "--search-rep", "600"]
        if prefix:
            cmd += ["--prefix", prefix]
        if a.effects:
            cmd += ["--effects"] + a.effects
        if a.propose:
            props = os.path.join("store", f"{a.tag}_r{r}_props.txt")
            subprocess.run([py, "-u", os.path.join(HERE, "propose.py"), "--db", db, "--half", a.half,
                            "--proposer", a.propose, "--per-trace", "2", "--keep", "12", "--min-support", "3",
                            "--replan", "--out", props], check=True)
            cmd += ["--extra", props]
        elif a.extra:
            cmd += ["--extra", a.extra]
        print(f"[round {r}] store {db}; patch so far: {prefix or '(none)'}", flush=True)
        t0 = time.time()
        subprocess.run(cmd, check=True)
        res = json.loads(open(out).read().strip().splitlines()[-1])
        rec = {"round": r, "db": db, "prefix": prefix, "certified": res["certified"], "attempts": res["attempts"],
               "search_requested": res["search_requested"], "total_executed": res["total_executed"],
               "secs": int(time.time() - t0)}
        if not res["certified"]:
            print(f"[round {r}] nothing certified; stopping", flush=True)
            with open(summ_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            break
        c = res["certified"]
        att = next(x for x in res["attempts"] if x["arm"] == c["arm"])
        full_text = att["full"]
        new_db = os.path.join("store", f"{a.tag}_r{r}.db")
        print(f"[round {r}] certified {100 * c['cert']:+.2f} pts (LCB {100 * c['lcb']:+.2f}): {c['arm'][27:]}",
              flush=True)
        rec["materialize"] = materialize(db, new_db, full_text, att["rep"], a.half, a.base)
        rec["patch"] = full_text
        with open(summ_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        db, prefix = new_db, full_text
    print("CSI_DONE", json.dumps({"patch": prefix}), flush=True)


if __name__ == "__main__":
    main()
