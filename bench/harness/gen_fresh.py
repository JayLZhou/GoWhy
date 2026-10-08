"""Play the agent from scratch on new ALFWorld games and save full trajectories.

  python harness/gen_fresh.py --n 300 --model Qwen2.5-7B-fresh --served qwen2.5-7b
  python harness/gen_fresh.py --n 300 --model Qwen3-8B-fresh --served qwen3-8b

Games: train split, alfworld's own filter, shuffled with --seed, first n.
Games that appear in AgentErrorBench are skipped so the two sets do not overlap.
Each line: tid, model, gamefile, won, n_steps, usage, steps[{raw, action,
fmt_valid, admissible, obs_after}]. Prompts are not stored: aeb.restore()
rebuilds them exactly from the gamefile and the actions.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aeb  # noqa: E402
import llm  # noqa: E402

_policy = None


def play(tid: str, model: str, gamefile: str, served: str, ports):
    global _policy
    if _policy is None:
        _policy = llm.Policy(served, ports=ports)
    t0 = time.time()
    u0 = _policy.usage()
    ep = aeb.start_episode(gamefile)
    try:
        steps = aeb.run_policy(ep, _policy)
    finally:
        ep.close()
    u1 = _policy.usage()
    return {"tid": tid, "model": model, "gamefile": gamefile, "won": int(ep.won), "n_steps": ep.k,
            "usage": {k: u1[k] - u0[k] for k in u1}, "secs": round(time.time() - t0, 1),
            "steps": [{"raw": s.raw, "action": s.action, "fmt_valid": s.fmt_valid,
                       "admissible": s.admissible, "obs_after": s.obs_after} for s in steps]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", required=True, help="label, e.g. Qwen3-8B-fresh")
    ap.add_argument("--served", required=True)
    ap.add_argument("--ports", type=int, nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", default=None)
    ap.add_argument("--env", choices=["alfworld", "webshop"], default="alfworld")
    a = ap.parse_args()

    out = a.out or os.path.join(aeb.ROOT, "data", "fresh", f"{a.model}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # One writer per output file: two concurrent runs once wrote 599 rows for 300 games.
    import fcntl
    lock = open(out + ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"another gen_fresh.py is already writing {out}")
    if a.env == "webshop":
        import webshop
        games, used = webshop.goals("train"), set()
    else:
        games = aeb.game_files("train")
        try:
            used = {aeb.resolve_gamefile(t.gamefile) for t in aeb.load_aeb()}
        except FileNotFoundError:
            used = set()
    games = [g for g in games if g not in used]
    random.Random(a.seed).shuffle(games)
    games = games[:a.n]

    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["tid"] for l in open(out)}
    jobs = [(f"{a.model}_{i:03d}", g) for i, g in enumerate(games) if f"{a.model}_{i:03d}" not in done]
    print(f"{len(games)} games, {len(jobs)} to play -> {out}", flush=True)

    t0, won, n = time.time(), 0, 0
    import multiprocessing as mp
    # WebShop runs Lucene in a JVM (pyjnius); a JVM does not survive fork, so use spawn
    ctx = mp.get_context("spawn") if a.env == "webshop" else None
    with ProcessPoolExecutor(a.workers, mp_context=ctx) as ex, open(out, "a") as fo:
        futs = [ex.submit(play, tid, a.model, g, a.served, a.ports) for tid, g in jobs]
        for fu in as_completed(futs):
            try:
                r = fu.result()
            except Exception as e:
                print("game failed:", repr(e)[:300], flush=True)
                continue
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            n += 1
            won += r["won"]
            if n % 25 == 0 or n == len(futs):
                print(f"  {n}/{len(futs)} played, success {won / n:.3f}, {time.time() - t0:.0f}s", flush=True)

    rows = [json.loads(l) for l in open(out)]
    print(f"total {len(rows)} games, success rate {sum(r['won'] for r in rows) / len(rows):.3f}")


if __name__ == "__main__":
    main()
