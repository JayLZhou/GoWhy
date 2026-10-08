"""Held-out A/B test: base policy vs a changed policy, from scratch, on new games.

The what-if and HOWTO results are measured on the games in the store. This
checks whether a change found there carries over to games never seen during
mining or testing: ALFWorld's evaluation splits (valid_seen: same scenes,
new tasks; valid_unseen: new scenes). Both arms play every game m times.

  python harness/ab.py --served qwen3-8b --m 2 \
      --change "UPDATE events SET action = hint:2 WHERE verb = 'take' AND obj_match = 0 SCOPE ALL"

The base arm is the null change "... WHERE k < 0 ...", which never fires, run
through the same engine, so both arms share every line of code except the change.
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "gowhy"))
import aeb  # noqa: E402
import store  # noqa: E402
from engine import replay_one  # noqa: E402

NULL = "UPDATE events SET action = reject:1 WHERE k < 0 SCOPE ALL"


def trace_of(gamefile: str, split: str) -> dict:
    obj, recep = store.game_parts(gamefile)
    if gamefile.startswith("webshop:"):
        tid = split + "/" + gamefile
    else:
        tid = split + "/" + gamefile.split("/json_2.1.1/")[1].split("/")[1] + "/" + gamefile.split("/")[-2]
    return {"tid": tid, "gamefile": gamefile, "obj": obj, "recep": recep,
            "task_type": aeb.task_type_of(gamefile)}


def run(arm, change, served, trace, rep):
    r = replay_one(change, served, trace, [], -1, rep)
    return {"arm": arm, "tid": trace["tid"], "split": trace["tid"].split("/")[0], "rep": rep,
            "won": r["won"], "score": r.get("score", r["won"]), "n_steps": r["n_steps"], "calls": r["calls"],
            "tokens": r["prompt_tokens"] + r["completion_tokens"]}


def mean(x):
    return sum(x) / len(x)


def se(x):
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1) / len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--change", required=True)
    ap.add_argument("--served", default="qwen3-8b")
    ap.add_argument("--splits", default="valid_seen,valid_unseen")
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--workers", type=int, default=160)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = a.out or os.path.join(aeb.ROOT, "store", f"ab_{a.served}.jsonl")

    def games(split):
        if split.startswith("webshop_"):  # webshop_test or webshop_train
            import webshop
            return webshop.goals(split.split("_", 1)[1])
        return aeb.game_files(split)

    traces = [trace_of(g, s) for s in a.splits.split(",") for g in games(s)]
    if a.limit:
        traces = traces[:a.limit]
    arms = {"base": NULL, "change": a.change}
    done = set()
    if os.path.exists(out):
        for line in open(out):
            d = json.loads(line)
            done.add((d["arm"], d["tid"], d["rep"]))
    jobs = [(arm, t, r) for t in traces for arm in arms for r in range(a.m) if (arm, t["tid"], r) not in done]
    print(f"{len(traces)} games x {len(arms)} arms x m={a.m}: {len(jobs)} runs to go -> {out}", flush=True)

    t0 = time.time()
    import multiprocessing as mp
    ctx = mp.get_context("spawn") if any(t["gamefile"].startswith("webshop:") for t in traces) else None
    with ProcessPoolExecutor(a.workers, mp_context=ctx) as ex, open(out, "a") as fo:
        futs = [ex.submit(run, arm, arms[arm], a.served, t, r) for arm, t, r in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                fo.write(json.dumps(fu.result()) + "\n")
                fo.flush()
            except Exception as e:
                print("run failed:", repr(e)[:200], flush=True)
            if i % 100 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} {time.time() - t0:.0f}s", flush=True)

    rows = [json.loads(l) for l in open(out)]
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["tid"]][r["arm"]].append(r["won"])
    print(f"\nchange: {a.change}")
    for split in a.splits.split(",") + ["all"]:
        tids = [t for t in by if (split == "all" or t.startswith(split + "/")) and by[t]["base"] and by[t]["change"]]
        b = [mean(by[t]["base"]) for t in tids]
        c = [mean(by[t]["change"]) for t in tids]
        d = [x - y for x, y in zip(c, b)]
        print(f"  [{split}] {len(tids)} games: base {100 * mean(b):.1f}%, change {100 * mean(c):.1f}%, "
              f"paired diff {100 * mean(d):+.1f} pts (SE {100 * se(d):.1f}, 95% CI "
              f"{100 * (mean(d) - 1.96 * se(d)):+.1f}..{100 * (mean(d) + 1.96 * se(d)):+.1f})")


if __name__ == "__main__":
    main()
