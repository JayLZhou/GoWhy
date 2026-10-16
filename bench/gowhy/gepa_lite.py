"""Prompt-optimization baseline in the style of GEPA (reflective prompt evolution), same budget.

The search space is an instruction appended to every prompt (effect guide:1:ID). Each
iteration:
  1. parent = the candidate with the best success on the full training set (the base
     policy's logged outcomes for the empty instruction);
  2. reflect: show the same model R failed training episodes of the parent (task, actions,
     outcome) and its current instruction; it writes an improved instruction;
  3. evaluate the new instruction from scratch on a minibatch of training games and compare
     with the parent on the same games; if it is at least as good, evaluate it on the full
     training set and add it to the pool.
The budget counts every model call the agent makes (reflection calls are counted too), so
it can be matched to GoWhy's search + certification.

  python gowhy/gepa_lite.py --db store/fresh_q25.db --half B --budget-calls 25000 \
      --msgs store/gepa_msgs.json --out store/gepa_q25.jsonl
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REFLECT = """You are improving the instructions of an LLM agent that acts in ALFWorld, a text-based household \
environment. The agent reads the task and admissible actions at every step and replies with <plan> and <action> tags.

Current additional instruction given to the agent (may be empty):
<instruction>
{current}
</instruction>

Here are {n} episodes in which the agent FAILED (it ran out of steps without finishing the task):
{episodes}

Diagnose what went wrong across these failures, then write an improved additional instruction that would help the \
agent succeed on tasks like these. It will be appended to every prompt the agent sees, so keep it general (not \
specific to one episode), concrete, and at most 80 words. Output it inside <instruction></instruction> tags."""


def episode_text(task, actions, max_actions=30):
    acts = actions[:max_actions]
    return f"Task: {task}\nActions taken: " + "; ".join(acts) + "\nOutcome: failed"


def lock_update(path, key, text):
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = json.load(open(path)) if os.path.exists(path) else {}
        d[key] = text
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--half", default="B")
    ap.add_argument("--base", default="qwen2.5-7b")
    ap.add_argument("--budget-calls", type=int, default=25000)
    ap.add_argument("--minibatch", type=int, default=10)
    ap.add_argument("--reflect-n", type=int, default=3)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--msgs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["GOWHY_MSGS"] = os.path.abspath(a.msgs)   # before engine/changes are imported
    from engine import Engine, policy, replay_one        # noqa: E402

    eng = Engine(a.db, a.base, a.workers, half=a.half)
    traces = eng.traces()
    tids = sorted(traces)
    rng = random.Random(a.seed)
    logged_actions = {t: [r[0] for r in eng.con.execute("SELECT action FROM events WHERE tid=? ORDER BY k", (t,))]
                      for t in tids}
    root = {"id": "", "text": "", "full": {t: float(traces[t]["won"]) for t in tids},
            "actions": logged_actions}
    pool_c = [root]
    used = {"calls": 0}
    log = open(a.out, "a")
    reflector = policy(a.base)
    ex = ProcessPoolExecutor(a.workers)
    it = 0

    def change(cid):
        return f"UPDATE events SET action = guide:1:{cid} WHERE k >= 0 SCOPE ALL"

    def rollouts(cid, games, rep):
        futs = {ex.submit(replay_one, change(cid), a.base, traces[t], [], -1, rep): t for t in games}
        out, acts = {}, {}
        for fu in as_completed(futs):
            t = futs[fu]
            try:
                r = fu.result()
            except Exception as e:
                print("rollout failed:", repr(e)[:200], flush=True)
                continue
            used["calls"] += r["calls"]
            out[t] = float(r["won"])
            acts[t] = json.loads(r["actions"])
            eng._insert_run(r)
        return out, acts

    t0 = time.time()
    while used["calls"] < a.budget_calls:
        it += 1
        parent = max(pool_c, key=lambda c: sum(c["full"].values()) / len(c["full"]))
        failed = [t for t in tids if parent["full"].get(t) == 0.0 and t in parent["actions"]]
        shown = rng.sample(failed, min(a.reflect_n, len(failed)))
        eps = "\n\n".join(f"Episode {j + 1}:\n" + episode_text(traces[t]["task"], parent["actions"][t])
                          for j, t in enumerate(shown))
        reply = reflector(REFLECT.format(current=parent["text"] or "(none)", n=len(shown), episodes=eps),
                          10_000 + 97 * it + a.seed)
        used["calls"] += 1
        m = re.search(r"<instruction>(.*?)</instruction>", reply, re.S | re.I)
        text = " ".join((m.group(1) if m else reply[-600:]).split())[:800]
        cid = f"gepa{a.seed}_{it}"
        lock_update(a.msgs, cid, text)
        mb = rng.sample(tids, a.minibatch)
        rep = 800 + it
        new_mb, _ = rollouts(cid, mb, rep)
        par_mb = sum(parent["full"][t] for t in mb if t in new_mb)
        new_s = sum(new_mb.values())
        rec = {"iter": it, "id": cid, "parent": parent["id"], "text": text, "minibatch": new_s,
               "parent_minibatch": par_mb, "calls_used": used["calls"]}
        if new_s >= par_mb and new_s > 0 and used["calls"] < a.budget_calls:
            full, acts = rollouts(cid, tids, rep)
            cand = {"id": cid, "text": text, "full": full, "actions": acts}
            pool_c.append(cand)
            rec["full"] = sum(full.values()) / len(full)
            rec["calls_used"] = used["calls"]
        print(json.dumps(rec)[:400], f"{int(time.time() - t0)}s", flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
    best = max(pool_c, key=lambda c: sum(c["full"].values()) / len(c["full"]))
    summ = {"final": True, "best_id": best["id"], "best_text": best["text"],
            "best_train": sum(best["full"].values()) / len(best["full"]),
            "base_train": sum(root["full"].values()) / len(root["full"]),
            "pool": len(pool_c), "iters": it, "calls_used": used["calls"],
            "change": change(best["id"]) if best["id"] else None}
    print(json.dumps(summ), flush=True)
    log.write(json.dumps(summ) + "\n")
    ex.shutdown()


if __name__ == "__main__":
    main()
