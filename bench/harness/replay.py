"""Replay logged trajectories from a fork point k.

fidelity: at step k rebuild the logged prompt, sample the local model `reps`
          times, and compare each projected action with the logged action.
          README: Qwen3-8B local agreement is about 0.76.
continue: restore the logged prefix of length k and let the policy play to the
          end; one output row per (trace, k, rep) with the outcome.

  python harness/replay.py --mode fidelity --models Qwen3-8B --ks all --reps 4
  python harness/replay.py --mode continue --source fresh:store/Qwen3-8B-fresh.jsonl --served qwen3-8b --ks 0,5,10 --reps 8

Output is jsonl and resumable: rows already present are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aeb  # noqa: E402
import llm  # noqa: E402

SERVED = {"Qwen3-8B": "qwen3-8b", "Qwen2.5-7B": "qwen2.5-7b"}
_policy = None


def _get_policy(name):
    global _policy
    if _policy is None or _policy.name != name:
        _policy = llm.Policy(name)
    return _policy


def fidelity_task(tr: aeb.Trace, k: int, reps: int, served: str):
    pol = _get_policy(served)
    ep = aeb.restore(tr.gamefile, tr.actions[:k])
    try:
        p = ep.prompt()
        prompt_ok = p == tr.prompts[k]
        logged = tr.actions[k]
        acts = [aeb.project(pol(p))[0] for _ in range(reps)]
        return {"tid": tr.tid, "k": k, "logged": logged, "acts": acts,
                "agree": sum(a == logged for a in acts), "reps": reps, "prompt_ok": prompt_ok}
    finally:
        ep.close()


def continue_task(tr: aeb.Trace, k: int, rep: int, served: str):
    pol = _get_policy(served)
    t0 = time.time()
    c0 = pol.usage()
    ep = aeb.restore(tr.gamefile, tr.actions[:k])
    try:
        steps = aeb.run_policy(ep, pol)
        c1 = pol.usage()
        return {"tid": tr.tid, "k": k, "rep": rep, "won": int(ep.won), "n_steps": ep.k,
                "actions": [s.action for s in steps], "raws": [s.raw for s in steps],
                "prompt_tokens": c1["prompt_tokens"] - c0["prompt_tokens"],
                "completion_tokens": c1["completion_tokens"] - c0["completion_tokens"],
                "calls": c1["calls"] - c0["calls"], "secs": round(time.time() - t0, 1)}
    finally:
        ep.close()


def load_source(src: str, models):
    if src == "aeb":
        return aeb.load_aeb(models)
    kind, path = src.split(":", 1)
    assert kind == "fresh"
    return aeb.load_jsonl(path)


def parse_ks(spec: str, n: int):
    if spec == "all":
        return list(range(n))
    return [k for k in (int(x) for x in spec.split(",")) if k < n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fidelity", "continue"], required=True)
    ap.add_argument("--source", default="aeb", help="aeb | fresh:<jsonl>")
    ap.add_argument("--models", nargs="*", default=None, help="AgentErrorBench model filter")
    ap.add_argument("--served", default=None, help="served model name (default from --models)")
    ap.add_argument("--ks", default="all")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--only-failed", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N traces only")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    traces = load_source(a.source, a.models)
    if a.only_failed:
        traces = [t for t in traces if not t.won]
    if a.limit:
        traces = traces[:a.limit]
    served = a.served or SERVED[a.models[0]]
    out = a.out or os.path.join(aeb.ROOT, "store", f"replay_{a.mode}_{served}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for line in f:
                d = json.loads(line)
                done.add((d["tid"], d["k"], d.get("rep", -1)))

    jobs = []
    for tr in traces:
        for k in parse_ks(a.ks, len(tr.raws)):
            if a.mode == "fidelity":
                if (tr.tid, k, -1) not in done:
                    jobs.append((fidelity_task, (tr, k, a.reps, served)))
            else:
                for r in range(a.reps):
                    if (tr.tid, k, r) not in done:
                        jobs.append((continue_task, (tr, k, r, served)))
    print(f"{len(traces)} traces, {len(jobs)} jobs to run, {len(done)} already done -> {out}", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex, open(out, "a") as fo:
        futs = [ex.submit(fn, *args) for fn, args in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                fo.write(json.dumps(fu.result()) + "\n")
                fo.flush()
            except Exception as e:  # keep going, report at the end
                print("job failed:", repr(e)[:300], flush=True)
            if i % 50 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}  {time.time() - t0:.0f}s", flush=True)

    summarize(a.mode, out, {t.tid for t in traces})


def summarize(mode, out, tids):
    rows = [json.loads(l) for l in open(out)]
    rows = [r for r in rows if r["tid"] in tids]
    if mode == "fidelity":
        agree = sum(r["agree"] for r in rows)
        tot = sum(r["reps"] for r in rows)
        bad = sum(not r["prompt_ok"] for r in rows)
        print(f"action agreement: {agree}/{tot} = {agree / max(tot, 1):.3f}; prompt mismatches: {bad}")
        byk = defaultdict(lambda: [0, 0])
        for r in rows:
            byk[r["k"]][0] += r["agree"]
            byk[r["k"]][1] += r["reps"]
        print("by k:", " ".join(f"{k}:{a / n:.2f}" for k, (a, n) in sorted(byk.items())))
    else:
        byk = defaultdict(list)
        for r in rows:
            byk[r["k"]].append(r["won"])
        print("V_k (mean win when continuing from k):",
              " ".join(f"{k}:{sum(v) / len(v):.3f}(n={len(v)})" for k, v in sorted(byk.items())))


if __name__ == "__main__":
    main()
