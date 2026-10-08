"""Check that our harness rebuilds every logged AgentErrorBench prompt exactly.

For each ALFWorld trajectory: start the game, and at every step compare the
prompt we build with the logged user message, then apply the logged reply.
Expected output: exact prompt match on all steps: 100/100; won during replay: 0
"""
import argparse
import difflib
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aeb  # noqa: E402


def check(tr: aeb.Trace):
    ep = aeb.Episode.start(tr.gamefile)
    try:
        for k, (p, raw) in enumerate(zip(tr.prompts, tr.raws)):
            if ep.done:
                return tr.tid, False, k, "episode ended before the log did", ep.won
            built = ep.prompt()
            if built != p:
                diff = "".join(list(difflib.unified_diff(
                    p.splitlines(True), built.splitlines(True), "logged", "built", n=1))[:30])
                return tr.tid, False, k, diff, ep.won
            ep.act(raw, built)
        return tr.tid, True, len(tr.prompts), "", ep.won
    finally:
        ep.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--show", type=int, default=3, help="print this many mismatch diffs")
    a = ap.parse_args()

    traces = aeb.load_aeb(a.models)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(check, traces))
    ok = sum(r[1] for r in res)
    won = sum(bool(r[4]) for r in res)
    by_model = {}
    for tr, r in zip(traces, res):
        m = by_model.setdefault(tr.model, [0, 0])
        m[0] += r[1]
        m[1] += 1
    for m, (g, n) in sorted(by_model.items()):
        print(f"  {m}: {g}/{n}")
    shown = 0
    for r in res:
        if not r[1] and shown < a.show:
            print(f"--- {r[0]} first mismatch at step {r[2]}\n{r[3]}")
            shown += 1
    print(f"exact prompt match on all steps: {ok}/{len(res)}; won during replay: {won}")


if __name__ == "__main__":
    main()
