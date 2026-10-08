"""Mine candidate changes from half A of a store; test them on half B.

Hypotheses come from association with failure, which is not causation: that is
what the replay test on the other half is for. Mining and testing must use
disjoint games. If they shared games, a predicate picked because it co-occurs
with Y = 0 would make D = V_hat - Y look positive even for a change with no
effect.

Atoms are "mistake-like" properties of a step, observable before it runs:
  valid = 0, repeat = 1, revisit = 1,
  verb = V AND obj_match = 0    (acting on an object the task does not need)
  verb = V AND recep_match = 0  (putting into a receptacle the task does not name)
Each atom is scored on half A by how much more often games containing it fail
(one-sided z against the half-A failure rate). The top K become two changes
each: reject:3 (resample) and hint:2 (feedback and retry), SCOPE ALL.

  python gowhy/mine.py store/fresh.db --k 4 > gowhy/mined_qwen3.txt
"""
from __future__ import annotations

import argparse
import math
import os
import sqlite3
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import half_of  # noqa: E402

OBJ_VERBS = ["take", "put", "move", "heat", "cool", "clean", "use", "examine"]
RECEP_VERBS = ["put", "move"]


def atoms():
    out = ["valid = 0", "repeat = 1", "revisit = 1"]
    out += [f"verb = '{v}' AND obj_match = 0" for v in OBJ_VERBS]
    out += [f"verb = '{v}' AND recep_match = 0" for v in RECEP_VERBS]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--min-support", type=int, default=8)
    a = ap.parse_args()
    con = sqlite3.connect(a.db)
    won = {t: w for t, w in con.execute("SELECT tid, won FROM traces") if half_of(t) == "A"}
    base = 1 - sum(won.values()) / len(won)
    scored = []
    for p in atoms():
        hit = {t for (t,) in con.execute(
            f"SELECT DISTINCT e.tid FROM events e JOIN traces t USING(tid) WHERE {p}") if t in won}
        n = len(hit)
        if n < a.min_support:
            print(f"# skip (support {n}): {p}", file=sys.stderr)
            continue
        f = sum(1 - won[t] for t in hit) / n
        z = (f - base) / math.sqrt(base * (1 - base) / n)
        scored.append((z, p, n, f))
    scored.sort(reverse=True)
    print(f"# mined on half A: {len(won)} games, failure rate {base:.2f}")
    for z, p, n, f in scored:
        print(f"#   z {z:5.2f}  games {n:3d}  fail {f:.2f}  {p}")
    for z, p, n, f in scored[:a.k]:
        print(f"UPDATE events SET action = reject:3 WHERE {p} SCOPE ALL")
        print(f"UPDATE events SET action = hint:2 WHERE {p} SCOPE ALL")


if __name__ == "__main__":
    main()
