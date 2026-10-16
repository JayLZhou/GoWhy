"""HOWTO over a lattice of predicates: candidate space, sound bounds, pruning.

Candidates are conjunctions of up to --max-atoms atomic conditions on step
attributes, each combined with every effect in --effects (SCOPE ALL).

Sound bounds from the logs alone (no replays). For predicate phi, F(phi) is the set
of episodes in which phi fires on the logged trajectory. Whatever the effect,
    -|F ∩ succeeded| / N  <=  Delta(phi, e)  <=  |F ∩ failed| / N =: UB(phi)
because an affected episode's outcome can at most move from 0 to 1 or from 1 to 0,
and unaffected episodes do not change (Theorem 1). Adding an atom can only shrink
F, so UB is anti-monotone along the lattice: once UB(phi) < LB* (a certified lower
bound of the incumbent), phi, all its refinements, and all its effects are pruned.

  python gowhy/lattice.py stats --db store/fresh.db --half B
"""
from __future__ import annotations

import argparse
import collections
import itertools
import os
import sqlite3
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import half_of  # noqa: E402

ALF_VERBS = ["go", "take", "put", "move", "open", "close", "examine", "use", "heat", "cool", "clean", "look",
             "inventory"]
TASKS = ["pick_and_place", "pick_two_obj_and_place", "look_at_obj_in_light", "pick_heat_then_place_in_recep",
         "pick_cool_then_place_in_recep", "pick_clean_then_place_in_recep"]


def atoms_alfworld():
    a = [(f"verb = '{v}'", lambda e, v=v: e["verb"] == v) for v in ALF_VERBS]
    a += [("valid = 0", lambda e: e["valid"] == 0), ("repeat = 1", lambda e: e["repeat"] == 1),
          ("revisit = 1", lambda e: e["revisit"] == 1),
          ("obj_match = 0", lambda e: e["obj_match"] == 0), ("obj_match = 1", lambda e: e["obj_match"] == 1),
          ("recep_match = 0", lambda e: e["recep_match"] == 0), ("recep_match = 1", lambda e: e["recep_match"] == 1),
          ("n_invalid >= 1", lambda e: e["n_invalid"] >= 1), ("n_invalid >= 3", lambda e: e["n_invalid"] >= 3),
          ("k < 5", lambda e: e["k"] < 5), ("k >= 5", lambda e: e["k"] >= 5), ("k >= 10", lambda e: e["k"] >= 10),
          ("k >= 15", lambda e: e["k"] >= 15), ("k >= 20", lambda e: e["k"] >= 20)]
    a += [(f"task_type = '{t}'", lambda e, t=t: e["task_type"] == t) for t in TASKS]
    return a


SW_VERBS = ["go", "look", "focus", "pick", "put", "move", "open", "close", "activate", "deactivate", "use",
            "connect", "disconnect", "wait", "read", "pour", "mix", "teleport", "inventory", "dunk"]


def atoms_sciworld():
    a = [(f"verb = '{v}'", lambda e, v=v: e["verb"] == v) for v in SW_VERBS]
    a += [("valid = 0", lambda e: e["valid"] == 0), ("repeat = 1", lambda e: e["repeat"] == 1),
          ("n_invalid >= 3", lambda e: e["n_invalid"] >= 3), ("n_invalid >= 8", lambda e: e["n_invalid"] >= 8),
          ("k < 3", lambda e: e["k"] < 3), ("k >= 5", lambda e: e["k"] >= 5), ("k >= 10", lambda e: e["k"] >= 10),
          ("k >= 20", lambda e: e["k"] >= 20), ("action LIKE 'go %'", lambda e: e["action"].startswith("go ")),
          ("action LIKE 'go to %'", lambda e: e["action"].startswith("go to "))]
    return a


def atoms_webshop():
    a = [("verb = 'search'", lambda e: e["verb"] == "search"), ("verb = 'click'", lambda e: e["verb"] == "click"),
         ("action = 'click[buy now]'", lambda e: e["action"] == "click[buy now]"),
         ("action = 'click[next >]'", lambda e: e["action"] == "click[next >]"),
         ("action = 'click[< prev]'", lambda e: e["action"] == "click[< prev]"),
         ("action = 'click[back to search]'", lambda e: e["action"] == "click[back to search]"),
         ("action LIKE 'click[b0%'", lambda e: e["action"].startswith("click[b0")),
         ("valid = 0", lambda e: e["valid"] == 0), ("repeat = 1", lambda e: e["repeat"] == 1),
         ("k < 3", lambda e: e["k"] < 3), ("k >= 5", lambda e: e["k"] >= 5), ("k >= 8", lambda e: e["k"] >= 8),
         ("k >= 15", lambda e: e["k"] >= 15), ("k >= 22", lambda e: e["k"] >= 22)]
    return a


class Lattice:
    def __init__(self, db, half="B", max_atoms=3, metric="won"):
        con = sqlite3.connect(db)
        con.row_factory = sqlite3.Row
        tr = {r["tid"]: dict(r) for r in con.execute("SELECT * FROM traces")}
        self.tids = sorted(t for t in tr if not half or half_of(t) == half)
        self.N = len(self.tids)
        self.webshop = all(tr[t]["task_type"] == "webshop" for t in self.tids)
        self.sciworld = all(str(tr[t]["task_type"]).startswith("sciworld") for t in self.tids)
        self.atoms = atoms_webshop() if self.webshop else atoms_sciworld() if self.sciworld else atoms_alfworld()
        self.Y = np.array([float(tr[t][metric] if tr[t][metric] is not None else tr[t]["won"]) for t in self.tids])
        idx = {t: i for i, t in enumerate(self.tids)}
        # per episode: list of per-step atom bitmasks
        steps = collections.defaultdict(list)
        for r in con.execute("SELECT e.*, t.task_type FROM events e JOIN traces t USING(tid) ORDER BY e.tid, e.k"):
            if r["tid"] in idx:
                e = dict(r)
                mask = 0
                for j, (_, f) in enumerate(self.atoms):
                    if f(e):
                        mask |= 1 << j
                steps[idx[r["tid"]]].append(mask)
        self.masks = [np.array(steps[i], dtype=np.int64) for i in range(self.N)]
        self.max_atoms = max_atoms

    def affected(self, combo):
        m = 0
        for j in combo:
            m |= 1 << j
        return np.array([bool(((s & m) == m).any()) for s in self.masks])

    def predicate(self, combo):
        return " AND ".join(self.atoms[j][0] for j in combo)

    def enumerate(self, min_fail=1):
        """Apriori-style: grow conjunctions only from parents that still have failed affected episodes."""
        fail = self.Y < 1.0                                # any room to improve (continuous scores too)
        out = {}
        frontier = []
        # atoms that fire in every episode are left out by default (a change there is a redeployment);
        # GOWHY_ALLOW_FULL=1 keeps them, for stores where the failure mode is everywhere (Llama harness)
        allow_full = os.environ.get("GOWHY_ALLOW_FULL") == "1"
        for j in range(len(self.atoms)):
            F = self.affected((j,))
            if (F & fail).sum() >= min_fail and (allow_full or F.sum() < self.N):
                out[(j,)] = F
                frontier.append((j,))
        for size in range(2, self.max_atoms + 1):
            nxt = []
            seen = set()
            for combo in frontier:
                for j in range(combo[-1] + 1, len(self.atoms)):
                    c = combo + (j,)
                    if c in seen or any(tuple(x for x in c if x != y) not in out for y in c):
                        continue
                    seen.add(c)
                    F = out[combo] & self.affected((j,))
                    F = self.affected(c)                      # exact: atoms must hold at the same step
                    if (F & fail).sum() >= min_fail:
                        out[c] = F
                        nxt.append(c)
            frontier = nxt
        return out

    def ub(self, F):
        return float((1.0 - self.Y)[F].sum()) / self.N


def stats(a):
    L = Lattice(a.db, a.half, a.max_atoms, a.metric)
    preds = L.enumerate()
    sets = {}
    for c, F in preds.items():
        sets.setdefault(F.tobytes(), []).append(c)
    n_eff = len(a.effects)
    print(f"{a.db} half {a.half}: N = {L.N}, base {'score' if a.metric == 'score' else 'success'} "
          f"{L.Y.mean():.3f}, atoms {len(L.atoms)}, predicates (<= {a.max_atoms} atoms, >= 1 failed affected) "
          f"{len(preds)}, distinct affected sets {len(sets)}, candidates x{n_eff} effects = {len(preds) * n_eff}")
    ubs = sorted((L.ub(F) for F in preds.values()), reverse=True)
    print("  UB distribution: max %.3f, median %.3f" % (ubs[0], ubs[len(ubs) // 2]))
    for lb in a.lbs:
        alive = [c for c, F in preds.items() if L.ub(F) >= lb]
        alive_sets = {preds[c].tobytes() for c in alive}
        print(f"  incumbent LB {lb:.3f}: predicates surviving {len(alive)} ({100 * len(alive) / len(preds):.1f}%), "
              f"distinct affected sets {len(alive_sets)}, candidates {len(alive_sets) * n_eff}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stats"])
    ap.add_argument("--db", required=True)
    ap.add_argument("--half", default="B")
    ap.add_argument("--max-atoms", type=int, default=3)
    ap.add_argument("--metric", default="won")
    ap.add_argument("--effects", nargs="*", default=["hint:2", "replan:2", "reject:3"])
    ap.add_argument("--lbs", type=float, nargs="*", default=[0.02, 0.04, 0.06, 0.08, 0.10])
    a = ap.parse_args()
    stats(a)


if __name__ == "__main__":
    main()
