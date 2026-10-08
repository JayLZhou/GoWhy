"""Replay engine: answer a change by forking logged trajectories.

For a change C and a logged trace, k0 is the first logged step whose reply
matches C's predicate. Before k0 the changed policy behaves exactly like the
logged one, so the prefix is reused as is. At k0 the logged reply is the first
sample the changed policy would also have drawn; C's effect is applied to it,
and the game continues under the changed policy. Traces with no match are
unaffected: under C they would have unfolded identically, so their outcome is
unchanged and no replay is needed.

k0 = -1 means a from-scratch deployment of C on the same game (the ground
truth used to check what-if answers).

Policies are routed to vLLM servers by served model name (harness/llm.py,
SERVERS or GOWHY_SERVERS).
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "harness"))
import aeb  # noqa: E402
import llm  # noqa: E402
import store  # noqa: E402
from changes import Change, set_action_reply  # noqa: E402

SEEDED_REP0 = 100  # reps >= 100 are seeded: common random numbers across changes


def seed_of(tid: str, rep: int, k: int, j: int) -> int:
    import hashlib
    return int(hashlib.sha256(f"{tid}|{rep}|{k}|{j}".encode()).hexdigest(), 16) % (2 ** 31)

_policies: dict = {}


def policy(name: str) -> llm.Policy:
    if name not in _policies:
        _policies[name] = llm.Policy(name)
    return _policies[name]


def _attrs(ep, raw, obj, task_type, recep=""):
    action, fv = ep.project(raw)
    a = store.event_attrs(ep, action, fv, obj, recep)
    a["task_type"] = task_type
    return a


def _usage_sum():
    tot = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    for p in _policies.values():
        u = p.usage()
        for k in tot:
            tot[k] += u[k]
    return tot


def replay_one(change_text: str, base: str, trace: dict, events: list, k0: int, rep: int) -> dict:
    """One continuation of `trace` from k0 under the change. events: logged [(action, raw)]."""
    ch = Change.parse(change_text)
    obj, task_type, recep = trace["obj"], trace["task_type"], trace["recep"]
    u0 = _usage_sum()
    t0 = time.time()
    ep = aeb.start_episode(trace["gamefile"])
    try:
        start = max(k0, 0)
        for action, _raw in events[:start]:
            a = store.event_attrs(ep, action, 1, obj, recep)
            ep.act_projected(action)
            store.note_step(ep, a["valid"])
        pol = policy(base)
        active = True
        first = k0 >= 0
        seeded = rep >= SEEDED_REP0
        actions = []
        while not ep.done:
            p = ep.prompt()
            j = 0  # sample index within this step, for the seed

            def draw(prompt):
                nonlocal j
                sd = seed_of(trace["tid"], rep, ep.k, j) if seeded else None
                j += 1
                return pol(prompt, sd) if sd is not None else pol(prompt)

            if first:
                raw = events[k0][1]
                first = False
                j = 1
            else:
                raw = draw(p)
            a = _attrs(ep, raw, obj, task_type, recep)
            if active and ch.matches(a):
                if ch.effect == "reject":
                    for _ in range(ch.tries):
                        raw = draw(p)
                        a = _attrs(ep, raw, obj, task_type, recep)
                        if not ch.matches(a):
                            break
                elif ch.effect in ("hint", "replan"):
                    for _ in range(ch.tries):
                        raw = draw(p + "\n\n" + ch.feedback(ep.project(raw)[0], ep))
                        a = _attrs(ep, raw, obj, task_type, recep)
                        if not ch.matches(a):
                            break
                elif ch.effect == "swap":
                    pol = policy(ch.arg)
                    raw = draw(p)
                elif ch.effect == "set":
                    raw = set_action_reply(ch.arg)
                elif ch.effect == "reparse":
                    from changes import lenient_action
                    fixed = lenient_action(raw, ep.admissible)
                    if fixed is not None:
                        raw = set_action_reply(fixed)
                elif ch.effect == "unquote":
                    act, _ = ep.project(raw)
                    raw = set_action_reply(act.replace("'", "").replace('"', ""))
                if ch.scope == "FIRST":
                    active = False
                a = _attrs(ep, raw, obj, task_type, recep)
            st = ep.act(raw, p)
            store.note_step(ep, a["valid"])
            actions.append(st.action)
    finally:
        ep.close()
    u1 = _usage_sum()
    return {"change": ch.text, "tid": trace["tid"], "k0": k0, "rep": rep, "won": int(ep.won),
            "score": float(ep.score),
            "n_steps": ep.k, "calls": u1["calls"] - u0["calls"],
            "prompt_tokens": u1["prompt_tokens"] - u0["prompt_tokens"],
            "completion_tokens": u1["completion_tokens"] - u0["completion_tokens"],
            "actions": json.dumps(actions), "ts": time.time()}


def half_of(tid: str) -> str:
    """Fixed split of stored games: mine hypotheses on A, test them on B.
    Salted so it is independent of every other hash of tid. The Qwen3 store chose
    one of two duplicate runs per game by sha256(tid) % 2; an unsalted split equal
    to that choice put all first-finished (win-biased) runs in one half."""
    import hashlib
    return "A" if int(hashlib.sha256(("split:" + tid).encode()).hexdigest(), 16) % 2 == 0 else "B"


class Engine:
    def __init__(self, db: str, base: str, workers: int = 64, limit: int = 0, half: str = ""):
        self.db = db
        self.limit = limit
        self.half = half
        self.base = base
        self.workers = workers
        self.con = store.connect(db)
        self.con.row_factory = sqlite3.Row

    def _insert_run(self, r: dict):
        vals = tuple(r[k] for k in ["change", "tid", "k0", "rep", "won", "n_steps", "calls",
                                    "prompt_tokens", "completion_tokens", "actions", "ts", "score"])
        for attempt in range(10):
            try:
                self.con.execute(
                    "INSERT INTO runs(change,tid,k0,rep,won,n_steps,calls,prompt_tokens,"
                    "completion_tokens,actions,ts,score) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", vals)
                self.con.commit()
                return
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) or attempt == 9:
                    raise
                time.sleep(2 + 3 * attempt)

    def traces(self) -> dict:
        rows = [dict(r) for r in self.con.execute("SELECT * FROM traces ORDER BY tid")]
        if self.half:
            rows = [r for r in rows if half_of(r["tid"]) == self.half]
        if self.limit:
            rows = rows[:self.limit]
        return {r["tid"]: r for r in rows}

    def events(self, tid: str) -> list:
        return [(r["action"], r["raw"]) for r in
                self.con.execute("SELECT action, raw FROM events WHERE tid=? ORDER BY k", (tid,))]

    def fork_points(self, ch: Change) -> dict:
        return {tid: k0 for tid, k0 in self.con.execute(ch.first_match_sql())}

    def cached(self, ch: Change, tid: str, k0: int) -> list:
        return [dict(r) for r in self.con.execute(
            "SELECT * FROM runs WHERE change=? AND tid=? AND k0=? ORDER BY rep", (ch.text, tid, k0))]

    def run(self, ch: Change, n: int, deploy: bool = False, reps_offset: int = 0,
            tids=None, verbose: bool = True) -> list:
        """Replays (or from-scratch deployments) for every affected trace, n reps each.
        Cached rows are reused; returns all rows for these (tid, k0)."""
        traces = self.traces()
        fps = {t: -1 for t in traces} if deploy else self.fork_points(ch)
        fps = {t: k for t, k in fps.items() if t in traces}
        if tids is not None:
            fps = {t: k for t, k in fps.items() if t in tids}
        jobs = []
        for tid, k0 in fps.items():
            have = {r["rep"] for r in self.cached(ch, tid, k0)}
            for rep in range(reps_offset, reps_offset + n):
                if rep not in have:
                    jobs.append((tid, k0, rep))
        if verbose:
            print(f"[{'deploy' if deploy else 'what-if'}] {ch.text}\n  {len(fps)} traces to run, "
                  f"{len(jobs)} new runs", flush=True)
        if jobs:
            ev = {tid: self.events(tid) for tid in {j[0] for j in jobs}}
            t0 = time.time()
            import multiprocessing as mp
            ctx = mp.get_context("spawn") if any(traces[j[0]]["gamefile"].startswith("webshop:") for j in jobs) else None
            with ProcessPoolExecutor(self.workers, mp_context=ctx) as ex:
                futs = [ex.submit(replay_one, ch.text, self.base, traces[tid], ev[tid], k0, rep)
                        for tid, k0, rep in jobs]
                for i, fu in enumerate(as_completed(futs), 1):
                    try:
                        r = fu.result()
                    except Exception as e:
                        print("  run failed:", repr(e)[:300], flush=True)
                        continue
                    self._insert_run(r)  # committed at once, so a crash loses at most one row
                    if i % 20 == 0 or i == len(futs):
                        if verbose:
                            print(f"  {i}/{len(futs)} runs, {time.time() - t0:.0f}s", flush=True)
            self.con.commit()
        rows = []
        for tid, k0 in fps.items():
            rows += [r for r in self.cached(ch, tid, k0) if reps_offset <= r["rep"] < reps_offset + n]
        return rows
