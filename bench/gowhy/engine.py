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


# step-log fields kept per replayed step, for sharing replays across changes (Lattice-Share)
LOG_ATTRS = ("k", "verb", "valid", "fmt_valid", "quoted", "revisit", "repeat", "obj_match", "recep_match",
             "n_invalid", "task_type", "action")


def pack_steps(steps: list) -> bytes:
    import zlib
    return zlib.compress(json.dumps(steps).encode())


def unpack_steps(blob) -> list:
    import zlib
    return json.loads(zlib.decompress(blob).decode()) if blob else []


def replay_one(change_text: str, base: str, trace: dict, events: list, k0: int, rep: int,
               donor: list | None = None, log_steps: bool = False) -> dict:
    """One continuation of `trace` from k0 under the change. events: logged [(action, raw)].

    Sharing (Lattice-Share). With log_steps, every replayed step records the base draw (the
    policy's reply before any effect), its attributes, whether the change fired, and the
    reply finally used. With donor = such a log from a replay of another change with the same
    effect, episode, fork step and seed family, this replay walks the donor: at each step it
    takes the donor's base draw, evaluates its own predicate on the stored attributes, and if
    its firing decision equals the donor's it reuses the donor's final reply (no model call).
    At the first step where the decisions differ it applies its own decision to the donor's
    base draw and continues with fresh draws. Both policies coincide on every step before the
    divergence, so the shared prefix is a valid sample path of this change (a coupling), and
    the result is an unbiased replay at a fraction of the cost.
    """
    if ";;" in change_text:
        return replay_patch(change_text, base, trace, events, k0, rep, log_steps)
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
        guided = ""                      # guide effect: suffix on every prompt once activated
        steps = []                       # step log (log_steps)
        follow = donor is not None and ch.effect != "guide"
        s_idx = 0
        reused = 0                       # steps taken entirely from the donor (no model call)
        while not ep.done:
            p = ep.prompt() + guided
            j = 0  # sample index within this step, for the seed

            def draw(prompt):
                nonlocal j
                sd = seed_of(trace["tid"], rep, ep.k, j) if seeded else None
                j += 1
                return pol(prompt, sd) if sd is not None else pol(prompt)

            d = donor[s_idx] if follow and s_idx < len(donor) else None
            if d is None:
                follow = False
            # base draw: the donor's (shared randomness), the logged reply at the fork, or fresh
            if d is not None:
                raw = d["base"]
                first = False
                j = 1
            elif first:
                raw = events[k0][1]
                first = False
                j = 1
            else:
                raw = draw(p)
            base_raw = raw
            a = _attrs(ep, raw, obj, task_type, recep)
            base_attrs = {k: a.get(k) for k in LOG_ATTRS}
            fire = bool(active and ch.matches(a))
            fresh = d is None                         # did this step need any new model call?
            if d is not None and fire != bool(d["fired"]):
                fresh = True                          # decisions differ on the shared base draw
            retries = []
            if fire:
                R = d["retries"] if (d is not None and d["fired"]) else []
                if ch.effect in ("reject", "hint", "replan"):
                    # retry until the reply no longer matches; walk the donor's retries while the
                    # prompt (feedback text) is the same, then draw fresh
                    for t in range(ch.tries):
                        fb = "" if ch.effect == "reject" else ch.feedback(ep.project(raw)[0], ep)
                        if not fresh and t < len(R) and R[t]["fb"] == fb:
                            raw = R[t]["raw"]
                        else:
                            fresh = True
                            j = t + 1
                            raw = draw(p + ("\n\n" + fb if fb else ""))
                        retries.append({"raw": raw, "fb": fb})
                        a = _attrs(ep, raw, obj, task_type, recep)
                        if not ch.matches(a):
                            break
                elif ch.effect == "guide":
                    if not guided:
                        from changes import guide_text
                        guided = guide_text(ch)
                        p = p + guided
                        raw = draw(p)
                        fresh = True
                elif ch.effect == "swap":
                    pol = policy(ch.arg)
                    if not fresh and R:
                        raw = R[0]["raw"]
                    else:
                        fresh = True
                        raw = draw(p)
                    retries.append({"raw": raw, "fb": "swap"})
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
            if d is not None:
                if not fresh and raw == d["final"]:
                    reused += 1
                else:
                    follow = False                    # diverged: the rest of the episode is fresh
            if log_steps:
                steps.append({"base": base_raw, "attrs": base_attrs, "fired": int(fire),
                              "retries": retries, "final": raw})
            st = ep.act(raw, p)
            store.note_step(ep, a["valid"])
            actions.append(st.action)
            s_idx += 1
    finally:
        ep.close()
    u1 = _usage_sum()
    return {"change": ch.text, "tid": trace["tid"], "k0": k0, "rep": rep, "won": int(ep.won),
            "score": float(ep.score),
            "n_steps": ep.k, "calls": u1["calls"] - u0["calls"],
            "prompt_tokens": u1["prompt_tokens"] - u0["prompt_tokens"],
            "completion_tokens": u1["completion_tokens"] - u0["completion_tokens"],
            "actions": json.dumps(actions), "ts": time.time(),
            "steps": pack_steps(steps) if log_steps else None, "reused": reused}


def replay_patch(change_text: str, base: str, trace: dict, events: list, k0: int, rep: int,
                 log_steps: bool = False) -> dict:
    """Replay under a patch C1 ;; ... ;; Cm (changes applied in order at every step).

    Used by Counterfactual Store Iteration: the store holds trajectories of the policy already
    patched with C1..C(m-1) (each event's raw is the reply after those changes), and k0 is the
    first step where Cm fires on it. At k0 only Cm is applied to the stored reply (the earlier
    changes already acted on it); from the next step on, every fresh base draw goes through the
    whole patch. With k0 = -1 (from scratch) the whole patch applies from step 0.
    """
    from changes import parse_patch, patch_text, lenient_action
    patch = parse_patch(change_text)
    text = patch_text(patch)
    obj, task_type, recep = trace["obj"], trace["task_type"], trace["recep"]
    u0 = _usage_sum()
    ep = aeb.start_episode(trace["gamefile"])
    try:
        start = max(k0, 0)
        for action, _raw in events[:start]:
            a = store.event_attrs(ep, action, 1, obj, recep)
            ep.act_projected(action)
            store.note_step(ep, a["valid"])
        pol = policy(base)
        active = [True] * len(patch)
        first = k0 >= 0
        seeded = rep >= SEEDED_REP0
        actions, steps = [], []
        while not ep.done:
            p = ep.prompt()
            j = 0

            def draw(prompt):
                nonlocal j
                sd = seed_of(trace["tid"], rep, ep.k, j) if seeded else None
                j += 1
                return pol(prompt, sd) if sd is not None else pol(prompt)

            if first:
                raw = events[k0][1]
                j = 1
                skip = len(patch) - 1
                first = False
            else:
                raw = draw(p)
                skip = 0
            base_raw = raw
            a = _attrs(ep, raw, obj, task_type, recep)
            base_attrs = {k: a.get(k) for k in LOG_ATTRS}
            fired = []
            for idx, c in enumerate(patch):
                if idx < skip or not active[idx] or not c.matches(a):
                    continue
                fired.append(idx)
                if c.effect == "reject":
                    for _ in range(c.tries):
                        raw = draw(p)
                        a = _attrs(ep, raw, obj, task_type, recep)
                        if not c.matches(a):
                            break
                elif c.effect in ("hint", "replan"):
                    for _ in range(c.tries):
                        raw = draw(p + "\n\n" + c.feedback(ep.project(raw)[0], ep))
                        a = _attrs(ep, raw, obj, task_type, recep)
                        if not c.matches(a):
                            break
                elif c.effect == "set":
                    raw = set_action_reply(c.arg)
                elif c.effect == "reparse":
                    fixed = lenient_action(raw, ep.admissible)
                    if fixed is not None:
                        raw = set_action_reply(fixed)
                elif c.effect == "unquote":
                    act, _ = ep.project(raw)
                    raw = set_action_reply(act.replace("'", "").replace('"', ""))
                else:
                    raise ValueError(f"effect {c.effect} not supported in patches")
                if c.scope == "FIRST":
                    active[idx] = False
                a = _attrs(ep, raw, obj, task_type, recep)
            if log_steps:
                steps.append({"base": base_raw, "attrs": base_attrs, "fired": fired, "final": raw})
            st = ep.act(raw, p)
            store.note_step(ep, a["valid"])
            actions.append(st.action)
    finally:
        ep.close()
    u1 = _usage_sum()
    return {"change": text, "tid": trace["tid"], "k0": k0, "rep": rep, "won": int(ep.won),
            "score": float(ep.score), "n_steps": ep.k, "calls": u1["calls"] - u0["calls"],
            "prompt_tokens": u1["prompt_tokens"] - u0["prompt_tokens"],
            "completion_tokens": u1["completion_tokens"] - u0["completion_tokens"],
            "actions": json.dumps(actions), "ts": time.time(),
            "steps": pack_steps(steps) if log_steps else None, "reused": 0}


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
        vals = tuple(r.get(k) for k in ["change", "tid", "k0", "rep", "won", "n_steps", "calls",
                                        "prompt_tokens", "completion_tokens", "actions", "ts", "score",
                                        "steps"])
        for attempt in range(10):
            try:
                self.con.execute(
                    "INSERT INTO runs(change,tid,k0,rep,won,n_steps,calls,prompt_tokens,"
                    "completion_tokens,actions,ts,score,steps) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", vals)
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
            ctx = mp.get_context("spawn") if any(traces[j[0]]["gamefile"].startswith(("webshop:", "sciworld:"))
                                                 for j in jobs) else None
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
