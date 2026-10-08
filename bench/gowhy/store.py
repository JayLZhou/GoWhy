"""Trajectory store: SQLite with traces / events / runs.

  python gowhy/store.py fresh_q25   # data/fresh/Qwen2.5-7B-fresh.jsonl -> store/fresh_q25.db
  python gowhy/store.py fresh       # data/fresh/Qwen3-8B-fresh.jsonl   -> store/fresh.db
  python gowhy/store.py aeb         # AgentErrorBench ALFWorld          -> store/gowhy.db

Event attributes are computed from the state *before* the step runs, by the
same function the replay engine uses on new replies (event_attrs). So a WHERE
predicate means the same thing on logged events and during replay.

  k            step index (0-based)
  verb         first word of the action (go, take, put, open, ...)
  valid        1 if the action is in the admissible list
  fmt_valid    AgentDebug's format check (<plan> and <action> tags, no Chinese)
  quoted       1 if the action contains ' or "
  revisit      1 if it is "go to X" and X was already a go-to target earlier
  repeat       1 if it equals the previous action
  obj_match    1 if it mentions the task's target object type
  recep_match  1 if it mentions the task's target receptacle type
  n_invalid    number of earlier non-admissible actions in this episode
  action       the action text itself (for LIKE patterns)
  obs          the observation the agent sees before this step
  task         the task description (traces.task in SQL)
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "harness"))
import aeb  # noqa: E402

SOURCES = {
    "fresh": ("data/fresh/Qwen3-8B-fresh.jsonl", "store/fresh.db"),
    "fresh_q25": ("data/fresh/Qwen2.5-7B-fresh.jsonl", "store/fresh_q25.db"),
    "aeb": (None, "store/gowhy.db"),
    "ws_q3": ("data/fresh/Qwen3-8B-ws.jsonl", "store/ws_q3.db"),
    "fresh_llama": ("data/fresh/Llama-3.1-8B-fresh.jsonl", "store/fresh_llama.db"),
}

ATTRS = ["k", "verb", "valid", "fmt_valid", "quoted", "revisit", "repeat", "obj_match", "recep_match", "n_invalid"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces(
  tid TEXT PRIMARY KEY, model TEXT, gamefile TEXT, task_type TEXT, obj TEXT, recep TEXT,
  won INTEGER, n_steps INTEGER, calls INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER);
CREATE TABLE IF NOT EXISTS events(
  tid TEXT, k INTEGER, action TEXT, raw TEXT, verb TEXT, valid INTEGER, fmt_valid INTEGER,
  quoted INTEGER, revisit INTEGER, repeat INTEGER, obj_match INTEGER, n_invalid INTEGER,
  obs_after TEXT, recep_match INTEGER, PRIMARY KEY(tid, k));
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, change TEXT, tid TEXT, k0 INTEGER, rep INTEGER,
  won INTEGER, n_steps INTEGER, calls INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER,
  actions TEXT, ts REAL);
CREATE INDEX IF NOT EXISTS runs_key ON runs(change, tid, rep);
"""


def game_parts(gamefile: str):
    """pick_and_place_simple-Newspaper-None-Sofa-224 -> ('newspaper', 'sofa'); ('', '') for WebShop."""
    if gamefile.startswith("webshop:"):
        return "", ""
    d = gamefile.split("/json_2.1.1/")[-1].split("/")[1]
    p = d.split("-")
    return p[1].lower(), p[3].lower()


def event_attrs(ep: aeb.Episode, action: str, fmt_valid: int, obj: str, recep: str = "") -> dict:
    """Attributes of taking `action` in the current state of ep (before stepping)."""
    prev = [m["action"] for m in ep.memory]
    goto = re.match(r"go to (.+)$", action)
    targets = [a[6:] for a in prev if a.startswith("go to ")]
    return {
        "k": ep.k,
        "verb": (re.match(r"[a-z]*", action).group(0) if "[" in action else action.split(" ", 1)[0]) if action else "",
        "valid": ep.is_valid(action),
        "fmt_valid": int(fmt_valid),
        "quoted": int("'" in action or '"' in action),
        "revisit": int(bool(goto) and goto.group(1) in targets),
        "repeat": int(bool(prev) and prev[-1] == action),
        "obj_match": int(bool(obj) and obj in action.replace(" ", "")),
        "recep_match": int(bool(recep) and recep in action.replace(" ", "")),
        "n_invalid": ep.meta_invalid if hasattr(ep, "meta_invalid") else 0,
        "action": action,
        "obs": ep.obs,
        "task": ep.task,
    }


def note_step(ep: aeb.Episode, valid: int):
    ep.meta_invalid = getattr(ep, "meta_invalid", 0) + (1 - valid)


def connect(path: str) -> sqlite3.Connection:
    # Several query processes may write runs at once: WAL mode, long busy timeout.
    con = sqlite3.connect(path, timeout=600)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=600000")
    con.executescript(SCHEMA)
    # columns added after the first stores were built
    for table, col, typ in [("events", "recep_match", "INTEGER"), ("events", "obs", "TEXT"),
                            ("traces", "task", "TEXT"), ("traces", "score", "REAL"), ("runs", "score", "REAL")]:
        if col not in {r[1] for r in con.execute(f"PRAGMA table_info({table})")}:
            try:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError as e:  # another process added it first
                if "duplicate column" not in str(e):
                    raise
    con.commit()
    return con


def trace_events(tr: aeb.Trace):
    """Replay a logged trace and return its event rows (runs in a worker process)."""
    obj, recep = game_parts(tr.gamefile)
    rows = []
    ep = aeb.start_episode(tr.gamefile)
    try:
        for raw in tr.raws:
            action, fv = ep.project(raw)
            a = event_attrs(ep, action, fv, obj, recep)
            ep.act_projected(action)
            note_step(ep, a["valid"])
            rows.append((tr.tid, a["k"], action, raw, a["verb"], a["valid"], a["fmt_valid"], a["quoted"],
                         a["revisit"], a["repeat"], a["obj_match"], a["n_invalid"], ep.obs, a["recep_match"],
                         a["obs"]))
        won = bool(ep.won)
        task = ep.task
        score = float(ep.score)
    finally:
        ep.close()
    return tr.tid, rows, won, task, score


def build(name: str, workers: int = 32):
    from concurrent.futures import ProcessPoolExecutor

    src, dbp = SOURCES[name]
    root = aeb.ROOT
    traces = aeb.load_aeb() if src is None else aeb.load_jsonl(os.path.join(root, src))
    con = connect(os.path.join(root, dbp))
    for tr in traces:
        obj, recep = game_parts(tr.gamefile)
        u = tr.meta.get("usage") or {}
        con.execute("INSERT OR REPLACE INTO traces(tid,model,gamefile,task_type,obj,recep,won,n_steps,calls,"
                    "prompt_tokens,completion_tokens) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (tr.tid, tr.model, tr.gamefile, tr.task_type, obj, recep, int(tr.won), len(tr.raws),
                     u.get("calls"), u.get("prompt_tokens"), u.get("completion_tokens")))
    logged_won = {tr.tid: tr.won for tr in traces}
    n_ev = 0
    import multiprocessing as mp
    ctx = mp.get_context("spawn") if any(t.gamefile.startswith("webshop:") for t in traces) else None
    with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        for tid, rows, won, task, score in ex.map(trace_events, traces, chunksize=4):
            con.executemany("INSERT OR REPLACE INTO events(tid,k,action,raw,verb,valid,fmt_valid,quoted,revisit,"
                            "repeat,obj_match,n_invalid,obs_after,recep_match,obs) "
                            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            con.execute("UPDATE traces SET task=?, score=? WHERE tid=?", (task, score, tid))
            n_ev += len(rows)
            if won != bool(logged_won[tid]):
                print(f"warning: {tid} logged won={logged_won[tid]} but replay won={won}")
    con.commit()
    print(f"{dbp}: {len(traces)} traces, {n_ev} events")
    for row in con.execute("SELECT task_type, COUNT(*), AVG(won) FROM traces GROUP BY task_type"):
        print("  ", row)
    print("  overall success:", con.execute("SELECT AVG(won) FROM traces").fetchone()[0])
    print("  invalid-action rate:", con.execute("SELECT 1 - AVG(valid) FROM events").fetchone()[0])


if __name__ == "__main__":
    build(sys.argv[1])
