"""LLM-proposed candidate changes, from failed games of half A only.

The proposer model reads one failed trajectory at a time and writes a general
rule in the change language: a SQL predicate over step attributes saying when
the agent is about to make a mistake, and one sentence of feedback for it.
Proposals are then screened on half A (they parse, they match enough games, they
co-occur with failure) and deduplicated by the set of games they match. Half B
is never read here, so testing on half B stays clean.

  python gowhy/propose.py --db store/fresh.db --proposer qwen3-8b --per-trace 2 --keep 30 \
      --out gowhy/proposed_qwen3.txt
Writes the candidates file and merges the feedback messages into gowhy/messages.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "harness"))
import llm  # noqa: E402
from changes import _MSGS_PATH, Change  # noqa: E402
from engine import half_of  # noqa: E402

PROMPT = """You are debugging a text-game agent (ALFWorld household tasks). Below is one game it FAILED.

Task: {task}
Task type: {task_type}

Steps (step number: action -> what the game replied):
{steps}

Find the agent's most important MISTAKE in this game, and state it as a GENERAL rule that would catch the same
kind of mistake in other games too. Write the rule as a SQL boolean expression over these columns of the step
the agent is about to take:
  k (int, 0-based step), verb (first word of the action: 'go','take','put','move','open','close','examine',
  'use','heat','cool','clean','look','inventory'), valid (1 if the action is admissible), repeat (1 if same as
  the previous action), revisit (1 if 'go to X' and X was visited before), obj_match (1 if the action mentions
  the task's target object type), recep_match (1 if it mentions the task's target receptacle type),
  n_invalid (count of earlier invalid actions), task_type (one of 'pick_and_place','pick_two_obj_and_place',
  'look_at_obj_in_light','pick_heat_then_place_in_recep','pick_cool_then_place_in_recep',
  'pick_clean_then_place_in_recep'), action (text), obs (text the agent currently sees), task (task text).
Use LIKE for text, e.g. action LIKE '%fridge%'. Do NOT mention specific object numbers like 'apple 1'.
The rule must fire BEFORE the bad action is executed, so it can only use the columns above.

Then write ONE sentence of feedback that would help the agent avoid the mistake (general, not game-specific).

Answer with only a JSON object on the last line:
{{"mistake": "...", "predicate": "...", "feedback": "..."}}"""


PROMPT_WEBSHOP = """You are debugging a web-shopping agent (WebShop). Below is one episode it FAILED: it did not
buy a product that matches every requirement of the instruction.

Instruction: {task}

Steps (step number: action -> what the page showed next):
{steps}

Find the agent's most important MISTAKE in this episode, and state it as a GENERAL rule that would catch the same
kind of mistake in other episodes too. Write the rule as a SQL boolean expression over these columns of the step
the agent is about to take:
  k (int, 0-based step), verb ('search' or 'click'), valid (1 if the action is available on the page),
  repeat (1 if same as the previous action), n_invalid (count of earlier invalid actions),
  action (text, e.g. 'search[red shoes]', 'click[buy now]', 'click[b07xyz]', 'click[size 9]'),
  obs (text of the page the agent currently sees), task (instruction text).
Use LIKE for text, e.g. action LIKE 'click[buy now]' or obs LIKE '%Total results%'. Do NOT mention specific product
ids, brands or option values.
The rule must fire BEFORE the bad action is executed, so it can only use the columns above.

Then write ONE sentence of feedback that would help the agent avoid the mistake (general, not episode-specific).

Answer with only a JSON object on the last line:
{{"mistake": "...", "predicate": "...", "feedback": "..."}}"""


def transcript(con, tid, max_obs=110):
    rows = con.execute("SELECT k, action, obs_after FROM events WHERE tid=? ORDER BY k", (tid,)).fetchall()
    return "\n".join(f"{k}: {a} -> {o.replace(chr(10), ' ')[:max_obs]}" for k, a, o in rows)


def parse(reply: str):
    m = re.findall(r"\{[^{}]*\"predicate\"[^{}]*\}", reply, re.S)
    if not m:
        return None
    try:
        d = json.loads(m[-1])
        return d if d.get("predicate") and d.get("feedback") else None
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--proposer", default="qwen3-8b")
    ap.add_argument("--per-trace", type=int, default=2)
    ap.add_argument("--keep", type=int, default=30)
    ap.add_argument("--min-support", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    con = sqlite3.connect(a.db)
    tr = {t: (w, tt, task) for t, w, tt, task in con.execute("SELECT tid, won, task_type, task FROM traces")}
    A = {t for t in tr if half_of(t) == "A"}
    failed = sorted(t for t in A if not tr[t][0])
    base_fail = len(failed) / len(A)
    pol = llm.Policy(a.proposer, temperature=0.7)
    jobs = [(t, i) for t in failed for i in range(a.per_trace)]

    def ask(job):
        t, i = job
        tmpl = PROMPT_WEBSHOP if tr[t][1] == "webshop" else PROMPT
        q = tmpl.format(task=tr[t][2], task_type=tr[t][1], steps=transcript(con_local(), t, 160 if tr[t][1] == "webshop" else 110))
        return t, parse(pol(q, seed=1000 + i))

    _cons = {}

    def con_local():
        import threading
        k = threading.get_ident()
        if k not in _cons:
            _cons[k] = sqlite3.connect(a.db)
        return _cons[k]

    with ThreadPoolExecutor(32) as ex:
        props = [p for p in ex.map(ask, jobs)]
    ok = [(t, d) for t, d in props if d]
    print(f"{len(jobs)} proposals asked, {len(ok)} parsed", file=sys.stderr)

    scored = {}
    bad = 0
    for t, d in ok:
        pred = " ".join(d["predicate"].split())
        try:
            Change.parse(f"UPDATE events SET action = reject:1 WHERE {pred} SCOPE ALL")
            hit = {x for (x,) in con.execute(
                f"SELECT DISTINCT e.tid FROM events e JOIN traces t USING(tid) WHERE {pred}") if x in A}
        except Exception:
            bad += 1
            continue
        n = len(hit)
        if n < a.min_support or n > 0.7 * len(A) or t not in hit:
            continue  # too rare, too broad, or does not even catch the game it came from
        f = sum(1 - tr[x][0] for x in hit) / n
        z = (f - base_fail) / math.sqrt(base_fail * (1 - base_fail) / n)
        key = frozenset(hit)
        if key not in scored or z > scored[key][0]:
            scored[key] = (z, pred, d["feedback"].strip(), n, f, d.get("mistake", ""))
    print(f"{bad} predicates failed to run; {len(scored)} distinct screened rules", file=sys.stderr)

    # keep the top rules by failure association, dropping near-duplicates (Jaccard > 0.8)
    kept = []
    for z, pred, fb, n, f, mis in sorted(scored.values(), key=lambda x: -x[0]):
        hit = {x for (x,) in con.execute(
            f"SELECT DISTINCT e.tid FROM events e JOIN traces t USING(tid) WHERE {pred}") if x in A}
        if any(len(hit & h) / len(hit | h) > 0.8 for h, *_ in kept):
            continue
        kept.append((hit, z, pred, fb, n, f, mis))
        if len(kept) == a.keep:
            break

    msgs = json.load(open(_MSGS_PATH)) if os.path.exists(_MSGS_PATH) else {}
    with open(a.out, "w") as fo:
        fo.write(f"# proposed by {a.proposer} from {len(failed)} failed half-A games of {a.db}\n")
        for hit, z, pred, fb, n, f, mis in kept:
            mid = hashlib.sha1(fb.encode()).hexdigest()[:8]
            msgs[mid] = fb
            fo.write(f"#   z {z:5.2f} games {n:3d} fail {f:.2f} | {mis[:100]} | feedback: {fb}\n")
            fo.write(f"UPDATE events SET action = hint:2:{mid} WHERE {pred} SCOPE ALL\n")
    json.dump(msgs, open(_MSGS_PATH, "w"), indent=1)
    print(f"wrote {len(kept)} candidates to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
