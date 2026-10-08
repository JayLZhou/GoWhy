"""A change to the agent, written as an update to its future events.

  UPDATE events SET action = <effect> WHERE <predicate> SCOPE FIRST|ALL

predicate  SQL boolean expression over event attributes (see store.ATTRS), plus
           task_type. Evaluated by SQLite itself, on logged events and on new
           replies during replay.
SCOPE      FIRST: the effect applies at the first matching step only.
           ALL:   it applies at every matching step from then on (a policy change).

Effects
  reject:N        a verifier: when the reply matches, resample up to N times
                  until it no longer matches (keep the last sample otherwise)
  swap:<served>   from the matching step on, the policy is another model
  set:<action>    replace the action with a fixed one, e.g. set:look
  unquote         strip ' and " from the action (a harness fix, not a model change)
  hint:N          feedback and retry: append a checker's feedback naming the flagged
                  action to the prompt (as AgentDebug injects debugger feedback) and
                  resample, up to N times, until the reply no longer matches
  hint:N:ID       the same with a custom feedback message, looked up by ID in the
                  messages file (GOWHY_MSGS, default gowhy/messages.json)
  reparse         a harness fix: when the strict parser yields an inadmissible action, read
                  the text after <action> leniently (no closing tag needed, quotes and
                  "I choose to" phrasing removed) and map it to the closest admissible action
  replan:N[:ID]   stronger feedback and retry: the checker also restates the task, the
                  actions taken so far and the places already visited, and asks the agent
                  to name the unfinished part of the task before choosing again
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from dataclasses import dataclass

_RX = re.compile(
    r"^\s*UPDATE\s+events\s+SET\s+action\s*=\s*(?P<effect>\S+)\s+WHERE\s+(?P<pred>.+?)"
    r"\s+SCOPE\s+(?P<scope>FIRST|ALL)\s*$", re.I | re.S)

_COLS = ["k", "verb", "valid", "fmt_valid", "quoted", "revisit", "repeat", "obj_match", "recep_match",
         "n_invalid", "task_type", "action", "obs", "task"]
_MSGS_PATH = os.environ.get("GOWHY_MSGS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "messages.json"))
_msgs = None


def message(mid: str) -> str:
    global _msgs
    if _msgs is None:
        _msgs = json.load(open(_MSGS_PATH)) if os.path.exists(_MSGS_PATH) else {}
    return _msgs[mid]
_local = threading.local()


def _con():
    if not hasattr(_local, "con"):
        c = sqlite3.connect(":memory:")
        c.execute(f"CREATE TABLE ev({', '.join(_COLS)})")
        _local.con = c
    return _local.con


@dataclass(frozen=True)
class Change:
    text: str
    effect: str
    arg: str
    pred: str
    scope: str

    @classmethod
    def parse(cls, text: str) -> "Change":
        m = _RX.match(text)
        if not m:
            raise ValueError(f"cannot parse change: {text!r}")
        eff = m.group("effect")
        name, _, arg = eff.partition(":")
        if name not in ("reject", "swap", "set", "unquote", "hint", "replan", "reparse"):
            raise ValueError(f"unknown effect {name!r}")
        ch = cls(" ".join(text.split()), name, arg, m.group("pred").strip(), m.group("scope").upper())
        ch.matches({c: (0 if c not in ("action", "obs", "task", "verb", "task_type") else "") for c in _COLS})
        if name in ("hint", "replan") and ":" in arg:
            message(arg.split(":", 1)[1])  # must exist
        return ch

    @property
    def tries(self) -> int:
        return int(self.arg.split(":", 1)[0] or 1)

    def feedback(self, action: str, ep=None) -> str:
        if self.effect == "replan":
            why = message(self.arg.split(":", 1)[1]) if ":" in self.arg else f"({hint_reason(self.pred)})"
            return replan_text(ep, action, why)
        if self.effect == "hint" and ":" in self.arg:
            return HINT_CUSTOM.format(action=action, msg=message(self.arg.split(":", 1)[1]))
        return HINT.format(action=action, reason=hint_reason(self.pred))

    def matches(self, attrs: dict) -> bool:
        c = _con()
        c.execute("DELETE FROM ev")
        c.execute(f"INSERT INTO ev VALUES({','.join('?' * len(_COLS))})", [attrs.get(k) for k in _COLS])
        return c.execute(f"SELECT 1 FROM ev WHERE {self.pred}").fetchone() is not None

    def first_match_sql(self) -> str:
        """SQL giving (tid, k0) = first matching logged event per trace."""
        return (f"SELECT e.tid, MIN(e.k) FROM events e JOIN traces t USING(tid) "
                f"WHERE {self._qualified()} GROUP BY e.tid")

    def _qualified(self) -> str:
        return self.pred


HINT = ("Feedback from a checker: your chosen action '{action}' looks unhelpful here "
        "({reason}). Re-read the task and your memory, then choose a different admissible action.")


HINT_CUSTOM = ("Feedback from a checker: your chosen action '{action}' looks unhelpful here. {msg} "
               "Choose a different admissible action.")


def replan_text(ep, action: str, why: str) -> str:
    """Checker message for replan: task, history and visited places, then a request to replan."""
    acts = [m["action"] for m in ep.memory] if ep is not None else []
    shown = acts[-15:]
    lines = [f"A checker stopped your chosen action '{action}'. {why}",
             f"Re-read the task: {ep.task if ep is not None else ''}",
             "Actions you have taken so far: " + ("; ".join(shown) if shown else "none") +
             (f" (and {len(acts) - len(shown)} earlier)" if len(acts) > len(shown) else "") + "."]
    visited = sorted({a[6:] for a in acts if a.startswith("go to ")})
    if visited:
        lines.append("Places you have already visited: " + ", ".join(visited) + ".")
    lines.append("First state in one sentence which part of the task is still unfinished, then choose ONE "
                 "different admissible action that makes progress on it.")
    return "\n".join(lines)


def hint_reason(pred: str) -> str:
    """Plain-words reason for the feedback, from the predicate's atoms."""
    r = []
    if re.search(r"valid\s*=\s*0", pred) and "fmt_valid" not in pred:
        r.append("it is not one of the admissible actions")
    if re.search(r"repeat\s*=\s*1", pred):
        r.append("it repeats your previous action")
    if re.search(r"revisit\s*=\s*1", pred):
        r.append("you have already been to that place")
    if re.search(r"obj_match\s*=\s*0", pred):
        r.append("it does not involve the object the task asks for")
    if re.search(r"recep_match\s*=\s*0", pred):
        r.append("it does not use the receptacle the task asks for")
    return "; ".join(r) or "it does not seem to make progress"


def lenient_action(raw: str, admissible: list) -> str | None:
    """Closest admissible action to what the reply meant, or None."""
    import difflib
    low = raw.lower()
    i = low.rfind("<action>")
    text = low[i + len("<action>"):] if i != -1 else low[-200:]
    j = text.find("</action>")
    if j != -1:
        text = text[:j]
    text = re.sub(r"\b(i choose to|i will|i'll|i choose|choose|action:|next action is)\b", " ", text)
    text = " ".join(text.replace("'", " ").replace('"', " ").replace("`", " ").split()).strip(" .:")
    adm = [a for a in admissible if a != "help"]
    if text in adm:
        return text
    inside = [a for a in adm if a in text]
    if inside:
        return max(inside, key=len)
    best = difflib.get_close_matches(text, adm, n=1, cutoff=0.6)
    return best[0] if best else None


def set_action_reply(action: str) -> str:
    """A reply whose projected action is `action` (format tags kept valid)."""
    return f"<plan>forced by change</plan>\n<action>{action}</action>"
