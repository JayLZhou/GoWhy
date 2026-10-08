"""AgentDebug ALFWorld harness, reproduced verbatim.

Source of truth: ulab-uiuc/AgentDebug
  agentdebug/environments/prompts/alfworld.py      (templates)
  agentdebug/environments/alfworld/projection.py   (action parsing)
  agentdebug/environments/env_manager.py           (prompt assembly, memory)
  agentdebug/memory/memory.py                      (history formatting)
  agentdebug/rollout/rollout.py                    (loop, retries, fallback "None")

An Episode is one game played under this harness. It can be restored to any
step k of a logged trajectory by replaying the logged (projected) actions, which
is exact because the text environment is deterministic.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

ALFWORLD_DATA = os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
ROOT = os.environ.get("GOWHY_ROOT", "/data2/yujia/Yingli/gowhy")
AEB_DIR = os.path.join(ROOT, "data", "aeb")
MAX_STEPS = 30
TEMPERATURE = 0.7
HISTORY_LENGTH = 10 ** 6  # AgentErrorBench logs never truncate history

# --------------------------------------------------------------------------- #
# Templates: copied character for character, including the "observaitons" typo.
# --------------------------------------------------------------------------- #
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your task is: {task_description}
Your current observation is: {current_observation}
Your admissible actions of the current situation are: {admissible_actions}.

Please begin by analyzing the situation and planning your approach:

<plan>
Plan the next step:
- Given what I've learned, what should I do next?
- Please explain why this plan is helpful for the next action?
- What do I expect this action to achieve?
</plan>

<action>
Finally, choose ONE admissible action for the current step and choose it within {admissible_actions}.
</action>
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observaitons and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: {admissible_actions}.

Now it's your turn to take an action.

You should first recall relevant past experiences and reason from our conversation history, then MUST summarize within <memory> </memory> tags like this:

<memory>
Look at the past observations and actions from our conversation history.
- Please retrieve the most relavent memory for this step including the relevant observation and action in a RAG style along with the step number.
- These memory should be helpful milestones to solve this task.
</memory>

After that, you should reflect on the last action and its outcome, then MUST summarize within <reflection> </reflection> tags like this:

<reflection>
Reflect on the last action and its outcome
- Did I complete the task goal?
- Was last action successful or did it encounter issues?
- Am I making progress toward the task goal?
- If the action did not go as expected and did not result in progress, provide constructive feedback to guide the next planning step.
</reflection>

After that, you should plan the next step based on memory and reflection, then MUST summarize within <plan> </plan> tags like this:

<plan>
Plan the next step based on memory and reflection
- Given what I've learned, what should I do next?
- Please explain why this plan is helpful for the next action?
- What do I expect this action to achieve?
</plan>

<action>
Finally, choose ONE admissible action for the current step and choose it within {admissible_actions}.
</action>
"""

TASK_TYPES = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]


def task_type_of(gamefile: str) -> str:
    if gamefile.startswith("webshop:"):
        return "webshop"
    for t in TASK_TYPES:
        if t in gamefile:
            return t
    return "other"


# --------------------------------------------------------------------------- #
# Action parsing (alfworld_projection), one action at a time.
# --------------------------------------------------------------------------- #
def project(raw: str) -> tuple[str, int]:
    """Return (action sent to env, format_valid) exactly as AgentDebug does."""
    low = raw.lower()
    s, e = low.find("<action>"), low.find("</action>")
    if s == -1 or e == -1:
        return low[-30:], 0
    action = low[s + len("<action>"):e].strip().lower()
    valid = 1
    if raw.find("<plan>") == -1 or raw.find("</plan>") == -1:
        valid = 0
    if re.search(r"[一-鿿]", raw):
        valid = 0
    return action, valid


def format_admissible(cmds: list[str]) -> str:
    return "\n ".join(f"'{s}'" for s in cmds if s != "help")


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def resolve_gamefile(path: str) -> str:
    """Map a logged gamefile path (e.g. /root/.cache/alfworld/...) to local data."""
    if os.path.exists(path):
        return path
    i = path.find("json_2.1.1/")
    if i == -1:
        raise FileNotFoundError(path)
    local = os.path.join(ALFWORLD_DATA, path[i:])
    if not os.path.exists(local):
        raise FileNotFoundError(local)
    return local


def make_env(gamefile: str):
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    req = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    env_id = textworld.gym.register_game(
        gamefile, req, max_episode_steps=50,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
    )
    return textworld.gym.make(env_id)


@dataclass
class Step:
    k: int
    prompt: str
    raw: str
    action: str          # projected action actually sent to the env
    fmt_valid: int       # AgentDebug's is_action_valid (tags present, no Chinese)
    admissible: int      # 1 if action is in the admissible list
    obs_after: str
    won: bool
    done: bool


@dataclass
class Episode:
    gamefile: str
    env: object = None
    task: str = ""
    obs: str = ""              # current raw observation
    admissible: list = field(default_factory=list)
    memory: list = field(default_factory=list)   # [{'text_obs', 'action'}]
    won: bool = False
    done: bool = False

    @classmethod
    def start(cls, gamefile: str) -> "Episode":
        ep = cls(gamefile=resolve_gamefile(gamefile))
        ep.env = make_env(ep.gamefile)
        obs, info = ep.env.reset()
        ep.obs = obs
        ep.admissible = list(info["admissible_commands"])
        i = obs.find("Your task is to: ")
        if i == -1:
            raise ValueError("Task description not found in text observation.")
        ep.task = obs[i + len("Your task is to: "):].strip()
        return ep

    @property
    def k(self) -> int:
        return len(self.memory)

    def prompt(self) -> str:
        adm = format_admissible(self.admissible)
        if self.k == 0:
            return ALFWORLD_TEMPLATE_NO_HIS.format(
                task_description=self.task, current_observation=self.obs,
                admissible_actions=adm)
        recent = self.memory[-HISTORY_LENGTH:]
        start = len(self.memory) - len(recent)
        lines = [f"[Observation {start + j + 1}: '{r['text_obs']}', Action {start + j + 1}: '{r['action']}']"
                 for j, r in enumerate(recent)]
        return ALFWORLD_TEMPLATE.format(
            task_description=self.task, step_count=len(self.memory),
            history_length=len(recent), action_history="\n".join(lines),
            current_step=len(self.memory) + 1, current_observation=self.obs,
            admissible_actions=adm)

    def act(self, raw: str, prompt: Optional[str] = None) -> Step:
        """Apply one model reply. Returns the step record."""
        assert not self.done
        if prompt is None:
            prompt = self.prompt()
        action, fmt_valid = project(raw)
        adm_ok = int(action in self.admissible)
        obs, _score, done, info = self.env.step(action)
        self.memory.append({"text_obs": self.obs, "action": action})
        self.obs = obs
        self.admissible = list(info["admissible_commands"])
        self.won = bool(info.get("won", False))
        self.done = bool(done) or self.won
        if self.k >= MAX_STEPS:
            self.done = True
        return Step(self.k - 1, prompt, raw, action, fmt_valid, adm_ok, obs, self.won, self.done)

    def act_projected(self, action: str) -> None:
        """Replay an already-projected action (used to restore a prefix)."""
        obs, _score, done, info = self.env.step(action)
        self.memory.append({"text_obs": self.obs, "action": action})
        self.obs = obs
        self.admissible = list(info["admissible_commands"])
        self.won = bool(info.get("won", False))
        self.done = bool(done) or self.won or self.k >= MAX_STEPS

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    def project(self, raw: str):
        return project(raw)

    def is_valid(self, action: str) -> int:
        return int(action in self.admissible)

    @property
    def score(self) -> float:
        return float(self.won)


def start_episode(gamefile: str):
    """Episode for any supported environment, chosen by the game id."""
    if gamefile.startswith("webshop:"):
        import webshop
        return webshop.Episode.start(gamefile)
    return Episode.start(gamefile)


def restore(gamefile: str, actions: list[str]):
    """New episode advanced by the given projected actions (fork point = len(actions))."""
    ep = start_episode(gamefile)
    for a in actions:
        if ep.done:
            raise RuntimeError("prefix ends the episode before the fork point")
        ep.act_projected(a)
    return ep


def run_policy(ep: Episode, policy: Callable[[str], str], max_steps: int = MAX_STEPS,
               on_step: Optional[Callable[[Step], None]] = None) -> list[Step]:
    """Continue ep under the policy until done. policy(prompt) -> raw reply."""
    steps = []
    while not ep.done and ep.k < max_steps:
        p = ep.prompt()
        st = ep.act(policy(p), p)
        steps.append(st)
        if on_step:
            on_step(st)
    return steps


# --------------------------------------------------------------------------- #
# Trajectory loading
# --------------------------------------------------------------------------- #
@dataclass
class Trace:
    tid: str
    model: str
    gamefile: str
    task_type: str
    prompts: list
    raws: list
    won: bool
    meta: dict = field(default_factory=dict)

    @property
    def actions(self) -> list[str]:
        if self.gamefile.startswith("webshop:"):
            import webshop
            return [webshop.project(r)[0] for r in self.raws]
        return [project(r)[0] for r in self.raws]


def load_aeb(models: Optional[list[str]] = None) -> list[Trace]:
    """ALFWorld trajectories from AgentErrorBench (HF davide221/agenterrorbench)."""
    import pandas as pd

    files = sorted(glob.glob(os.path.join(AEB_DIR, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet in {AEB_DIR}")
    df = pd.concat([pd.read_parquet(f) for f in files])
    df = df[df.task_type == "alfworld"]
    out = []
    for _, r in df.iterrows():
        t = json.loads(r.full_trajectory)
        md = t["metadata"]
        if models and r.llm_model not in models:
            continue
        msgs = t["messages"]
        prompts = [m["content"] for m in msgs if m["role"] == "user"]
        raws = [m["content"] for m in msgs if m["role"] == "assistant"]
        out.append(Trace(r.trajectory_id, r.llm_model, md["gamefile"], task_type_of(md["gamefile"]),
                         prompts, raws, bool(md.get("won", False)),
                         {"critical_failure_step": r.critical_failure_step,
                          "critical_failure_module": r.critical_failure_module}))
    return sorted(out, key=lambda x: x.tid)


def load_jsonl(path: str) -> list[Trace]:
    """Fresh trajectories written by gen_fresh.py."""
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out.append(Trace(d["tid"], d["model"], d["gamefile"], task_type_of(d["gamefile"]),
                             [s.get("prompt") for s in d["steps"]], [s["raw"] for s in d["steps"]],
                             bool(d["won"]), {"steps": d["steps"], "usage": d.get("usage")}))
    return out


_TT = {1: "pick_and_place_simple", 2: "look_at_obj_in_light", 3: "pick_clean_then_place_in_recep",
       4: "pick_heat_then_place_in_recep", 5: "pick_cool_then_place_in_recep", 6: "pick_two_obj_and_place"}


def game_files(split: str = "train") -> list[str]:
    """Same filter as alfworld AlfredTWEnv.collect_game_files (task types 1-6), sorted."""
    base = os.path.join(ALFWORLD_DATA, "json_2.1.1", split)
    out = []
    for root, _dirs, files in os.walk(base):
        if "traj_data.json" not in files:
            continue
        if "movable" in root or "Sliced" in root:
            continue
        with open(os.path.join(root, "traj_data.json")) as f:
            if json.load(f)["task_type"] not in _TT.values():
                continue
        gp = os.path.join(root, "game.tw-pddl")
        if not os.path.exists(gp):
            continue
        with open(gp) as f:
            if not json.load(f).get("solvable", False):
                continue
        out.append(gp)
    return sorted(out)
