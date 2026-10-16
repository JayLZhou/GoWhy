"""ScienceWorld harness in the AgentDebug style (third environment).

A game id is "sciworld:<task name>:<variation>". Each step is one stateless prompt
built from the task, the most recent HISTORY_LENGTH observations and actions, the
current observation, and the admissible actions; the reply's first <action> block is
the action (AgentDebug parsing). The simulator is deterministic given the task, the
variation and the action sequence, so a prefix is restored by replaying its actions.

Outcome: score = max(task score, 0) / 100 in [0, 1]; won = task score >= 100.
One ScienceWorldEnv (a JVM) per process, reused across episodes.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

JAVA_HOME = os.environ.get("SCIWORLD_JAVA_HOME", "/data2/yujia/Yingli/gowhy_setup/envs/webshop/lib/jvm")
MAX_STEPS = 30
HISTORY_LENGTH = 10
SIMPLIFICATION = "easy"

TEMPLATE_NO_HIS = """
You are an expert agent operating in the ScienceWorld environment.
Your task is: {task}
Your current observation is: {obs}
Your admissible actions of the current situation are: {actions}.

Please begin by analyzing the situation and planning your approach:

<plan>
Plan the next step:
- Given what I've learned, what should I do next?
- Please explain why this plan is helpful for the next action?
- What do I expect this action to achieve?
</plan>

<action>
Finally, choose ONE admissible action for the current step and choose it within {actions}.
</action>
"""

TEMPLATE = """
You are an expert agent operating in the ScienceWorld environment. Your task is to: {task}
Prior to this step, you have already taken {k} step(s). Below are the most recent {h} observations and the corresponding actions you took: {history}
You are now at step {step} and your current observation is: {obs}
Your admissible actions of the current situation are: {actions}.

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
Finally, choose ONE admissible action for the current step and choose it within {actions}.
</action>
"""


def project(raw: str) -> tuple[str, int]:
    low = raw.lower()
    s, e = low.find("<action>"), low.find("</action>")
    if s == -1 or e == -1:
        return low[-30:], 0
    action = low[s + len("<action>"):e].strip().lower()
    valid = int(raw.find("<plan>") != -1 and raw.find("</plan>") != -1 and not re.search(r"[一-鿿]", raw))
    return action, valid


_ENV = None


def _env():
    global _ENV
    if _ENV is None:
        os.environ.setdefault("JAVA_HOME", JAVA_HOME)
        os.environ["PATH"] = os.path.join(JAVA_HOME, "bin") + os.pathsep + os.environ.get("PATH", "")
        from scienceworld import ScienceWorldEnv
        _ENV = ScienceWorldEnv("", envStepLimit=MAX_STEPS + 10)
    return _ENV


def parse_id(gamefile: str):
    _, task, var = gamefile.split(":")
    return task, int(var)


@dataclass
class Episode:
    gamefile: str
    task: str = ""
    obs: str = ""
    admissible: list = field(default_factory=list)
    templates: list = field(default_factory=list)
    objects: list = field(default_factory=list)
    memory: list = field(default_factory=list)
    won: bool = False
    done: bool = False
    raw_score: float = 0.0

    @classmethod
    def start(cls, gamefile: str) -> "Episode":
        ep = cls(gamefile=gamefile)
        env = _env()
        task, var = parse_id(gamefile)
        env.load(task, var, SIMPLIFICATION)
        obs, info = env.reset()
        ep.task = env.get_task_description()
        ep.obs = obs.strip()
        ep._refresh(info)
        return ep

    def _refresh(self, info):
        """Admissible actions are checked against the full list, but the prompt shows action
        templates and the objects the agent can refer to (the full list can reach thousands)."""
        env = _env()
        self.admissible = sorted(info["valid"])
        self.templates = sorted(set(env.get_possible_actions()))
        self.objects = sorted(set(env.get_possible_objects()))

    @property
    def k(self) -> int:
        return len(self.memory)

    @property
    def score(self) -> float:
        return max(self.raw_score, 0.0) / 100.0

    def project(self, raw: str):
        return project(raw)

    def is_valid(self, action: str) -> int:
        return int(action in self.admissible)

    def prompt(self) -> str:
        acts = ("action templates (replace OBJ with an object) " + ", ".join(f"'{a}'" for a in self.templates) +
                "; objects you can refer to: " + ", ".join(self.objects))
        if self.k == 0:
            return TEMPLATE_NO_HIS.format(task=self.task, obs=self.obs, actions=acts)
        recent = self.memory[-HISTORY_LENGTH:]
        start = len(self.memory) - len(recent)
        hist = "\n".join(f"[Observation {start + j + 1}: '{r['text_obs']}', Action {start + j + 1}: '{r['action']}']"
                         for j, r in enumerate(recent))
        return TEMPLATE.format(task=self.task, k=self.k, h=len(recent), history=hist, step=self.k + 1,
                               obs=self.obs, actions=acts)

    def act_projected(self, action: str) -> None:
        obs, _reward, done, info = _env().step(action)
        self.memory.append({"text_obs": self.obs, "action": action})
        self.obs = obs.strip()
        self._refresh(info)
        self.raw_score = float(info["score"])
        self.won = self.raw_score >= 100
        self.done = bool(done) or self.won or self.k >= MAX_STEPS

    def act(self, raw: str, prompt: str | None = None):
        from aeb import Step
        if prompt is None:
            prompt = self.prompt()
        action, fv = project(raw)
        ok = self.is_valid(action)
        self.act_projected(action)
        return Step(self.k - 1, prompt, raw, action, fv, ok, self.obs, self.won, self.done)

    def close(self):
        pass


def games(split: str, per_task: int | None = None, seed: int = 0):
    """'train' / 'dev' / 'test' variations of all tasks, as game ids."""
    import random
    env = _env()
    out = []
    for task in env.get_task_names():
        env.load(task, 0, SIMPLIFICATION)
        vs = {"train": env.get_variations_train, "dev": env.get_variations_dev,
              "test": env.get_variations_test}[split]()
        vs = list(vs)
        random.Random(seed).shuffle(vs)
        out += [f"sciworld:{task}:{v}" for v in (vs[:per_task] if per_task else vs)]
    return out
