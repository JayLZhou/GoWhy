"""AgentDebug WebShop harness, reproduced from ulab-uiuc/AgentDebug.

  agentdebug/environments/prompts/webshop.py   (templates, verbatim)
  agentdebug/environments/webshop/projection.py (action parsing: last 20 chars on failure)
  agentdebug/environments/env_manager.py       (WebshopEnvironmentManager: format_obs,
                                                format_avail_actions, build_text_obs)
  agentdebug/environments/webshop/envs.py      (won = done and reward == 1.0)

Settings: SimpleMemory with history_length = 3 (rollout.py default), max 30 steps,
temperature 0.7, 1000-product catalogue (items_shuffle_1000.json), synthetic goals.
Goals 0-499 are AgentDebug's test split, 500+ its train split.

The AgentErrorBench WebShop logs used an LLM-written history summary, so they
cannot be reproduced exactly; our stores use the deterministic SimpleMemory.

A game id is "webshop:<goal index>". One WebAgentTextEnv per process is reused
across episodes (loading the catalogue, index and spaCy is slow).
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field

WEBSHOP_DIR = os.environ.get("WEBSHOP_DIR", "/data2/yujia/Yingli/gowhy_setup/webshop_src/webshop")
MAX_STEPS = 30
HISTORY_LENGTH = 3

WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert agent operating in the WebShop e-commerce environment.
Your task is: {task_description}
Your current observation is: {current_observation}
Your admissible actions of the current situation are: {available_actions}.

Please begin by analyzing the situation and planning your approach:

<plan>
Plan the next step:
- Given what I've learned, what should I do next?
- Please explain why this plan is helpful for the next action?
- What do I expect this action to achieve?
</plan>

<action>
Finally, choose ONE admissible action for the current step and choose it within {available_actions}.
</action>
"""

WEBSHOP_TEMPLATE = """
You are an expert agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below is a compact summary of all steps: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: {available_actions}.

Now it's your turn to take an action.

You should first recall relevant past experience and reason from the history context, then MUST summarize within <memory> </memory> tags like this:

<memory>
Look at the history context above.
- Please retrieve the most relevant memory for this step including the relevant observation and action in a RAG style along with the step number.
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
Finally, choose ONE admissible action for the current step and choose it within {available_actions}.
</action>
"""


def project(raw: str) -> tuple[str, int]:
    """webshop_projection for one action: (action sent to env, format_valid)."""
    low = raw.lower()
    s, e = low.find("<action>"), low.find("</action>")
    if s == -1 or e == -1:
        return low[-20:], 0
    action = low[s + len("<action>"):e].strip().lower()
    valid = 1
    if raw.find("<plan>") == -1 or raw.find("</plan>") == -1:
        valid = 0
    if re.search(r"[一-鿿]", raw):
        valid = 0
    return action, valid


def format_avail(avail: dict) -> list[str]:
    acts = []
    for key in avail:
        if key not in ("has_search_bar", "clickables"):
            raise ValueError(f"Unknown key in available actions: {key}")
    if avail["has_search_bar"]:
        acts.append("search[<your query>]")
    for txt in avail["clickables"]:
        acts.append(f"click[{txt}]")
    return acts


def is_valid_action(action: str, avail: dict) -> int:
    if action.startswith("search[") and action.endswith("]"):
        return int(bool(avail["has_search_bar"]) and len(action) > len("search[]"))
    m = re.fullmatch(r"click\[(.+)\]", action)
    return int(bool(m) and m.group(1) in [c.lower() for c in avail["clickables"]])


_ENV = None


JAVA_HOME = os.environ.get("WEBSHOP_JAVA_HOME", "/data2/yujia/Yingli/gowhy_setup/envs/webshop/lib/jvm")


def _env():
    global _ENV
    if _ENV is None:
        # pyserini runs Lucene through pyjnius, which needs the conda JDK's libjvm.so
        os.environ.setdefault("JAVA_HOME", JAVA_HOME)
        os.environ.setdefault("JVM_PATH", os.path.join(JAVA_HOME, "lib", "server", "libjvm.so"))
        if WEBSHOP_DIR not in sys.path:
            sys.path.append(WEBSHOP_DIR)
        import gym
        from web_agent_site.envs import WebAgentTextEnv  # noqa: F401  (registers the env)
        _ENV = gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=None, seed=0)
    return _ENV


def goal_of(gamefile: str) -> int:
    assert gamefile.startswith("webshop:"), gamefile
    return int(gamefile.split(":", 1)[1])


@dataclass
class Episode:
    gamefile: str
    task: str = ""
    obs: str = ""              # formatted observation (format_obs)
    avail: dict = field(default_factory=dict)
    admissible: list = field(default_factory=list)
    memory: list = field(default_factory=list)
    won: bool = False
    done: bool = False
    score: float = 0.0

    @classmethod
    def start(cls, gamefile: str) -> "Episode":
        ep = cls(gamefile=gamefile)
        env = _env()
        obs, _ = env.reset(session=goal_of(gamefile))
        parts = obs.split(" [SEP] ")
        assert parts[1] == "Instruction:", obs[:200]
        ep.task = parts[2]
        ep.obs = ep._format(obs)
        ep._refresh()
        return ep

    def _format(self, raw_obs: str) -> str:
        parts = raw_obs.split(" [SEP] ")
        try:
            i = parts.index(self.task)
            return " [SEP] ".join(f"'{p}'" for p in parts[i + 1:])
        except ValueError:
            return raw_obs

    def _refresh(self):
        self.avail = _env().get_available_actions()
        self.admissible = format_avail(self.avail)

    @property
    def k(self) -> int:
        return len(self.memory)

    def project(self, raw: str):
        return project(raw)

    def is_valid(self, action: str) -> int:
        return is_valid_action(action, self.avail)

    def prompt(self) -> str:
        adm = "\n".join(f"'{s}'," for s in self.admissible)
        if self.k == 0:
            return WEBSHOP_TEMPLATE_NO_HIS.format(task_description=self.task, current_observation=self.obs,
                                                  available_actions=adm)
        recent = self.memory[-HISTORY_LENGTH:]
        start = len(self.memory) - len(recent)
        hist = "\n".join(f"[Observation {start + j + 1}: '{r['text_obs']}', Action {start + j + 1}: '{r['action']}']"
                         for j, r in enumerate(recent))
        p = WEBSHOP_TEMPLATE.format(task_description=self.task, step_count=len(self.memory),
                                    history_length=len(recent), action_history=hist,
                                    current_step=len(self.memory) + 1, current_observation=self.obs,
                                    available_actions=adm)
        if len(p) > 13000:
            p = WEBSHOP_TEMPLATE_NO_HIS.format(task_description=self.task, current_observation=self.obs,
                                               available_actions=adm)
        return p

    def act_projected(self, action: str) -> None:
        obs, reward, done, _info = _env().step(action)
        self.memory.append({"text_obs": self.obs, "action": action})
        self.obs = self._format(obs)
        self._refresh()
        self.score = float(reward or 0.0)
        self.won = bool(done) and reward == 1.0
        self.done = bool(done) or self.k >= MAX_STEPS

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


def goals(split: str) -> list[str]:
    """'test': AgentDebug's test goals 0-499; 'train': 500 onward."""
    n = len(_env().server.goals)
    rng = range(0, 500) if split == "test" else range(500, n)
    return [f"webshop:{i}" for i in rng]
