"""Policy calls to local vLLM servers, matching AgentDebug's ChatOpenAI usage.

One user message, no system prompt, temperature 0.7, n = 1, no max_tokens,
reply stripped. Three attempts with backoff 0.5 s doubling; if all fail the
reply is the string "None" (AgentDebug's alfworld default).

Servers are found by served model name. Override with
GOWHY_SERVERS='{"qwen3-8b": [8041, 8042, 8043]}'.
"""
from __future__ import annotations

import itertools
import json
import os
import threading
import time

SERVERS = {
    "qwen2.5-7b": [8032],
    "qwen3-8b": [8041, 8042, 8043],
}
SERVERS.update(json.loads(os.environ.get("GOWHY_SERVERS", "{}")))
HOST = os.environ.get("GOWHY_HOST", "localhost")


class Policy:
    def __init__(self, name: str, temperature: float = 0.7, ports=None, timeout: float = 600):
        from openai import OpenAI

        self.name = name
        self.temperature = temperature
        ports = ports or SERVERS[name]
        self.clients = [OpenAI(base_url=f"http://{HOST}:{p}/v1", api_key="EMPTY", timeout=timeout,
                               max_retries=0) for p in ports]
        # random starting replica per process: with a fixed start, every worker's first call
        # hit the first replica and overloaded it (212 running + 117 waiting vs 61 elsewhere)
        import random
        off = random.randrange(len(self.clients))
        self._rr = itertools.cycle([(off + i) % len(self.clients) for i in range(len(self.clients))])
        self._lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.failures = 0

    def _client(self):
        with self._lock:
            return self.clients[next(self._rr)]

    def __call__(self, prompt: str, seed: int | None = None) -> str:
        """seed: vLLM per-request seed. Replays of different changes that use the
        same seed at the same (game, rep, step) are coupled (common random numbers)."""
        delay = 0.5
        extra = {} if seed is None else {"seed": int(seed)}
        for attempt in range(3):
            try:
                r = self._client().chat.completions.create(
                    model=self.name, messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature, n=1, **extra)
                with self._lock:
                    self.calls += 1
                    if r.usage:
                        self.prompt_tokens += r.usage.prompt_tokens
                        self.completion_tokens += r.usage.completion_tokens
                return (r.choices[0].message.content or "").strip()
            except Exception:
                if attempt == 2:
                    with self._lock:
                        self.failures += 1
                    return "None"
                time.sleep(delay)
                delay *= 2

    def usage(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "failures": self.failures}
