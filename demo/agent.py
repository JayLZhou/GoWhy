"""A scripted "consult before you act" agent that talks to the GoWhy demo server over real MCP stdio.

    python3 agent.py

No LLM is involved: the decision rule is three lines of arithmetic. The point is
the wire: every number the agent acts on comes from a what_if call, and the
action tool refuses to run without one.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Demo economics, per customer (arbitrary numbers, only here to give the agent a decision rule).
LTV, REFUND_COST = 300.0, 250.0
OFFER_COST = {"discount": 6.0, "free_shipping": 4.0, "support_call": 3.0}


class MCPClient:
    def __init__(self, *server_args: str):
        self.proc = subprocess.Popen([sys.executable, os.path.join(HERE, "server.py"), *server_args],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self._id = 0
        self.info = self.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                                "clientInfo": {"name": "scripted-agent", "version": "0"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _send(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def request(self, method, params=None):
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}})
        reply = json.loads(self.proc.stdout.readline())
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["result"]

    def call(self, name, **arguments):
        r = self.request("tools/call", {"name": name, "arguments": arguments})
        return r["content"][0]["text"], r.get("structuredContent"), r["isError"]

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


def say(text=""):
    print(text)


def show(text):
    print("\n".join("    | " + line for line in text.splitlines()))


def consider(mcp: MCPClient, offer: str, segment: str):
    """Ask before acting. Returns (query_id, net value per customer or None if the question cannot be answered)."""
    say(f"\n[agent] 候选动作：给 {segment} 段客户 {offer}。先问 what_if。")
    text, q, _ = mcp.call("what_if", set={offer: 1}, outcomes=["retained", "refund"], where={"segment": segment})
    show(text)
    effects = {r["outcome"]: r["value"] for r in q["results"]}
    if any(v is None for v in effects.values()):
        say("[agent] 有结果不可识别，没有可用的数，不做这个动作。")
        return q["query_id"], None
    net = effects["retained"] * LTV - effects["refund"] * REFUND_COST - OFFER_COST[offer]
    say(f"[agent] 每客户净值 = {effects['retained']:+.3f}×{LTV:.0f} − {effects['refund']:+.3f}×{REFUND_COST:.0f} "
        f"− {OFFER_COST[offer]:.0f} = {net:+.2f}")
    return q["query_id"], net


def main():
    mcp = MCPClient()
    tools = [t["name"] for t in mcp.request("tools/list")["tools"]]
    say(f"[mcp] 已连接 {mcp.info['serverInfo']['name']}，工具：{', '.join(tools)}")
    say("[agent] 目标：提升 B 段客户留存。")

    say("\n=== 第 0 幕：不问就动 ===")
    text, _, _ = mcp.call("apply_offer", segment="B", offer="discount", query_id="none")
    show(text)

    say("\n=== 第 1 幕：原计划是打折 ===")
    _, net = consider(mcp, "discount", "B")
    say("[agent] 净值为负，放弃打折。" if net is not None and net < 0 else "[agent] 净值为正。")

    say("\n=== 第 2 幕：换成包邮 ===")
    qid, net = consider(mcp, "free_shipping", "B")
    if net is not None and net > 0:
        say(f"[agent] 净值为正，执行，带上 {qid} 作为依据。")
        text, _, _ = mcp.call("apply_offer", segment="B", offer="free_shipping", query_id=qid)
        show(text)

    say("\n=== 第 3 幕：主动回访这个动作，日志答不了 ===")
    qid_call, net = consider(mcp, "support_call", "B")
    say("[agent] 如果硬要执行：")
    text, _, _ = mcp.call("apply_offer", segment="B", offer="support_call", query_id=qid_call)
    show(text)

    say("\n=== 第 4 幕：事后审计，这次行动凭的是什么 ===")
    text, _, _ = mcp.call("explain", query_id=qid)
    show(text)
    mcp.close()

    say("\n=== 第 5 幕：按建议把 frustration 记进日志之后，同一个问题 ===")
    mcp = MCPClient("--record-frustration")
    consider(mcp, "support_call", "B")
    mcp.close()


if __name__ == "__main__":
    main()
