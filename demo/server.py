"""GoWhy demo MCP server (stdio, newline-delimited JSON-RPC, standard library only).

    python3 server.py [--record-frustration]

Tools: what_if, identify, explain (GoWhy), and apply_offer (the demo world's
action tool). apply_offer is gated: it runs only when given the query_id of a
what_if that asked about that exact action and came back identifiable.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import world  # noqa: E402
from engine import Engine  # noqa: E402

PROTOCOLS = ["2025-06-18", "2025-03-26", "2024-11-05"]

INSTRUCTIONS = (
    "GoWhy is a causal database. Before calling any tool that changes the world (here: apply_offer), "
    "call what_if for that exact action and population, and read the status of every outcome. "
    "If an outcome is NOT IDENTIFIABLE, no number is returned on purpose: do not substitute the naive log "
    "contrast or a guess; report what is missing instead. Call explain with no arguments to see the variables."
)

OBJ = {"type": "object", "additionalProperties": True}
TOOLS = [
    {"name": "what_if",
     "description": ("Causal effect of taking an action, estimated from the logs, with a confidence interval and an "
                     "identifiability certificate. Returns no point estimate when the logs cannot answer the question. "
                     "Example: set={'discount': 1}, outcomes=['retained', 'refund'], where={'segment': 'B'}."),
     "inputSchema": {"type": "object", "required": ["set", "outcomes"], "properties": {
         "set": {**OBJ, "description": "the action: one variable and the value to set it to"},
         "outcomes": {"type": "array", "items": {"type": "string"}, "description": "outcome variables"},
         "where": {**OBJ, "description": "population, by pre-treatment variables, e.g. {'segment': 'B'}"},
         "baseline": {**OBJ, "description": "comparison value; defaults to the other value of a binary action"}}}},
    {"name": "identify",
     "description": ("Checks whether the effect of `treatment` on `outcome` can be answered from the recorded data, "
                     "without estimating it. If not, says what to log or how large an experiment to run."),
     "inputSchema": {"type": "object", "required": ["treatment", "outcome"], "properties": {
         "treatment": {"type": "string"}, "outcome": {"type": "string"}, "where": OBJ}}},
    {"name": "explain",
     "description": ("With a query_id: the estimand, identification path, assumptions, provenance of the edges relied "
                     "on, and which actions the query authorized. Without arguments: an overview of the database."),
     "inputSchema": {"type": "object", "properties": {"query_id": {"type": "string"}}}},
    {"name": "apply_offer",
     "description": ("DEMO WORLD ACTION (not part of GoWhy): roll out an offer to a customer segment. Requires the "
                     "query_id of a what_if that asked about this offer for this segment; refuses otherwise."),
     "inputSchema": {"type": "object", "required": ["segment", "offer", "query_id"], "properties": {
         "segment": {"type": "string", "enum": ["A", "B"]},
         "offer": {"type": "string", "enum": world.ACTIONS},
         "query_id": {"type": "string"}}}},
]


class Refused(Exception):
    pass


def apply_offer(engine: Engine, segment: str, offer: str, query_id: str):
    q = engine.queries.get(query_id)
    if q is None:
        raise Refused(f"no what_if with query_id {query_id!r}: ask what_if about this action first")
    if q["set"] != {offer: 1} or q["where"] != {"segment": segment}:
        raise Refused(f"{query_id} asked about {q['set']} WHERE {q['where']}, not {offer}=1 WHERE segment={segment}")
    blocked = [r["outcome"] for r in q["results"] if r["certificate"]["status"] != "identifiable"]
    if blocked:
        raise Refused(f"{query_id} could not identify the effect on {blocked}; acting on it would be a guess")

    q["authorized"].append(f"apply_offer(segment={segment}, offer={offer})")
    lines = [f"applied {offer} to segment {segment}, authorized by {query_id}",
             "realized effect (simulator ground truth) vs what GoWhy predicted:"]
    realized = {}
    for r in q["results"]:
        truth = world.true_effect(offer, r["outcome"], segment)
        realized[r["outcome"]] = truth
        lo, hi = r["interval"]
        inside = "inside" if lo <= truth <= hi else "OUTSIDE"
        lines.append(f"  {r['outcome']:<10}realized {truth:+.3f}   predicted {r['value']:+.3f} "
                     f"[{lo:+.3f}, {hi:+.3f}]   {inside} the interval")
    return "\n".join(lines), {"authorized_by": query_id, "realized": realized}


def call_tool(engine: Engine, name: str, args: dict):
    if name == "what_if":
        q = engine.what_if(args["set"], args["outcomes"], args.get("where"), args.get("baseline"))
        return engine.render(q), q
    if name == "identify":
        cert = engine.identify(args["treatment"], args["outcome"], args.get("where"))
        return json.dumps(cert, indent=2), cert
    if name == "explain":
        return engine.explain(args.get("query_id")), None
    if name == "apply_offer":
        return apply_offer(engine, args["segment"], args["offer"], args["query_id"])
    raise ValueError(f"unknown tool {name!r}")


def handle(engine: Engine, msg: dict):
    method, params = msg.get("method"), msg.get("params") or {}
    if method == "initialize":
        asked = params.get("protocolVersion")
        return {"protocolVersion": asked if asked in PROTOCOLS else PROTOCOLS[0],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "gowhy-demo", "version": "0.0.1"},
                "instructions": INSTRUCTIONS}
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        try:
            text, structured = call_tool(engine, params.get("name"), params.get("arguments") or {})
            out = {"content": [{"type": "text", "text": text}], "isError": False}
            if structured is not None:
                out["structuredContent"] = structured
            return out
        except (Refused, ValueError, KeyError) as e:
            kind = "REFUSED" if isinstance(e, Refused) else "ERROR"
            return {"content": [{"type": "text", "text": f"{kind}: {e}"}], "isError": True}
    raise LookupError(method)


def main():
    record = "--record-frustration" in sys.argv
    engine = Engine(world.NODES, world.EDGES, world.logs(record_frustration=record))
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if "id" not in msg:  # notification
            continue
        try:
            reply = {"jsonrpc": "2.0", "id": msg["id"], "result": handle(engine, msg)}
        except LookupError:
            reply = {"jsonrpc": "2.0", "id": msg["id"],
                     "error": {"code": -32601, "message": f"method not found: {msg.get('method')}"}}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
