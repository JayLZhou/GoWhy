"""AgentsNet harness: replayable multi-agent executions with per-agent, per-turn logs.

Protocol, prompts, parsers, fallbacks and scorers follow AgentsNet (Groetschla et al., ICML
2025, github.com/floriangroetschla/AgentsNet, LiteralMessagePassing.py) verbatim. Each agent
is a stateful chat: a system prompt, then one user message per turn (the neighbours'
messages and the instructions) and the agent's reply. Turn 0 is the bootstrap ("first
messages"), turns 1..T-1 exchange messages, turn T asks for the final answer. Message
passing is synchronous: a message sent at turn t is read at turn t+1.

Replay. An execution log records, for every agent and turn, the hash of the agent's
whole input (its chat history including the new user message), its raw reply and any
fallback exchange. Running the same graph again with a log reuses the logged reply
whenever an agent's input is identical to the logged one, and calls the model only when
it differs. Since the logged reply was drawn from the same conditional distribution, the
result is an unbiased sample of the (possibly intervened) execution, and the model calls
are exactly the agent-turns inside the causal cone of the intervention.

Interventions at (agent v, turn t): resample (fresh draw), hint:<text> (append text to the
user message, then draw), replace:<text> (use text as the reply).

  python harness/agentsnet.py selftest
  python harness/agentsnet.py gen --task coloring --sizes 4 8 --reps 3 --served qwen2.5-7b
  python harness/agentsnet.py replay --trace store/agentsnet/coloring/ws_8_0_rep0.json --agent 3 --turn 2 --op resample
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
from networkx.readwrite import json_graph

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROOT = os.environ.get("GOWHY_ROOT", "/data2/yujia/Yingli/gowhy")
PARQUET = os.path.join(ROOT, "data", "agentsnet", "train.parquet")

# utils.names of AgentsNet: parse_messages accepts only these as recipients
NAMES = ["James", "Michael", "Robert", "John", "David", "William", "Richard", "Joseph", "Thomas", "Christopher",
         "Charles", "Daniel", "Matthew", "Anthony", "Mark", "Donald", "Steven", "Andrew", "Paul", "Joshua",
         "Kenneth", "Kevin", "Brian", "Timothy", "Ronald", "George", "Jason", "Edward", "Jeffrey", "Ryan", "Jacob",
         "Nicholas", "Gary", "Eric", "Jonathan", "Stephen", "Larry", "Justin", "Scott", "Brandon", "Benjamin",
         "Samuel", "Gregory", "Alexander", "Patrick", "Frank", "Raymond", "Jack", "Dennis", "Jerry", "Tyler",
         "Aaron", "Jose", "Adam", "Nathan", "Henry", "Zachary", "Douglas", "Peter", "Kyle", "Noah", "Ethan",
         "Jeremy", "Christian", "Walter", "Keith", "Austin", "Roger", "Terry", "Sean", "Gerald", "Carl", "Dylan",
         "Harold", "Jordan", "Jesse", "Bryan", "Lawrence", "Arthur", "Gabriel", "Bruce", "Logan", "Billy", "Joe",
         "Alan", "Juan", "Elijah", "Willie", "Albert", "Wayne", "Randy", "Mason", "Vincent", "Liam", "Roy", "Bobby",
         "Caleb", "Bradley", "Russell", "Lucas", "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara",
         "Susan", "Jessica", "Karen", "Sarah", "Lisa", "Nancy", "Sandra", "Betty", "Ashley", "Emily", "Kimberly",
         "Margaret", "Donna", "Michelle", "Carol", "Amanda", "Melissa", "Deborah", "Stephanie", "Rebecca", "Sharon",
         "Laura", "Cynthia", "Dorothy", "Amy", "Kathleen", "Angela", "Shirley", "Emma", "Brenda", "Pamela", "Nicole",
         "Anna", "Samantha", "Katherine", "Christine", "Debra", "Rachel", "Carolyn", "Janet", "Maria", "Olivia",
         "Heather", "Helen", "Catherine", "Diane", "Julie", "Victoria", "Joyce", "Lauren", "Kelly", "Christina",
         "Ruth", "Joan", "Virginia", "Judith", "Evelyn", "Hannah", "Andrea", "Megan", "Cheryl", "Jacqueline",
         "Madison", "Teresa", "Abigail", "Sophia", "Martha", "Sara", "Gloria", "Janice", "Kathryn", "Ann",
         "Isabella", "Judy", "Charlotte", "Julia", "Grace", "Amber", "Alice", "Jean", "Denise", "Frances",
         "Danielle", "Marilyn", "Natalie", "Beverly", "Diana", "Brittany", "Theresa", "Kayla", "Alexis", "Doris",
         "Lori", "Tiffany"]
NAME_SET = set(NAMES)

BOOTSTRAP = """You are an agent that is connected with other agents (your neighbors), who you communicate with. Your neighbors can in turn communicate with their neighbors and so forth. {short_task_description}
The rules are as follows:
1. There are {num_agents} agents in total. Everybody has a unique name. Your name is {own_name}.
2. You can only communicate with your immediate neighbors ({neighbors}). You cannot see or directly communicate with anyone else, unless information is relayed by intermediate agents.
3. You can exchange text-based messages with your neighbors in rounds. In each round, you will first receive the last messages sent by your neighbors and then be asked to generate your response messages which your neighbors receive in the next round. This process repeats for {num_rounds} rounds of message passing. Importantly, the process is synchronous: Every agent decides on which messages to send at the same time and sees the messages from other agents only in the next round.
4. Everybody (including you) decides what to share or request from neighbors. In every round, think step-by-step about the next set of messages you want to send. Output a JSON string that contains your response messages.
5. The messages you send to your neighbors are formatted as JSON. For example, if your neighbors are Alan and Bob, your output should look as follows:
```
{{
    "Alan": "Message that will be sent to Alan.",
    "Bob": "Message that will be sent to Bob.",
}}
```
It is not mandatory to send a message to every neighbor in every round. If you do not want to send a message to a particular neighbor, you may omit their name from the JSON.
6. After {num_rounds} message passes, you have to solve the following task: {long_problem_description}"""

COT = "Elaborate your chain of thought step-by-step first, then output the messages for your neighbors."
COT_FINAL = "\n\nElaborate your chain of thought step-by-step first, then answer the following: "
ASK_FIRST = "What are the first messages you want to send to your neighbors? {cot_prompt} Output your messages in JSON format as specified earlier."
FORMAT = ("{question} Format your answer as follows: '### Final Answer ###', followed by your final answer. "
          "Don't use any text for your final answer except one of these valid options: {valid_answers}")
FALLBACK_JSON = "Your messages could not be parsed into JSON. Please check your response and try again."
FALLBACK_ANSWER = "Your answer could not be parsed. Please check your answer and try again."

REPLAN_HINT = ("Before you reply, re-read the task rules and the messages you have received so far, state in one "
               "sentence what is still unresolved between you and your neighbors, and make your messages "
               "address exactly that.")


# ----------------------------------------------------------------------------- tasks
class Task:
    name = ""
    short = ""
    long = ""
    question = ""

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def valid_answers(self, node):
        raise NotImplementedError

    def score(self, answers):
        raise NotImplementedError


class Consensus(Task):
    name = "consensus"
    short = "Your goal is to agree on a single value with the other agents. The possible values that you can decide on are either 0 or 1."
    long = "After the last round, each agent must decide on a single value."
    question = "Please answer with the value you agreed on."

    def valid_answers(self, node):
        return ["0", "1"]

    def score(self, answers):
        valid = {"0", "1"}
        if not all(x in valid for x in answers) or (
                answers.count("0") != self.graph.order() and answers.count("0") != 0):
            return 0.0
        return 1.0


class LeaderElection(Task):
    name = "leader_election"
    short = "Your task is to collaboratively solve the problem of electing a single leader."
    long = ("You will be requested to state whether or not you are the leader. The response should either be 'Yes' or 'No'. "
            "The final result should be such that exactly one agent responds with 'Yes' and all others say 'No' as there should be exactly one leader.")
    question = "Are you the leader?"

    def valid_answers(self, node):
        return ["Yes", "No"]

    def score(self, answers):
        all_valid = all(x in {"No", "Yes"} for x in answers)
        return float(all_valid and sum(1 for x in answers if x == "Yes") == 1)


class Matching(Task):
    name = "matching"
    short = "Your task is to find build groups of two agents each which can communicate with each other."
    long = ("You will be requested to name one of your neighbors that you build a group with or 'None' if all your neighbors are already assigned to other groups and cannot be in a group with you."
            "In the end, every agent should only be in at most one group and agents in the same group have to name each other as the second group member consistently.")
    question = "Please answer with the name of the neighbor you build a group with or 'None' if all your neighbors are already assigned to other groups."

    def valid_answers(self, node):
        return [self.graph.nodes[u]["name"] for u in self.graph.neighbors(node)] + ["None"]

    def score(self, answers):
        g = self.graph
        names = [g.nodes[v]["name"] for v in g.nodes]
        name_to_match = {names[i]: answers[i] for i in range(len(names))}
        bad = 0
        for v in g.nodes:
            m = answers[v]
            if m != "None":
                if m not in [names[u] for u in g.neighbors(v)]:
                    bad += 1
                elif name_to_match[m] != names[v]:
                    bad += 1
            else:
                for u in g.neighbors(v):
                    if answers[u] == "None":
                        bad += 1
                        break
        return (g.order() - bad) / g.order()


class Coloring(Task):
    name = "coloring"
    short = "Your task is to partition yourselves into groups such that agents who are neighbors are never in the same group."
    question = "Which group do you assign yourself to?"

    def __init__(self, graph):
        super().__init__(graph)
        self.num_colors = max(graph.degree(n) for n in graph.nodes) + 1
        self.colors = [f"Group {i + 1}" for i in range(self.num_colors)]
        self.long = (f"You will be requested to state which group you assign yourself to. There are exactly {self.num_colors} groups available: Group 1,...,Group {self.num_colors}. You should assign yourself to exactly one of these groups. "
                     "The final result should be such that any two agents who are neighbors are in different groups. In particular, you should assign yourself to a group that is different from all of your neighbors' groups. ")

    def valid_answers(self, node):
        return self.colors

    def score(self, answers):
        all_valid = all(x in self.colors for x in answers)
        ok = sum(1 for u, v in self.graph.edges if answers[u] != answers[v])
        return float(all_valid) * ok / self.graph.number_of_edges()


class VertexCover(Task):
    name = "vertex_cover"
    short = """Your task is to select, among all agents, a group of coordinators
such that whenever two agents communicate at least one of them is a coordinator. The group of coordinators
should be selected such that every coordinator has at least one neighbor who is not a coordinator.
"""
    long = "You will be requested to state whether you are a coordinator. The response should either be 'Yes' or 'No'. "
    question = "Are you a coordinator?"

    def valid_answers(self, node):
        return ["Yes", "No"]

    def score(self, results):
        g = self.graph

        def cover(res):
            return sum(1 for u, v in g.edges if res[u] == "Yes" or res[v] == "Yes") / g.number_of_edges()

        coverage = cover(results)
        minimality = 0
        size = 0
        for u in g.nodes:
            if results[u] is not None and results[u] == "Yes":
                size += 1
                r = list(results)
                r[u] = "No"
                if cover(r) < 1.0:
                    minimality += 1
        return coverage * minimality / size if size else 0.0   # AgentsNet divides by zero here


TASKS = {t.name: t for t in (Consensus, LeaderElection, Matching, Coloring, VertexCover)}
LOCAL_ROUNDS = {4: 4, 8: 5, 16: 6}          # main.sh of AgentsNet


def rounds_for(task: str, graph: nx.Graph) -> int:
    if task in ("consensus", "leader_election") or graph.number_of_nodes() > 16:
        return 2 * nx.diameter(graph) + 1
    return LOCAL_ROUNDS[graph.number_of_nodes()]


# ----------------------------------------------------------------------------- graphs
_GRAPHS = None


def load_graph(generator: str, n: int, index: int) -> nx.Graph:
    global _GRAPHS
    if _GRAPHS is None:
        import pandas as pd
        df = pd.read_parquet(PARQUET)
        _GRAPHS = {(r.graph_generator, int(r.num_nodes), int(r["index"])): r.graph for _, r in df.iterrows()}
    d = json.loads(_GRAPHS[(generator, n, index)])
    return json_graph.node_link_graph(d["graph"], edges="links")


def spec_key(spec: dict) -> str:
    return f"{spec['task']}:{spec['generator']}:{spec['n']}:{spec['index']}"


# ----------------------------------------------------------------------------- parsing (verbatim semantics)
def parse_messages(response: str):
    import regex
    response = response.replace("\\n", "\n")
    for cand in regex.findall(r"(\{.*\})", response, regex.DOTALL, overlapped=True):
        try:
            out = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if all(k in NAME_SET for k in out.keys()) and all(isinstance(v, str) for v in out.values()):
            return out
    return None


def parse_answer(valid, message: str):
    pat = r"### Final Answer ###\s*(" + "|".join(re.escape(a) for a in valid) + r")"
    m = re.search(pat, message)
    return m.group(1) if m else None


def seed_of(key: str, rep: int, agent: int, turn: int, j: int) -> int:
    return int(hashlib.sha256(f"{key}|{rep}|{agent}|{turn}|{j}".encode()).hexdigest(), 16) % (2 ** 31)


def hist_hash(history: list) -> str:
    return hashlib.sha256(json.dumps(history, ensure_ascii=False).encode()).hexdigest()


# ----------------------------------------------------------------------------- execution
def run_execution(spec: dict, policy, rep: int, log: dict | None = None, intervene: dict | None = None,
                  workers: int = 16, rounds: int | None = None) -> dict:
    """Run (or replay) one execution. policy.chat(messages, seed) -> str.

    log: a previous trace of the same spec; replies are reused where inputs are identical.
    intervene: {"agent": v, "turn": t, "op": "resample" | "hint" | "replace", "text": ...}.
    Returns a trace with per-agent turn records, answers, score, success, calls, reused.
    """
    g = load_graph(spec["generator"], spec["n"], spec["index"])
    task = TASKS[spec["task"]](g)
    T = rounds or rounds_for(spec["task"], g)
    key = spec_key(spec)
    nodes = list(g.nodes())
    name = {v: g.nodes[v]["name"] for v in nodes}
    hist = {v: [] for v in nodes}
    out_msgs = {v: {} for v in nodes}           # messages produced at the current turn
    records = {v: [] for v in nodes}
    answers = [None] * len(nodes)
    stats = {"calls": 0, "reused": 0}
    logged = {int(v): recs for v, recs in (log or {}).get("agents", {}).items()}

    def user_text(v, turn):
        if turn == 0:
            return ASK_FIRST.format(cot_prompt=COT)
        rounds_left = T - turn
        last = rounds_left == 0
        received = {}
        for u in g.neighbors(v):
            if name[v] in out_msgs[u]:
                received[name[u]] = out_msgs[u][name[v]]
        s = ("Message passing has finished, here are the last messages you got from your neighbors:\n\n" if last
             else "These are the messages from your neighbors:\n\n")
        for nm, m in received.items():
            s += f"Message from {nm}:\n\n{m}\n\n"
        neigh_names = [name[u] for u in g.neighbors(v)]
        silent = [nm for nm in neigh_names if nm not in received]
        if silent:
            s += f"The following neighbors did not send you a messages in this round: {', '.join(silent)}\n\n"
        if not last:
            s += (f"{COT} Output your messages in JSON format as specified earlier. You have {rounds_left} rounds "
                  f"of communication left before you need to decide. Your neighbors are: {', '.join(neigh_names)} ")
            if rounds_left == 1:
                s += "These are the last messages that your neighbors will receive from you."
        else:
            s += COT_FINAL + FORMAT.format(question=task.question, valid_answers=", ".join(task.valid_answers(v)))
        return s

    def step(v, turn):
        """One agent-turn: build input, reuse or draw the reply, parse with one fallback."""
        last = turn == T
        iv = intervene if (intervene and intervene["agent"] == v and intervene["turn"] == turn) else None
        if turn == 0:
            sys_msg = BOOTSTRAP.format(short_task_description=task.short, long_problem_description=task.long,
                                       num_agents=g.order(), own_name=name[v],
                                       neighbors=", ".join(name[u] for u in g.neighbors(v)), num_rounds=T)
            hist[v] = [{"role": "system", "content": sys_msg}]
        u_text = user_text(v, turn)
        if iv and iv["op"] == "hint":
            u_text = u_text + "\n\n" + iv.get("text", REPLAN_HINT)
        hist[v].append({"role": "user", "content": u_text})
        h = hist_hash(hist[v])
        prev = logged.get(v)
        rec_log = prev[turn] if prev and turn < len(prev) else None
        reuse = rec_log is not None and rec_log["hist_hash"] == h and iv is None
        rec = {"turn": turn, "hist_hash": h, "user": u_text, "intervened": bool(iv)}
        if iv and iv["op"] == "replace":
            reply = iv["text"]
            rec["seed"] = None
        elif reuse:
            reply = rec_log["reply"]
            rec["seed"] = rec_log.get("seed")
            stats["reused"] += 1
        else:
            sd = seed_of(key, rep, v, turn, 0)
            reply = policy.chat(hist[v], sd)
            rec["seed"] = sd
            stats["calls"] += 1
        hist[v].append({"role": "assistant", "content": reply})
        rec["reply"] = reply
        parsed = parse_answer(task.valid_answers(v), reply) if last else parse_messages(reply)
        if parsed is None:
            hist[v].append({"role": "user", "content": FALLBACK_ANSWER if last else FALLBACK_JSON})
            if reuse and rec_log.get("fallback_reply") is not None:
                fb = rec_log["fallback_reply"]
                stats["reused"] += 1
            else:
                fb = policy.chat(hist[v], seed_of(key, rep, v, turn, 1))
                stats["calls"] += 1
                reuse = False
            hist[v].append({"role": "assistant", "content": fb})
            rec["fallback_reply"] = fb
            parsed = parse_answer(task.valid_answers(v), fb) if last else parse_messages(fb)
            if parsed is None and not last:
                parsed = {}
        rec["parsed"] = parsed
        rec["reused"] = reuse
        records[v].append(rec)
        return v, parsed

    t0 = time.time()
    with ThreadPoolExecutor(min(workers, len(nodes))) as ex:
        for turn in range(T + 1):
            res = list(ex.map(lambda v: step(v, turn), nodes))
            if turn < T:
                for v, parsed in res:
                    out_msgs[v] = parsed or {}
            else:
                for v, parsed in res:
                    answers[v] = parsed
    score = task.score(answers)
    return {"spec": spec, "rep": rep, "rounds": T, "graph": json_graph.node_link_data(g, edges="links"),
            "names": {str(v): name[v] for v in nodes}, "agents": {str(v): records[v] for v in nodes},
            "answers": answers, "score": score, "success": score >= 1.0, "calls": stats["calls"],
            "reused": stats["reused"], "intervene": intervene, "secs": round(time.time() - t0, 1)}


def causal_cone(g: nx.Graph, v: int, t: int, T: int) -> int:
    """Agent-turns whose input can depend on agent v's reply at turn t: v at turns t..T, and any u at turn s>t
    with dist(u,v) <= s-t. (Used to check that replay regenerates exactly these.)"""
    dist = nx.single_source_shortest_path_length(g, v)
    return sum(1 for s in range(t, T + 1) for u in g.nodes if dist.get(u, 10 ** 9) <= s - t)


# ----------------------------------------------------------------------------- CLI
class FakePolicy:
    """Deterministic stand-in for self-tests: reply depends on the input hash and the seed."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, seed=None):
        self.calls += 1
        h = hashlib.sha256((hist_hash(messages) + str(seed)).encode()).hexdigest()[:8]
        last = messages[-1]["content"]
        m = re.search(r"valid options: (.+)$", last, re.S)
        if "### Final Answer ###" in last and m:
            opts = [o.strip() for o in m.group(1).split(",")]
            return f"thinking {h}\n### Final Answer ###\n{opts[int(h, 16) % len(opts)]}"
        nb = re.search(r"immediate neighbors \(([^)]*)\)", messages[0]["content"])
        neigh = [x.strip() for x in nb.group(1).split(",")] if nb else NAMES[:12]
        return "thinking " + h + "\n```\n" + json.dumps({n: f"m{h}" for n in neigh}) + "\n```"


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--task", default="coloring")
    s = sub.add_parser("gen")
    s.add_argument("--task", nargs="+", default=["coloring"])
    s.add_argument("--generators", nargs="+", default=["ws", "ba", "dt"])
    s.add_argument("--sizes", type=int, nargs="+", default=[4, 8])
    s.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2, 3])
    s.add_argument("--reps", type=int, default=1)
    s.add_argument("--served", default="qwen2.5-7b")
    s.add_argument("--out", default=os.path.join(ROOT, "store", "agentsnet"))
    s.add_argument("--procs", type=int, default=8, help="executions in parallel")
    s = sub.add_parser("replay")
    s.add_argument("--trace", required=True)
    s.add_argument("--agent", type=int, required=True)
    s.add_argument("--turn", type=int, required=True)
    s.add_argument("--op", default="resample")
    s.add_argument("--text", default=None)
    s.add_argument("--rep", type=int, default=100)
    s.add_argument("--served", default="qwen2.5-7b")
    a = ap.parse_args()

    if a.cmd == "selftest":
        spec = {"task": a.task, "generator": "ws", "n": 8, "index": 0}
        pol = FakePolicy()
        tr = run_execution(spec, pol, rep=0)
        print(f"run: T={tr['rounds']} calls={tr['calls']} score={tr['score']:.2f}")
        pol2 = FakePolicy()
        tr2 = run_execution(spec, pol2, rep=0, log=tr)
        assert tr2["calls"] == 0 and tr2["score"] == tr["score"] and tr2["answers"] == tr["answers"], "replay differs"
        print(f"replay without intervention: calls={tr2['calls']} reused={tr2['reused']} (identical)")
        g = load_graph("ws", 8, 0)
        for (v, t) in [(3, 2), (0, 0), (5, tr["rounds"] - 1)]:
            pol3 = FakePolicy()
            tr3 = run_execution(spec, pol3, rep=1, log=tr, intervene={"agent": v, "turn": t, "op": "resample"})
            cone = causal_cone(g, v, t, tr["rounds"])
            print(f"intervene ({v},{t}): calls={tr3['calls']} cone={cone} reused={tr3['reused']} score={tr3['score']:.2f}")
            assert tr3["calls"] <= cone, "regenerated outside the causal cone"
        tr4 = run_execution(spec, FakePolicy(), rep=1, log=tr,
                            intervene={"agent": 3, "turn": 2, "op": "replace", "text": tr["agents"]["3"][2]["reply"]})
        assert tr4["calls"] == 0, "replacing with the logged reply should change nothing"
        print("replace with logged reply: calls=0 (ok)")
        print("SELFTEST OK")
        return

    import llm
    pol = llm.Policy(a.served)
    if a.cmd == "replay":
        log = json.load(open(a.trace))
        iv = {"agent": a.agent, "turn": a.turn, "op": a.op, "text": a.text}
        tr = run_execution(log["spec"], pol, rep=a.rep, log=log, intervene=iv)
        g = load_graph(log["spec"]["generator"], log["spec"]["n"], log["spec"]["index"])
        print(json.dumps({"score": tr["score"], "success": tr["success"], "calls": tr["calls"], "reused": tr["reused"],
                          "cone": causal_cone(g, a.agent, a.turn, log["rounds"]), "secs": tr["secs"]}))
        return

    # gen
    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.makedirs(a.out, exist_ok=True)
    jobs = [({"task": t, "generator": gen, "n": n, "index": i}, r)
            for t in a.task for gen in a.generators for n in a.sizes for i in a.indices for r in range(a.reps)]
    summ = open(os.path.join(a.out, "summary.jsonl"), "a")
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(a.procs) as ex:
        futs = {ex.submit(_gen_one, spec, r, a.served, a.out): (spec, r) for spec, r in jobs}
        for fu in as_completed(futs):
            spec, r = futs[fu]
            try:
                row = fu.result()
            except Exception as e:
                print("failed:", spec, r, repr(e)[:200], flush=True)
                continue
            summ.write(json.dumps(row) + "\n")
            summ.flush()
            done += 1
            print(f"  {done}/{len(jobs)} {spec_key(spec)} rep{r} score={row['score']:.2f} calls={row['calls']} "
                  f"{int(time.time() - t0)}s", flush=True)
    print("GEN_DONE", flush=True)


def _gen_one(spec, rep, served, out):
    import llm
    pol = llm.Policy(served)
    tr = run_execution(spec, pol, rep=rep)
    d = os.path.join(out, spec["task"])
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{spec['generator']}_{spec['n']}_{spec['index']}_rep{rep}.json")
    with open(path, "w") as f:
        json.dump(tr, f)
    return {"spec": spec, "rep": rep, "rounds": tr["rounds"], "score": tr["score"], "success": tr["success"],
            "calls": tr["calls"], "secs": tr["secs"], "path": path}


if __name__ == "__main__":
    main()
