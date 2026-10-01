"""Minimal GoWhy engine for the demo: identify -> estimate -> certificate.

Scope (see ../DESIGN.md section 7 for the full pipeline): one treatment, binary
outcomes, one observational view. Identification is back-door with a complete
ID-algorithm check behind it, so a "not identifiable" verdict is not just
"no adjustment set found".
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd


class Graph:
    """Causal DAG. Variables that are not logged are simply absent from `recorded`."""

    def __init__(self, nodes, edges):
        self.pa = {v: set() for v in nodes}
        for a, b in edges:
            self.pa[b].add(a)

    @property
    def nodes(self):
        return set(self.pa)

    def edges(self):
        return {(a, b) for b, ps in self.pa.items() for a in ps}

    def children(self):
        ch = {v: set() for v in self.pa}
        for a, b in self.edges():
            ch[a].add(b)
        return ch

    def _closure(self, start, step):
        out, stack = set(start), list(start)
        while stack:
            for w in step[stack.pop()]:
                if w not in out:
                    out.add(w)
                    stack.append(w)
        return out

    def ancestors(self, s):
        return self._closure(s, self.pa)

    def descendants(self, s):
        return self._closure(s, self.children())

    def without_outgoing(self, s):
        return Graph(self.nodes, [(a, b) for a, b in self.edges() if a not in s])

    def d_separated(self, xs, ys, zs):
        """xs _||_ ys | zs, via the moralized ancestral graph."""
        xs, ys, zs = set(xs), set(ys), set(zs)
        anc = self.ancestors(xs | ys | zs)
        adj = {v: set() for v in anc}
        for v in anc:
            for p in self.pa[v]:
                adj[v].add(p)
                adj[p].add(v)
            for p, q in itertools.combinations(self.pa[v], 2):
                adj[p].add(q)
                adj[q].add(p)
        seen = set(xs - zs)
        stack = list(seen)
        while stack:
            v = stack.pop()
            if v in ys:
                return False
            for w in adj[v]:
                if w not in seen and w not in zs:
                    seen.add(w)
                    stack.append(w)
        return True


def backdoor_set(g: Graph, x: str, y: str, recorded: set, must: set = frozenset()):
    """A minimal back-door set among recorded non-descendants of x, containing `must`; None if none exists."""
    candidates = recorded - g.descendants({x}) - {y}
    if not set(must) <= candidates:
        return None
    gx = g.without_outgoing({x})
    # If any separator exists inside `candidates`, this ancestral one is a separator (van der Zander et al.).
    z = (gx.ancestors({x, y} | set(must)) & candidates) | set(must)
    if not gx.d_separated({x}, {y}, z):
        return None
    for v in sorted(z - set(must)):
        if gx.d_separated({x}, {y}, z - {v}):
            z = z - {v}
    return z


def _project(g: Graph, recorded: set):
    """Latent projection onto the recorded variables: directed and bidirected edges."""
    ch = g.children()

    def reach(v):  # recorded nodes reachable through unrecorded intermediates only
        out, seen, stack = set(), set(), [v]
        while stack:
            for c in ch[stack.pop()]:
                if c in recorded:
                    out.add(c)
                elif c not in seen:
                    seen.add(c)
                    stack.append(c)
        return out

    di = {(a, b) for a in recorded for b in reach(a)}
    bi = set()
    for hidden in g.nodes - recorded:
        for a, b in itertools.combinations(sorted(reach(hidden)), 2):
            bi.add(frozenset((a, b)))
    return di, bi


def _id(y: frozenset, x: frozenset, v: frozenset, di, bi) -> bool:
    """Shpitser-Pearl ID algorithm, decision only: is P(y | do(x)) identifiable?"""
    def an(s, edges):
        pa = {n: set() for n in v}
        for a, b in edges:
            pa[b].add(a)
        out, stack = set(s), list(s)
        while stack:
            for p in pa[stack.pop()]:
                if p not in out:
                    out.add(p)
                    stack.append(p)
        return frozenset(out)

    def components(nodes):
        nodes, out = set(nodes), []
        while nodes:
            comp, stack = set(), [next(iter(nodes))]
            while stack:
                n = stack.pop()
                if n in comp:
                    continue
                comp.add(n)
                stack.extend(m for e in bi if n in e for m in e if m in nodes and m not in comp)
            out.append(frozenset(comp))
            nodes -= comp
        return out

    def restrict(s):
        return {(a, b) for a, b in di if a in s and b in s}, {e for e in bi if e <= s}

    if not x:
        return True
    a = an(y, di)
    if a != v:
        return _id(y, x & a, a, *restrict(a))
    w = (v - x) - an(y, {(p, c) for p, c in di if c not in x})
    if w:
        return _id(y, x | w, v, di, bi)
    comps = components(v - x)
    if len(comps) > 1:
        return all(_id(s, v - s, v, di, bi) for s in comps)
    s, whole = comps[0], components(v)
    if len(whole) == 1:
        return False  # hedge
    if s in whole:
        return True
    sp = next(c for c in whole if s < c)
    return _id(y, x & sp, sp, *restrict(sp))


def identifiable(g: Graph, x: str, y: str, recorded: set) -> bool:
    di, bi = _project(g, recorded)
    return _id(frozenset({y}), frozenset({x}), frozenset(recorded), di, bi)


def _f(v):
    return f"{v:+.3f}"


class Engine:
    def __init__(self, nodes, edges, data: pd.DataFrame, view: str = "orders_log",
                 evidence: str = "observational"):
        self.graph = Graph(nodes, [(a, b) for a, b, _ in edges])
        self.provenance = {(a, b): p for a, b, p in edges}
        self.data = data
        self.recorded = set(data.columns)
        self.view, self.evidence = view, evidence
        self.queries: dict[str, dict] = {}

    # ---------------------------------------------------------------- identify

    def _check(self, treatment, outcome, where):
        for name in [treatment, outcome, *where]:
            if name not in self.graph.nodes:
                raise ValueError(f"unknown variable {name!r}; known: {sorted(self.graph.nodes)}")
        post = set(where) & self.graph.descendants({treatment})
        if post:
            raise ValueError(f"WHERE may only use pre-treatment variables; {sorted(post)} are affected by {treatment}")

    def identify(self, treatment: str, outcome: str, where: dict | None = None) -> dict:
        where = where or {}
        self._check(treatment, outcome, where)
        g, rec = self.graph, self.recorded
        cert = {"status": "not", "strategy": None, "formula": None, "adjustment_set": [],
                "view": self.view, "evidence": self.evidence, "assumptions": [], "missing": [],
                "reason": None, "complete": True}

        absent = [v for v in [treatment, outcome, *where] if v not in rec]
        if absent:
            cert["reason"] = f"{absent} not recorded in {self.view}"
            cert["missing"] = [{"action": "log", "variables": absent}]
            return cert

        if outcome not in g.descendants({treatment}):
            cert.update(status="identifiable", strategy="no-causal-path", formula="effect = 0",
                        assumptions=[f"no directed path from {treatment} to {outcome} in the graph"])
            return cert

        z = backdoor_set(g, treatment, outcome, rec, set(where))
        if z is not None:
            adj = sorted(z - set(where))
            cert.update(
                status="identifiable", strategy="backdoor", adjustment_set=adj,
                formula="sum_z P(y | x, z, w) P(z | w)" if adj else "P(y | x, w)",
                assumptions=[
                    f"the recorded graph is correct among ancestors of {treatment} and {outcome}",
                    f"no unrecorded common cause of {treatment} and {outcome} beyond what {sorted(z) or '{}'} blocks",
                    f"both values of {treatment} occur in every stratum (checked on the data)",
                    f"{self.view} describes the population the action targets",
                ])
            return cert

        hidden = sorted((g.ancestors({treatment, outcome}) - rec))
        if identifiable(g, treatment, outcome, rec):
            cert.update(status="identifiable", strategy="id",
                        reason="identifiable by the ID algorithm, but this demo only estimates back-door queries")
            return cert

        cert["reason"] = (f"{treatment} and {outcome} share a cause that {self.view} does not record "
                          f"({', '.join(hidden)}); the ID algorithm returns a hedge")
        cert["missing"] = self._missing(treatment, outcome, where)
        return cert

    def _missing(self, treatment, outcome, where, eps: float = 0.02):
        out = []
        loggable = sorted(self.graph.nodes - self.recorded)
        for k in (1, 2):
            hits = [list(c) for c in itertools.combinations(loggable, k)
                    if backdoor_set(self.graph, treatment, outcome, self.recorded | set(c), set(where)) is not None]
            if hits:
                out += [{"action": "log", "variables": h} for h in hits]
                break
        p = float(self.data[outcome].mean())
        n = math.ceil(2 * 1.96 ** 2 * p * (1 - p) / eps ** 2)
        out.append({"action": "experiment", "randomize": [treatment], "n_per_arm": n,
                    "precision": f"+/-{eps} at 95%"})
        return out

    # ---------------------------------------------------------------- estimate

    def _subset(self, where):
        df = self.data
        for k, v in where.items():
            df = df[df[k].astype(str) == str(v)]
        if df.empty:
            raise ValueError(f"no rows match WHERE {where}")
        return df

    @staticmethod
    def _levels(df, treatment, value, baseline):
        levels = sorted(df[treatment].unique().tolist())
        match = [lv for lv in levels if str(lv) == str(value)]
        if not match:
            raise ValueError(f"{treatment}={value!r} never occurs in the log; observed values: {levels}")
        x1 = match[0]
        if baseline is not None:
            base = [lv for lv in levels if str(lv) == str(baseline)]
            if not base:
                raise ValueError(f"baseline {treatment}={baseline!r} never occurs; observed values: {levels}")
            return x1, base[0]
        others = [lv for lv in levels if lv != x1]
        if len(others) != 1:
            raise ValueError(f"{treatment} has values {levels}; pass a baseline")
        return x1, others[0]

    @staticmethod
    def _stratified(df, x, x1, x0, y, z, reps: int = 200, seed: int = 0):
        yv = df[y].to_numpy(float)
        a1, a0 = (df[x] == x1).to_numpy(float), (df[x] == x0).to_numpy(float)
        s = df.groupby(sorted(z)).ngroup().to_numpy() if z else np.zeros(len(df), dtype=int)
        k, n = int(s.max()) + 1, len(df)

        def effect(idx):
            si, yi = s[idx], yv[idx]
            w = np.bincount(si, minlength=k) / len(si)
            n1, n0 = (np.bincount(si, weights=a[idx], minlength=k) for a in (a1, a0))
            t1, t0 = (np.bincount(si, weights=a[idx] * yi, minlength=k) for a in (a1, a0))
            ok = (n1 > 0) & (n0 > 0)
            m1 = np.divide(t1, n1, out=np.zeros(k), where=n1 > 0)
            m0 = np.divide(t0, n0, out=np.zeros(k), where=n0 > 0)
            return w, m1, m0, ok, np.minimum(n1, n0)

        w, m1, m0, ok, cell = effect(np.arange(n))
        gap = float(w[~ok].sum())
        rng = np.random.default_rng(seed)
        boot = []
        for _ in range(reps):
            bw, b1, b0, bok, _ = effect(rng.integers(0, n, n))
            boot.append(float((bw[bok] * (b1 - b0)[bok]).sum() / bw[bok].sum()))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {"value": float((w * (m1 - m0)).sum()), "interval": [float(lo), float(hi)],
                "do_value": float((w * m1).sum()), "baseline_value": float((w * m0).sum()),
                "gap": gap, "strata": k, "min_cell": int(cell[w > 0].min()), "n": n,
                "bootstrap": reps, "seed": seed}

    @staticmethod
    def _bounds(df, x, x1, x0, y):
        """No-assumption bounds for a [0, 1] outcome."""
        def level(v):
            p = float((df[x] == v).mean())
            m = float(df.loc[df[x] == v, y].mean())
            return p * m, p * m + (1 - p)
        (l1, h1), (l0, h0) = level(x1), level(x0)
        return [l1 - h0, h1 - l0]

    # ------------------------------------------------------------------ what_if

    def what_if(self, set: dict, outcomes: list, where: dict | None = None, baseline: dict | None = None) -> dict:
        where = where or {}
        if len(set) != 1:
            raise ValueError("the demo engine takes exactly one action variable in `set`")
        (treatment, value), = set.items()
        df = self._subset(where)
        x1, x0 = self._levels(df, treatment, value, (baseline or {}).get(treatment))
        results = []
        for y in outcomes:
            cert = self.identify(treatment, y, where)
            naive = float(df.loc[df[treatment] == x1, y].mean() - df.loc[df[treatment] == x0, y].mean())
            res = {"outcome": y, "value": None, "interval": None, "certificate": cert,
                   "explain": {"naive_log_contrast": naive, "n": len(df)}}
            if cert["status"] == "identifiable" and cert["strategy"] == "backdoor":
                est = self._stratified(df, treatment, x1, x0, y, cert["adjustment_set"])
                if est["gap"] > 0:
                    cert.update(status="partial", reason=f"{est['gap']:.1%} of the population has no support for one arm")
                    res["interval"] = self._bounds(df, treatment, x1, x0, y)
                else:
                    res["value"], res["interval"] = est["value"], est["interval"]
                res["explain"].update(estimator="stratified back-door, bootstrap CI", **est)
            elif cert["status"] == "identifiable" and cert["strategy"] == "no-causal-path":
                res["value"], res["interval"] = 0.0, [0.0, 0.0]
            elif cert["status"] == "not" and treatment in self.recorded and y in self.recorded:
                res["interval"] = self._bounds(df, treatment, x1, x0, y)
                res["explain"]["estimator"] = "no-assumption bounds"
            results.append(res)

        qid = f"q-{len(self.queries) + 1:04d}"
        query = {"query_id": qid, "kind": "what_if", "set": {treatment: x1}, "baseline": {treatment: x0},
                 "where": where, "results": results, "authorized": []}
        self.queries[qid] = query
        return query

    # ------------------------------------------------------------------- text

    def render(self, q: dict) -> str:
        (t, x1), = q["set"].items()
        x0 = q["baseline"][t]
        where = " WHERE " + ", ".join(f"{k} = {v}" for k, v in q["where"].items()) if q["where"] else ""
        lines = [f"WHAT IF {t} = {x1} (vs {t} = {x0}){where}"]
        for r in q["results"]:
            c, ex = r["certificate"], r["explain"]
            pad = " " * 12
            if c["status"] == "identifiable":
                if c["strategy"] == "no-causal-path":
                    lines.append(f"  {r['outcome']:<10}no effect  IDENTIFIABLE  the graph has no causal path "
                                 f"from {t} to {r['outcome']}")
                    continue
                if r["value"] is None:
                    lines.append(f"  {r['outcome']:<10}IDENTIFIABLE in principle; {c['reason']}")
                    continue
                adj = c["adjustment_set"]
                how = f"back-door, adjusted for {{{', '.join(adj)}}}" if adj else "back-door, no adjustment needed"
                lo, hi = r["interval"]
                lines.append(f"  {r['outcome']:<10}{_f(r['value'])}  95% CI [{_f(lo)}, {_f(hi)}]  IDENTIFIABLE  "
                             f"{how}  ({c['view']}, n={ex['n']:,})")
                if not adj:  # the naive contrast is the estimate itself
                    continue
            else:
                label = "PARTIALLY IDENTIFIABLE" if c["status"] == "partial" else "NOT IDENTIFIABLE"
                lines.append(f"  {r['outcome']:<10}{label} from {c['view']}: no point estimate is returned")
                lines.append(f"{pad}reason: {c['reason']}")
                if r["interval"]:
                    lo, hi = r["interval"]
                    lines.append(f"{pad}the data alone only say the effect lies in [{_f(lo)}, {_f(hi)}]")
                for m in c["missing"]:
                    if m["action"] == "log":
                        lines.append(f"{pad}to answer it: start logging {', '.join(m['variables'])}")
                    elif m["action"] == "experiment":
                        lines.append(f"{pad}to answer it: randomize {', '.join(m['randomize'])} on "
                                     f"{m['n_per_arm']:,} customers per arm ({m['precision']})")
            lines.append(f"{pad}naive log contrast {_f(ex['naive_log_contrast'])}  (confounded, not a causal effect)")
        lines.append(f"query_id: {q['query_id']}  (explain shows the identification path and assumptions)")
        return "\n".join(lines)

    def explain(self, query_id: str | None = None) -> str:
        if query_id is None:
            g = self.graph
            lines = [f"GoWhy demo database: view {self.view} ({self.evidence}), {len(self.data):,} rows",
                     f"recorded variables: {', '.join(sorted(self.recorded))}",
                     f"in the graph but not recorded: {', '.join(sorted(g.nodes - self.recorded)) or '(none)'}",
                     "causal edges (with provenance):"]
            lines += [f"  {a} -> {b}   [{self.provenance[(a, b)]}]" for a, b in sorted(g.edges())]
            lines.append("ask: what_if(set, outcomes, where) / identify(treatment, outcome, where) / explain(query_id)")
            return "\n".join(lines)
        if query_id not in self.queries:
            raise ValueError(f"unknown query_id {query_id!r}; known: {sorted(self.queries)}")
        q = self.queries[query_id]
        (t, x1), = q["set"].items()
        lines = [self.render(q), "", "identification, per outcome:"]
        for r in q["results"]:
            c, ex = r["certificate"], r["explain"]
            y = r["outcome"]
            cond = "".join(f", {k}={v}" for k, v in q["where"].items())
            lines.append(f"- {y}: estimand E[{y} | do({t}={x1}){cond}] - E[{y} | do({t}={q['baseline'][t]}){cond}]")
            lines.append(f"    status {c['status']}; strategy {c['strategy']}; formula {c['formula']}")
            if "estimator" in ex:
                detail = (f", {ex['strata']} strata, smallest arm-cell {ex['min_cell']:,}, "
                          f"{ex['bootstrap']} bootstrap draws, seed {ex['seed']}") if "strata" in ex else ""
                lines.append(f"    estimator {ex['estimator']}{detail}")
            for a in c["assumptions"]:
                lines.append(f"    assumes: {a}")
            if c["reason"]:
                lines.append(f"    reason: {c['reason']}")
            anc = self.graph.ancestors({t, y})
            lines.append("    edges relied on:")
            lines += [f"      {a} -> {b}   [{self.provenance[(a, b)]}]"
                      for a, b in sorted(self.graph.edges()) if a in anc and b in anc]
        if q["authorized"]:
            lines.append("actions this query authorized:")
            lines += [f"  {a}" for a in q["authorized"]]
        else:
            lines.append("actions this query authorized: none")
        return "\n".join(lines)
