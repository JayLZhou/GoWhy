"""pytest test_demo.py"""
import pytest

import world
from agent import MCPClient
from engine import Engine, Graph, backdoor_set, identifiable


def engine(**kw):
    return Engine(world.NODES, world.EDGES, world.logs(**kw))


# ---- graph logic: fixed cases with known answers ----

def test_bow_graph_is_not_identifiable():
    g = Graph("XYU", [("X", "Y"), ("U", "X"), ("U", "Y")])
    assert backdoor_set(g, "X", "Y", {"X", "Y"}) is None
    assert not identifiable(g, "X", "Y", {"X", "Y"})


def test_front_door_graph_is_identifiable_without_a_backdoor_set():
    g = Graph("XMYU", [("X", "M"), ("M", "Y"), ("U", "X"), ("U", "Y")])
    assert backdoor_set(g, "X", "Y", {"X", "M", "Y"}) is None
    assert identifiable(g, "X", "Y", {"X", "M", "Y"})


def test_napkin_graph_is_identifiable():
    g = Graph(["W", "Z", "X", "Y", "U1", "U2"],
              [("W", "Z"), ("Z", "X"), ("X", "Y"), ("U1", "W"), ("U1", "X"), ("U2", "W"), ("U2", "Y")])
    rec = {"W", "Z", "X", "Y"}
    assert backdoor_set(g, "X", "Y", rec) is None
    assert identifiable(g, "X", "Y", rec)


def test_instrument_graph_is_not_identifiable():
    g = Graph("ZXYU", [("Z", "X"), ("X", "Y"), ("U", "X"), ("U", "Y")])
    assert not identifiable(g, "X", "Y", {"Z", "X", "Y"})


def test_m_bias_collider_is_not_adjusted_for():
    g = Graph(["X", "Y", "C", "A", "B"], [("X", "Y"), ("A", "X"), ("A", "C"), ("B", "C"), ("B", "Y")])
    assert backdoor_set(g, "X", "Y", {"X", "Y", "C"}) == set()


# ---- estimates against the simulator's ground truth ----

@pytest.mark.parametrize("action,outcome", [("discount", "retained"), ("discount", "refund"),
                                            ("free_shipping", "retained")])
def test_identifiable_effects_cover_truth_across_seeds(action, outcome):
    truth = world.true_effect(action, outcome, "B")
    hits = 0
    for seed in range(20):
        r = engine(seed=seed).what_if({action: 1}, [outcome], {"segment": "B"})["results"][0]
        assert r["certificate"]["status"] == "identifiable"
        hits += r["interval"][0] <= truth <= r["interval"][1]
    assert hits >= 16  # nominal 95%


def test_naive_contrast_has_the_wrong_sign_for_discount():
    r = engine().what_if({"discount": 1}, ["retained"], {"segment": "B"})["results"][0]
    assert r["explain"]["naive_log_contrast"] < 0 < r["value"]


def test_unrecorded_confounder_blocks_the_answer_and_says_what_is_missing():
    r = engine().what_if({"support_call": 1}, ["retained"], {"segment": "B"})["results"][0]
    cert = r["certificate"]
    assert cert["status"] == "not" and r["value"] is None
    assert {"action": "log", "variables": ["frustration"]} in cert["missing"]
    assert any(m["action"] == "experiment" for m in cert["missing"])
    lo, hi = r["interval"]
    assert lo <= world.true_effect("support_call", "retained", "B") <= hi


def test_logging_the_missing_variable_makes_it_identifiable():
    r = engine(record_frustration=True).what_if({"support_call": 1}, ["retained"], {"segment": "B"})["results"][0]
    assert r["certificate"]["adjustment_set"] == ["frustration"]
    assert r["interval"][0] <= world.true_effect("support_call", "retained", "B") <= r["interval"][1]


def test_where_on_a_post_treatment_variable_is_rejected():
    with pytest.raises(ValueError):
        engine().what_if({"discount": 1}, ["refund"], {"retained": 1})


# ---- the wire: MCP handshake, tools, and the action gate ----

def test_mcp_roundtrip_and_gate():
    mcp = MCPClient()
    try:
        assert mcp.info["serverInfo"]["name"] == "gowhy-demo"
        names = [t["name"] for t in mcp.request("tools/list")["tools"]]
        assert names == ["what_if", "identify", "explain", "apply_offer"]

        _, _, err = mcp.call("apply_offer", segment="B", offer="discount", query_id="q-9999")
        assert err

        _, q, err = mcp.call("what_if", set={"free_shipping": 1}, outcomes=["retained", "refund"],
                             where={"segment": "B"})
        assert not err
        _, _, err = mcp.call("apply_offer", segment="A", offer="free_shipping", query_id=q["query_id"])
        assert err  # asked about segment B, not A
        text, _, err = mcp.call("apply_offer", segment="B", offer="free_shipping", query_id=q["query_id"])
        assert not err and "authorized by" in text

        _, q, _ = mcp.call("what_if", set={"support_call": 1}, outcomes=["retained"], where={"segment": "B"})
        text, _, err = mcp.call("apply_offer", segment="B", offer="support_call", query_id=q["query_id"])
        assert err and "REFUSED" in text

        text, _, _ = mcp.call("explain", query_id="q-0001")
        assert "apply_offer(segment=B, offer=free_shipping)" in text
    finally:
        mcp.close()
