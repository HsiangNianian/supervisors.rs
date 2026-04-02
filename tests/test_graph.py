"""Tests for the graph orchestration engine."""

from __future__ import annotations

import asyncio

import pytest

from supervisor.graph import (
    AgentNode,
    ConditionalEdge,
    Edge,
    FunctionNode,
    Graph,
    GraphBuilder,
    Node,
    RouterNode,
    SubGraphNode,
)
from supervisor.graph.engine import GraphError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubAgent:
    """Minimal agent-like object for testing AgentNode."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.received: list[object] = []

    def handle_message(self, msg: object) -> None:
        self.received.append(msg)


class IncrementNode(Node):
    def execute(self, state: dict) -> dict:
        state["count"] = state.get("count", 0) + 1
        return state


# ---------------------------------------------------------------------------
# Node tests
# ---------------------------------------------------------------------------

class TestFunctionNode:
    def test_execute(self):
        fn = FunctionNode("inc", lambda s: {**s, "x": s.get("x", 0) + 1})
        result = fn.execute({"x": 5})
        assert result["x"] == 6

    def test_execute_none_returns_original(self):
        fn = FunctionNode("noop", lambda s: None)
        state = {"a": 1}
        assert fn.execute(state) is state

    def test_async_execute(self):
        async def async_fn(s):
            return {**s, "async": True}

        fn = FunctionNode("af", async_fn)
        result = asyncio.get_event_loop().run_until_complete(fn.aexecute({}))
        assert result["async"] is True


class TestAgentNode:
    def test_execute(self):
        agent = _StubAgent("echo")
        node = AgentNode("agent_node", agent)
        state = node.execute({"message": "hello", "sender": "user"})
        assert state["last_agent"] == "echo"
        assert len(agent.received) == 1


class TestRouterNode:
    def test_routes_based_on_state(self):
        router = RouterNode("r", lambda s: "a" if s.get("go_a") else "b")
        state = router.execute({"go_a": True})
        assert state["_router_next"] == "a"

        state2 = router.execute({"go_a": False})
        assert state2["_router_next"] == "b"


class TestSubGraphNode:
    def test_delegates_to_subgraph(self):
        sub = (
            GraphBuilder()
            .node("s1", lambda s: {**s, "sub": True})
            .edge("s1", "s2")
            .node("s2", lambda s: {**s, "sub2": True})
            .entry("s1")
            .exit("s2")
            .build()
        )
        node = SubGraphNode("sg", sub)
        result = node.execute({})
        assert result["sub"] is True
        assert result["sub2"] is True


class TestCustomNode:
    def test_abstract_execute(self):
        node = IncrementNode("inc")
        result = node.execute({"count": 0})
        assert result["count"] == 1

    def test_repr(self):
        node = IncrementNode("inc")
        assert "IncrementNode" in repr(node)
        assert "inc" in repr(node)


# ---------------------------------------------------------------------------
# Edge tests
# ---------------------------------------------------------------------------

class TestEdge:
    def test_should_follow_always(self):
        edge = Edge("a", "b")
        assert edge.should_follow({}) is True

    def test_repr(self):
        edge = Edge("a", "b")
        assert "a" in repr(edge) and "b" in repr(edge)

    def test_equality(self):
        assert Edge("a", "b") == Edge("a", "b")
        assert Edge("a", "b") != Edge("a", "c")

    def test_hash(self):
        assert hash(Edge("a", "b")) == hash(Edge("a", "b"))


class TestConditionalEdge:
    def test_condition_true(self):
        edge = ConditionalEdge("a", "b", lambda s: s.get("go", False))
        assert edge.should_follow({"go": True}) is True

    def test_condition_false(self):
        edge = ConditionalEdge("a", "b", lambda s: s.get("go", False))
        assert edge.should_follow({"go": False}) is False

    def test_repr(self):
        edge = ConditionalEdge("a", "b", lambda s: True)
        assert "Conditional" in repr(edge)


# ---------------------------------------------------------------------------
# Graph engine tests
# ---------------------------------------------------------------------------

class TestGraphValidation:
    def test_no_entry(self):
        g = Graph()
        g.add_node("a", FunctionNode("a", lambda s: s))
        g.set_exit("a")
        with pytest.raises(GraphError, match="No entry"):
            g.validate()

    def test_no_exit(self):
        g = Graph()
        g.add_node("a", FunctionNode("a", lambda s: s))
        g.set_entry("a")
        with pytest.raises(GraphError, match="No exit"):
            g.validate()

    def test_missing_entry_node(self):
        g = Graph()
        g.add_node("a", FunctionNode("a", lambda s: s))
        g.set_entry("missing")
        g.set_exit("a")
        with pytest.raises(GraphError, match="does not exist"):
            g.validate()

    def test_missing_exit_node(self):
        g = Graph()
        g.add_node("a", FunctionNode("a", lambda s: s))
        g.set_entry("a")
        g.set_exit("missing")
        with pytest.raises(GraphError, match="does not exist"):
            g.validate()

    def test_edge_references_missing_node(self):
        g = Graph()
        g.add_node("a", FunctionNode("a", lambda s: s))
        g.set_entry("a")
        g.set_exit("a")
        g.add_edge("a", "ghost")
        with pytest.raises(GraphError, match="does not exist"):
            g.validate()

    def test_unreachable_exit(self):
        g = Graph()
        g.add_node("a", FunctionNode("a", lambda s: s))
        g.add_node("b", FunctionNode("b", lambda s: s))
        g.set_entry("a")
        g.set_exit("b")
        # no edge from a to b
        with pytest.raises(GraphError, match="reachable"):
            g.validate()

    def test_duplicate_node(self):
        g = Graph()
        g.add_node("a", FunctionNode("a", lambda s: s))
        with pytest.raises(GraphError, match="already exists"):
            g.add_node("a", FunctionNode("a2", lambda s: s))

    def test_valid_graph(self):
        g = Graph()
        g.add_node("a", FunctionNode("a", lambda s: s))
        g.add_node("b", FunctionNode("b", lambda s: s))
        g.add_edge("a", "b")
        g.set_entry("a")
        g.set_exit("b")
        g.validate()  # should not raise


class TestGraphExecution:
    def test_simple_chain(self):
        g = (
            GraphBuilder()
            .node("a", lambda s: {**s, "a": True})
            .node("b", lambda s: {**s, "b": True})
            .edge("a", "b")
            .entry("a")
            .exit("b")
            .build()
        )
        result = g.run({})
        assert result == {"a": True, "b": True}

    def test_conditional_edge(self):
        g = (
            GraphBuilder()
            .node("start", lambda s: s)
            .node("yes", lambda s: {**s, "path": "yes"})
            .node("no", lambda s: {**s, "path": "no"})
            .edge("start", "yes", condition=lambda s: s.get("flag"))
            .edge("start", "no")
            .entry("start")
            .exit("yes")
            .exit("no")
            .build()
        )
        assert g.run({"flag": True})["path"] == "yes"
        assert g.run({"flag": False})["path"] == "no"

    def test_router_node(self):
        router = RouterNode("r", lambda s: s.get("dest", "fallback"))
        g = (
            GraphBuilder()
            .node("r", router)
            .node("x", lambda s: {**s, "result": "x"})
            .node("y", lambda s: {**s, "result": "y"})
            .edge("r", "x")
            .edge("r", "y")
            .entry("r")
            .exit("x")
            .exit("y")
            .build()
        )
        assert g.run({"dest": "x"})["result"] == "x"
        assert g.run({"dest": "y"})["result"] == "y"

    def test_async_run(self):
        g = (
            GraphBuilder()
            .node("a", lambda s: {**s, "v": 1})
            .node("b", lambda s: {**s, "v": s["v"] + 1})
            .edge("a", "b")
            .entry("a")
            .exit("b")
            .build()
        )
        result = asyncio.get_event_loop().run_until_complete(g.arun({}))
        assert result["v"] == 2

    def test_single_node_graph(self):
        g = (
            GraphBuilder()
            .node("only", lambda s: {**s, "done": True})
            .entry("only")
            .exit("only")
            .build()
        )
        assert g.run({})["done"] is True

    def test_no_outgoing_edge_error(self):
        g = Graph()
        g.add_node("a", FunctionNode("a", lambda s: s))
        g.add_node("b", FunctionNode("b", lambda s: s))
        g.add_edge("a", "b")
        g.add_node("c", FunctionNode("c", lambda s: s))
        g.add_edge("b", "c")
        g.set_entry("a")
        g.set_exit("c")
        # This should work fine
        g.run({})


class TestGraphMermaid:
    def test_mermaid_output(self):
        g = (
            GraphBuilder()
            .node("start", lambda s: s)
            .node("end", lambda s: s)
            .edge("start", "end")
            .entry("start")
            .exit("end")
            .build()
        )
        mermaid = g.to_mermaid()
        assert "graph TD" in mermaid
        assert "start" in mermaid
        assert "end" in mermaid
        assert "-->" in mermaid

    def test_mermaid_conditional_edge(self):
        g = (
            GraphBuilder()
            .node("a", lambda s: s)
            .node("b", lambda s: s)
            .edge("a", "b", condition=lambda s: True)
            .entry("a")
            .exit("b")
            .build()
        )
        mermaid = g.to_mermaid()
        assert "-.->" in mermaid


class TestGraphIntrospection:
    def test_properties(self):
        g = (
            GraphBuilder()
            .node("a", lambda s: s)
            .node("b", lambda s: s)
            .edge("a", "b")
            .entry("a")
            .exit("b")
            .build()
        )
        assert set(g.nodes.keys()) == {"a", "b"}
        assert len(g.edges) == 1
        assert g.entry == "a"
        assert g.exits == {"b"}

    def test_repr(self):
        g = (
            GraphBuilder()
            .node("a", lambda s: s)
            .entry("a")
            .exit("a")
            .build()
        )
        r = repr(g)
        assert "Graph" in r
        assert "a" in r


# ---------------------------------------------------------------------------
# Builder tests
# ---------------------------------------------------------------------------

class TestGraphBuilder:
    def test_fluent_api(self):
        builder = GraphBuilder()
        result = builder.node("a", lambda s: s).edge("a", "b").node("b", lambda s: s)
        assert result is builder

    def test_build_validates(self):
        with pytest.raises(GraphError):
            GraphBuilder().build()

    def test_accepts_node_instance(self):
        g = (
            GraphBuilder()
            .node("a", IncrementNode("a"))
            .entry("a")
            .exit("a")
            .build()
        )
        assert g.run({"count": 0})["count"] == 1

    def test_rejects_non_callable(self):
        with pytest.raises(GraphError, match="callable"):
            GraphBuilder().node("bad", 42)
