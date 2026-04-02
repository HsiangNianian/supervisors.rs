"""Tests for workflow patterns."""

from __future__ import annotations

import asyncio

import pytest

from supervisor.patterns import Loop, Parallel, Pipeline, Router, Sequential


class TestSequential:
    def test_run(self):
        seq = Sequential([
            lambda s: {**s, "a": 1},
            lambda s: {**s, "b": s["a"] + 1},
        ])
        assert seq.run({}) == {"a": 1, "b": 2}

    def test_none_passthrough(self):
        seq = Sequential([lambda s: None, lambda s: {**s, "x": 1}])
        assert seq.run({})["x"] == 1

    def test_async_run(self):
        async def step(s):
            return {**s, "async": True}

        seq = Sequential([lambda s: {**s, "sync": True}, step])
        result = asyncio.get_event_loop().run_until_complete(seq.arun({}))
        assert result["sync"] is True
        assert result["async"] is True

    def test_empty(self):
        assert Sequential([]).run({"x": 1}) == {"x": 1}


class TestParallel:
    def test_run_merges(self):
        par = Parallel([
            lambda s: {**s, "a": 1},
            lambda s: {**s, "b": 2},
        ])
        result = par.run({})
        assert result["a"] == 1
        assert result["b"] == 2

    def test_custom_merge(self):
        def merge(orig, results):
            return {"total": sum(r.get("v", 0) for r in results)}

        par = Parallel(
            [lambda s: {"v": 1}, lambda s: {"v": 2}],
            merge=merge,
        )
        assert par.run({}) == {"total": 3}

    def test_async_run(self):
        async def slow(s):
            return {**s, "slow": True}

        par = Parallel([slow, lambda s: {**s, "fast": True}])
        result = asyncio.get_event_loop().run_until_complete(par.arun({}))
        assert result["slow"] is True
        assert result["fast"] is True

    def test_isolation(self):
        """Each step gets a copy so mutations don't interfere."""
        def mutate_a(s):
            s["shared"] = "a"
            return s

        def mutate_b(s):
            s["shared"] = "b"
            return s

        par = Parallel([mutate_a, mutate_b])
        result = par.run({"shared": "orig"})
        # Last merge wins
        assert result["shared"] in ("a", "b")


class TestRouter:
    def test_routes(self):
        router = Router(
            route_fn=lambda s: s["kind"],
            routes={
                "greet": lambda s: {**s, "out": "hello"},
                "bye": lambda s: {**s, "out": "goodbye"},
            },
        )
        assert router.run({"kind": "greet"})["out"] == "hello"
        assert router.run({"kind": "bye"})["out"] == "goodbye"

    def test_default(self):
        router = Router(
            route_fn=lambda s: "unknown",
            routes={},
            default=lambda s: {**s, "out": "default"},
        )
        assert router.run({})["out"] == "default"

    def test_no_match_raises(self):
        router = Router(route_fn=lambda s: "nope", routes={})
        with pytest.raises(ValueError, match="No route"):
            router.run({})

    def test_async_run(self):
        async def async_handler(s):
            return {**s, "async_route": True}

        router = Router(
            route_fn=lambda s: "a",
            routes={"a": async_handler},
        )
        result = asyncio.get_event_loop().run_until_complete(router.arun({}))
        assert result["async_route"] is True


class TestLoop:
    def test_basic_loop(self):
        loop = Loop(
            body=lambda s: {**s, "i": s.get("i", 0) + 1},
            condition=lambda s: s.get("i", 0) >= 5,
        )
        result = loop.run({})
        assert result["i"] == 5

    def test_max_iterations(self):
        loop = Loop(
            body=lambda s: s,
            condition=lambda s: False,
            max_iterations=10,
        )
        loop.run({})  # should not hang

    def test_async_loop(self):
        async def body(s):
            return {**s, "i": s.get("i", 0) + 1}

        loop = Loop(body=body, condition=lambda s: s.get("i", 0) >= 3)
        result = asyncio.get_event_loop().run_until_complete(loop.arun({}))
        assert result["i"] == 3

    def test_immediate_stop(self):
        loop = Loop(
            body=lambda s: s,
            condition=lambda s: True,
        )
        result = loop.run({"x": 1})
        assert result["x"] == 1


class TestPipeline:
    def test_chain(self):
        pipe = Pipeline([
            lambda s: {**s, "x": 1},
            lambda s: {**s, "y": s["x"] * 2},
            lambda s: {**s, "z": s["y"] + 1},
        ])
        result = pipe.run({})
        assert result == {"x": 1, "y": 2, "z": 3}

    def test_empty(self):
        assert Pipeline([]).run({"a": 1}) == {"a": 1}

    def test_async_pipeline(self):
        async def double(s):
            return {**s, "v": s.get("v", 1) * 2}

        pipe = Pipeline([double, double])
        result = asyncio.get_event_loop().run_until_complete(pipe.arun({"v": 3}))
        assert result["v"] == 12


class TestComposability:
    def test_nested_sequential_in_pipeline(self):
        inner = Sequential([
            lambda s: {**s, "inner": True},
        ])
        pipe = Pipeline([inner.run, lambda s: {**s, "outer": True}])
        result = pipe.run({})
        assert result["inner"] is True
        assert result["outer"] is True

    def test_loop_inside_sequential(self):
        loop = Loop(
            body=lambda s: {**s, "n": s.get("n", 0) + 1},
            condition=lambda s: s.get("n", 0) >= 3,
        )
        seq = Sequential([loop.run, lambda s: {**s, "done": True}])
        result = seq.run({})
        assert result["n"] == 3
        assert result["done"] is True
