"""Composable workflow patterns."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional


class Sequential:
    """Executes a list of callables in order, threading state through.

    Args:
        steps: Ordered callables, each accepting and returning a state dict.
    """

    def __init__(self, steps: list[Callable[[dict[str, Any]], dict[str, Any]]]) -> None:
        self.steps = list(steps)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run all steps sequentially."""
        for step in self.steps:
            result = step(state)
            if result is not None:
                state = result
        return state

    async def arun(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run all steps sequentially, awaiting coroutines."""
        for step in self.steps:
            if asyncio.iscoroutinefunction(step):
                result = await step(state)
            else:
                result = step(state)
            if result is not None:
                state = result
        return state


class Parallel:
    """Executes callables concurrently, merging results into state.

    Each callable receives a *copy* of the input state so mutations are
    isolated.  After all complete, results are merged left-to-right into the
    original state dict.

    Args:
        steps: Callables to execute concurrently.
        merge: Optional custom merge function.  Receives the original state
            and a list of result dicts, and should return the merged state.
    """

    def __init__(
        self,
        steps: list[Callable[[dict[str, Any]], dict[str, Any]]],
        merge: Optional[
            Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any]]
        ] = None,
    ) -> None:
        self.steps = list(steps)
        self.merge = merge

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run steps and merge results (sync, sequentially)."""
        results = []
        for step in self.steps:
            result = step(dict(state))
            results.append(result if result is not None else dict(state))
        return self._merge(state, results)

    async def arun(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run steps concurrently with ``asyncio.gather``."""

        async def _run_one(
            step: Callable[[dict[str, Any]], dict[str, Any]],
        ) -> dict[str, Any]:
            copy = dict(state)
            if asyncio.iscoroutinefunction(step):
                result = await step(copy)
            else:
                result = step(copy)
            return result if result is not None else copy

        results = await asyncio.gather(*[_run_one(s) for s in self.steps])
        return self._merge(state, list(results))

    def _merge(
        self,
        original: dict[str, Any],
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.merge is not None:
            return self.merge(original, results)
        merged = dict(original)
        for result in results:
            merged.update(result)
        return merged


class Router:
    """Routes input to one of several handlers based on a routing function.

    Args:
        route_fn: Callable that receives state and returns a string key.
        routes: Mapping from route key to handler callable.
        default: Optional fallback handler when no route matches.
    """

    def __init__(
        self,
        route_fn: Callable[[dict[str, Any]], str],
        routes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
        default: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ) -> None:
        self.route_fn = route_fn
        self.routes = dict(routes)
        self.default = default

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Route and execute the selected handler."""
        key = self.route_fn(state)
        handler = self.routes.get(key, self.default)
        if handler is None:
            raise ValueError(f"No route matched for key {key!r}")
        result = handler(state)
        return result if result is not None else state

    async def arun(self, state: dict[str, Any]) -> dict[str, Any]:
        """Route and execute the selected handler asynchronously."""
        if asyncio.iscoroutinefunction(self.route_fn):
            key = await self.route_fn(state)
        else:
            key = self.route_fn(state)
        handler = self.routes.get(key, self.default)
        if handler is None:
            raise ValueError(f"No route matched for key {key!r}")
        if asyncio.iscoroutinefunction(handler):
            result = await handler(state)
        else:
            result = handler(state)
        return result if result is not None else state


class Loop:
    """Repeats execution until a condition is met.

    Args:
        body: Callable executed each iteration.
        condition: Callable that returns ``True`` when the loop should stop.
        max_iterations: Safety limit to prevent infinite loops.
    """

    def __init__(
        self,
        body: Callable[[dict[str, Any]], dict[str, Any]],
        condition: Callable[[dict[str, Any]], bool],
        max_iterations: int = 100,
    ) -> None:
        self.body = body
        self.condition = condition
        self.max_iterations = max_iterations

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the loop synchronously."""
        for _ in range(self.max_iterations):
            result = self.body(state)
            if result is not None:
                state = result
            if self.condition(state):
                return state
        return state

    async def arun(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the loop asynchronously."""
        for _ in range(self.max_iterations):
            if asyncio.iscoroutinefunction(self.body):
                result = await self.body(state)
            else:
                result = self.body(state)
            if result is not None:
                state = result
            if asyncio.iscoroutinefunction(self.condition):
                done = await self.condition(state)
            else:
                done = self.condition(state)
            if done:
                return state
        return state


class Pipeline:
    """Chain of transform functions applied sequentially.

    Unlike ``Sequential``, each step receives and returns just the state
    value, making it ideal for pure data transformations.

    Args:
        transforms: Ordered list of transform callables.
    """

    def __init__(
        self, transforms: list[Callable[[dict[str, Any]], dict[str, Any]]]
    ) -> None:
        self.transforms = list(transforms)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Apply all transforms sequentially."""
        for transform in self.transforms:
            result = transform(state)
            if result is not None:
                state = result
        return state

    async def arun(self, state: dict[str, Any]) -> dict[str, Any]:
        """Apply all transforms, awaiting coroutines."""
        for transform in self.transforms:
            if asyncio.iscoroutinefunction(transform):
                result = await transform(state)
            else:
                result = transform(state)
            if result is not None:
                state = result
        return state
