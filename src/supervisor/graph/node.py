"""Node types for the graph orchestration engine."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from .engine import Graph


class Node(ABC):
    """Base class for all graph nodes.

    Args:
        name: Unique name identifying this node within a graph.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute this node synchronously.

        Args:
            state: Current workflow state dictionary.

        Returns:
            Updated state dictionary.
        """

    async def aexecute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute this node asynchronously.

        Default implementation delegates to the synchronous ``execute``.

        Args:
            state: Current workflow state dictionary.

        Returns:
            Updated state dictionary.
        """
        return self.execute(state)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class AgentNode(Node):
    """Wraps an Agent, calling ``handle_message`` on execution.

    The node reads a ``message`` key from state, passes it to the agent, and
    stores the agent's name in ``state["last_agent"]``.

    Args:
        name: Node name.
        agent: An ``Agent`` instance with a ``handle_message`` method.
    """

    def __init__(self, name: str, agent: Any) -> None:
        super().__init__(name)
        self.agent = agent

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the wrapped agent."""
        from .._core import Message

        content = state.get("message", "")
        sender = state.get("sender", "user")
        msg = Message(sender, self.agent.name, str(content))
        self.agent.handle_message(msg)
        state["last_agent"] = self.agent.name
        return state

    async def aexecute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the wrapped agent asynchronously."""
        from .._core import Message

        content = state.get("message", "")
        sender = state.get("sender", "user")
        msg = Message(sender, self.agent.name, str(content))

        if asyncio.iscoroutinefunction(getattr(self.agent, "handle_message", None)):
            await self.agent.handle_message(msg)
        else:
            self.agent.handle_message(msg)

        state["last_agent"] = self.agent.name
        return state


class FunctionNode(Node):
    """Wraps a callable ``(state) -> state``.

    Args:
        name: Node name.
        func: A callable accepting and returning a state dict.
    """

    def __init__(self, name: str, func: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        super().__init__(name)
        self.func = func

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the wrapped function."""
        result = self.func(state)
        if result is None:
            return state
        return result

    async def aexecute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the wrapped function, awaiting if it's a coroutine."""
        if asyncio.iscoroutinefunction(self.func):
            result = await self.func(state)
        else:
            result = self.func(state)
        if result is None:
            return state
        return result


class RouterNode(Node):
    """Routes to different next nodes based on state.

    The routing function inspects state and returns the name of the next node
    to transition to.  The router itself does not modify state; it records the
    routing decision in ``state["_router_next"]`` so the engine can follow the
    correct edge.

    Args:
        name: Node name.
        route_fn: Callable that receives state and returns a node name string.
    """

    def __init__(
        self,
        name: str,
        route_fn: Callable[[dict[str, Any]], str],
    ) -> None:
        super().__init__(name)
        self.route_fn = route_fn

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Determine the next node and record it in state."""
        next_node = self.route_fn(state)
        state["_router_next"] = next_node
        return state

    async def aexecute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Determine the next node asynchronously."""
        if asyncio.iscoroutinefunction(self.route_fn):
            next_node = await self.route_fn(state)
        else:
            next_node = self.route_fn(state)
        state["_router_next"] = next_node
        return state


class SubGraphNode(Node):
    """Embeds another ``Graph`` as a single node.

    Args:
        name: Node name.
        graph: A ``Graph`` instance to execute as a sub-workflow.
    """

    def __init__(self, name: str, graph: Graph) -> None:
        super().__init__(name)
        self.graph = graph

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the embedded sub-graph synchronously."""
        return self.graph.run(state)

    async def aexecute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the embedded sub-graph asynchronously."""
        return await self.graph.arun(state)
