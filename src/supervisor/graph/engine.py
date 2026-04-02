"""Graph execution engine."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from .edge import ConditionalEdge, Edge
from .node import Node, RouterNode


class GraphError(Exception):
    """Raised when the graph is invalid or execution fails."""


class Graph:
    """A directed graph of nodes and edges that executes as a workflow.

    Nodes are processing steps; edges define transitions.  The graph supports
    conditional routing, sub-graphs, and async execution.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._entry: Optional[str] = None
        self._exits: set[str] = set()

    # -- mutators ----------------------------------------------------------

    def add_node(self, name: str, node: Node) -> None:
        """Add a node to the graph.

        Args:
            name: Unique name for the node.
            node: A ``Node`` instance.

        Raises:
            GraphError: If the name is already taken.
        """
        if name in self._nodes:
            raise GraphError(f"Node {name!r} already exists")
        node.name = name
        self._nodes[name] = node

    def add_edge(
        self,
        source: str,
        target: str,
        condition: Optional[Callable[[dict[str, Any]], bool]] = None,
    ) -> None:
        """Add an edge between two nodes.

        Args:
            source: Source node name.
            target: Target node name.
            condition: Optional predicate; creates a ``ConditionalEdge`` when
                provided.
        """
        if condition is not None:
            self._edges.append(ConditionalEdge(source, target, condition))
        else:
            self._edges.append(Edge(source, target))

    def set_entry(self, name: str) -> None:
        """Set the entry point of the graph.

        Args:
            name: Name of the entry node.
        """
        self._entry = name

    def set_exit(self, name: str) -> None:
        """Mark a node as an exit point.

        Multiple exit nodes are allowed.

        Args:
            name: Name of the exit node.
        """
        self._exits.add(name)

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Validate the graph structure.

        Checks:
        - An entry node is set and exists.
        - All exit nodes exist.
        - All edges reference existing nodes.
        - The entry node is reachable (trivially true) and exit nodes are
          reachable from the entry via some path.

        Raises:
            GraphError: If any check fails.
        """
        if self._entry is None:
            raise GraphError("No entry node set")
        if self._entry not in self._nodes:
            raise GraphError(f"Entry node {self._entry!r} does not exist")
        if not self._exits:
            raise GraphError("No exit node set")

        for name in self._exits:
            if name not in self._nodes:
                raise GraphError(f"Exit node {name!r} does not exist")

        for edge in self._edges:
            if edge.source not in self._nodes:
                raise GraphError(
                    f"Edge source {edge.source!r} does not exist"
                )
            if edge.target not in self._nodes:
                raise GraphError(
                    f"Edge target {edge.target!r} does not exist"
                )

        # Reachability check from entry to at least one exit
        reachable = self._reachable_from(self._entry)
        if not reachable & self._exits:
            raise GraphError(
                "No exit node is reachable from the entry node"
            )

    def _reachable_from(self, start: str) -> set[str]:
        """BFS to find all nodes reachable from *start*."""
        visited: set[str] = set()
        queue = [start]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for edge in self._edges:
                if edge.source == current and edge.target not in visited:
                    queue.append(edge.target)
        return visited

    # -- execution ---------------------------------------------------------

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the graph synchronously.

        Args:
            state: Initial workflow state.

        Returns:
            Final state after traversal.

        Raises:
            GraphError: If the graph is invalid or execution gets stuck.
        """
        self.validate()
        current = self._entry
        assert current is not None

        visited_count: dict[str, int] = {}
        max_visits = len(self._nodes) * 100

        while True:
            node = self._nodes[current]
            visited_count[current] = visited_count.get(current, 0) + 1

            if sum(visited_count.values()) > max_visits:
                raise GraphError("Possible infinite loop detected")

            state = node.execute(state)

            if current in self._exits:
                return state

            current = self._resolve_next(current, state)

    async def arun(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the graph asynchronously.

        Args:
            state: Initial workflow state.

        Returns:
            Final state after traversal.

        Raises:
            GraphError: If the graph is invalid or execution gets stuck.
        """
        self.validate()
        current = self._entry
        assert current is not None

        visited_count: dict[str, int] = {}
        max_visits = len(self._nodes) * 100

        while True:
            node = self._nodes[current]
            visited_count[current] = visited_count.get(current, 0) + 1

            if sum(visited_count.values()) > max_visits:
                raise GraphError("Possible infinite loop detected")

            state = await node.aexecute(state)

            if current in self._exits:
                return state

            current = self._resolve_next(current, state)

    def _resolve_next(self, current: str, state: dict[str, Any]) -> str:
        """Determine the next node to visit.

        For ``RouterNode`` instances the decision stored in
        ``state["_router_next"]`` takes precedence.  Otherwise edges are
        evaluated in order, with conditional edges checked first.
        """
        node = self._nodes[current]

        # RouterNode stores its decision in state
        if isinstance(node, RouterNode):
            router_next = state.pop("_router_next", None)
            if router_next is not None:
                if router_next not in self._nodes:
                    raise GraphError(
                        f"Router chose non-existent node {router_next!r}"
                    )
                return router_next

        outgoing = [e for e in self._edges if e.source == current]
        if not outgoing:
            raise GraphError(
                f"Node {current!r} has no outgoing edges and is not an exit"
            )

        # Evaluate conditional edges first, then unconditional
        for edge in outgoing:
            if isinstance(edge, ConditionalEdge) and edge.should_follow(state):
                return edge.target

        for edge in outgoing:
            if not isinstance(edge, ConditionalEdge):
                return edge.target

        raise GraphError(
            f"No edge from {current!r} matched the current state"
        )

    # -- visualisation -----------------------------------------------------

    def to_mermaid(self) -> str:
        """Export the graph as a Mermaid diagram string.

        Returns:
            A Mermaid ``graph TD`` string.
        """
        lines = ["graph TD"]

        for name in self._nodes:
            label = name
            if name == self._entry:
                label = f"{name}([{name}])"
            elif name in self._exits:
                label = f"{name}[/{name}/]"
            else:
                label = f"{name}[{name}]"
            lines.append(f"    {label}")

        for edge in self._edges:
            if isinstance(edge, ConditionalEdge):
                lines.append(f"    {edge.source} -.-> {edge.target}")
            else:
                lines.append(f"    {edge.source} --> {edge.target}")

        return "\n".join(lines)

    # -- introspection -----------------------------------------------------

    @property
    def nodes(self) -> dict[str, Node]:
        """Return a copy of the node mapping."""
        return dict(self._nodes)

    @property
    def edges(self) -> list[Edge]:
        """Return a copy of the edge list."""
        return list(self._edges)

    @property
    def entry(self) -> Optional[str]:
        """Return the entry node name."""
        return self._entry

    @property
    def exits(self) -> set[str]:
        """Return the set of exit node names."""
        return set(self._exits)

    def __repr__(self) -> str:
        return (
            f"Graph(nodes={list(self._nodes.keys())}, "
            f"entry={self._entry!r}, exits={self._exits})"
        )
