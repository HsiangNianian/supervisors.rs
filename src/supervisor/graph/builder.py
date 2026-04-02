"""Fluent graph builder API."""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

from .edge import ConditionalEdge, Edge
from .engine import Graph, GraphError
from .node import FunctionNode, Node


class GraphBuilder:
    """Fluent API for constructing a ``Graph``.

    Example::

        graph = (
            GraphBuilder()
            .node("start", lambda s: {**s, "started": True})
            .node("end", lambda s: {**s, "ended": True})
            .edge("start", "end")
            .entry("start")
            .exit("end")
            .build()
        )
    """

    def __init__(self) -> None:
        self._graph = Graph()

    def node(
        self,
        name: str,
        node_or_callable: Union[Node, Callable[[dict[str, Any]], dict[str, Any]]],
    ) -> GraphBuilder:
        """Add a node to the graph.

        If *node_or_callable* is a plain callable it is wrapped in a
        ``FunctionNode``.

        Args:
            name: Unique node name.
            node_or_callable: A ``Node`` instance or a ``(state) -> state``
                callable.

        Returns:
            This builder instance for chaining.
        """
        if isinstance(node_or_callable, Node):
            self._graph.add_node(name, node_or_callable)
        elif callable(node_or_callable):
            self._graph.add_node(name, FunctionNode(name, node_or_callable))
        else:
            raise GraphError(
                f"Expected a Node or callable, got {type(node_or_callable)!r}"
            )
        return self

    def edge(
        self,
        source: str,
        target: str,
        condition: Optional[Callable[[dict[str, Any]], bool]] = None,
    ) -> GraphBuilder:
        """Add an edge between two nodes.

        Args:
            source: Source node name.
            target: Target node name.
            condition: Optional predicate for conditional routing.

        Returns:
            This builder instance for chaining.
        """
        self._graph.add_edge(source, target, condition)
        return self

    def entry(self, name: str) -> GraphBuilder:
        """Set the entry node.

        Args:
            name: Name of the entry node.

        Returns:
            This builder instance for chaining.
        """
        self._graph.set_entry(name)
        return self

    def exit(self, name: str) -> GraphBuilder:
        """Mark a node as an exit.

        Args:
            name: Name of the exit node.

        Returns:
            This builder instance for chaining.
        """
        self._graph.set_exit(name)
        return self

    def build(self) -> Graph:
        """Validate and return the constructed graph.

        Returns:
            The built ``Graph``.

        Raises:
            GraphError: If the graph fails validation.
        """
        self._graph.validate()
        return self._graph
