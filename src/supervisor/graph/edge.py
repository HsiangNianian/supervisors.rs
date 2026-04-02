"""Edge types for the graph orchestration engine."""

from __future__ import annotations

from typing import Any, Callable, Optional


class Edge:
    """A directed edge between two nodes.

    Args:
        source: Name of the source node.
        target: Name of the target node.
    """

    def __init__(self, source: str, target: str) -> None:
        self.source = source
        self.target = target

    def should_follow(self, state: dict[str, Any]) -> bool:
        """Return whether this edge should be followed.

        The base edge is unconditional and always returns ``True``.
        """
        return True

    def __repr__(self) -> str:
        return f"Edge({self.source!r} -> {self.target!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return self.source == other.source and self.target == other.target

    def __hash__(self) -> int:
        return hash((self.source, self.target))


class ConditionalEdge(Edge):
    """An edge that is only followed when a condition is met.

    Args:
        source: Name of the source node.
        target: Name of the target node.
        condition: Callable receiving state and returning ``True`` if the
            edge should be followed.
    """

    def __init__(
        self,
        source: str,
        target: str,
        condition: Callable[[dict[str, Any]], bool],
    ) -> None:
        super().__init__(source, target)
        self.condition = condition

    def should_follow(self, state: dict[str, Any]) -> bool:
        """Evaluate the condition against current state."""
        return self.condition(state)

    def __repr__(self) -> str:
        return f"ConditionalEdge({self.source!r} -> {self.target!r})"
