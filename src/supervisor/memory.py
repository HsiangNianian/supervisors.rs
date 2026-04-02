"""Memory systems for conversation history management."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Optional


class BaseMemory(ABC):
    """Abstract base class for memory implementations.

    All memory types store role/content message pairs and can format them
    for inclusion in LLM prompts.
    """

    @abstractmethod
    def add(self, role: str, content: str) -> None:
        """Add a message to memory.

        Args:
            role: Message role (e.g. ``"user"``, ``"assistant"``).
            content: Message content.
        """

    @abstractmethod
    def get_messages(self) -> list[dict[str, str]]:
        """Return all stored messages.

        Returns:
            List of dicts with ``role`` and ``content`` keys.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all messages from memory."""

    def to_prompt_string(self) -> str:
        """Format stored messages as a plain-text prompt string.

        Returns:
            Newline-separated ``role: content`` pairs.
        """
        return "\n".join(
            f"{m['role']}: {m['content']}" for m in self.get_messages()
        )


class ConversationMemory(BaseMemory):
    """Sliding-window memory keeping the last *window* messages.

    Args:
        window: Maximum number of messages to retain.
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self._messages: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self.window:
            self._messages = self._messages[-self.window :]

    def get_messages(self) -> list[dict[str, str]]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()


class SummaryMemory(BaseMemory):
    """Memory that keeps a running summary plus recent messages.

    When the total character count exceeds *max_tokens* (approximated as
    characters), older messages are collapsed into a cumulative summary.

    Args:
        max_tokens: Approximate character budget before summarisation.
    """

    def __init__(self, max_tokens: int = 2000) -> None:
        self.max_tokens = max_tokens
        self._summary: str = ""
        self._messages: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        self._maybe_summarise()

    def _maybe_summarise(self) -> None:
        total = len(self._summary) + sum(
            len(m["content"]) for m in self._messages
        )
        if total <= self.max_tokens:
            return
        # Collapse the oldest half of messages into the summary
        half = max(len(self._messages) // 2, 1)
        to_summarise = self._messages[:half]
        self._messages = self._messages[half:]
        parts = [f"{m['role']}: {m['content']}" for m in to_summarise]
        addition = " | ".join(parts)
        if self._summary:
            self._summary = f"{self._summary} | {addition}"
        else:
            self._summary = addition

    def get_messages(self) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if self._summary:
            msgs.append({"role": "system", "content": f"Summary: {self._summary}"})
        msgs.extend(self._messages)
        return msgs

    def clear(self) -> None:
        self._summary = ""
        self._messages.clear()

    @property
    def summary(self) -> str:
        """Return the current running summary."""
        return self._summary


class VectorMemory(BaseMemory):
    """Simple in-memory vector similarity search using character n-grams.

    Messages are stored alongside a sparse n-gram vector for cosine-similarity
    retrieval.  No external dependencies are required.

    Args:
        n: Character n-gram size (default 3).
    """

    def __init__(self, n: int = 3) -> None:
        self._n = n
        self._messages: list[dict[str, str]] = []
        self._vectors: list[Counter[str]] = []

    # -- BaseMemory interface -----------------------------------------------

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        self._vectors.append(self._vectorise(content))

    def get_messages(self) -> list[dict[str, str]]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()
        self._vectors.clear()

    # -- similarity search --------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict[str, str]]:
        """Find the most similar messages to *query*.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of message dicts ordered by descending similarity.
        """
        if not self._vectors:
            return []
        q_vec = self._vectorise(query)
        scored: list[tuple[float, int]] = []
        for idx, vec in enumerate(self._vectors):
            sim = self._cosine(q_vec, vec)
            scored.append((sim, idx))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [self._messages[idx] for _, idx in scored[:top_k]]

    # -- internal helpers ---------------------------------------------------

    def _vectorise(self, text: str) -> Counter[str]:
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        ngrams: list[str] = []
        for i in range(len(text) - self._n + 1):
            ngrams.append(text[i : i + self._n])
        return Counter(ngrams)

    @staticmethod
    def _cosine(a: Counter[str], b: Counter[str]) -> float:
        keys = set(a.keys()) & set(b.keys())
        if not keys:
            return 0.0
        dot = sum(a[k] * b[k] for k in keys)
        mag_a = math.sqrt(sum(v * v for v in a.values()))
        mag_b = math.sqrt(sum(v * v for v in b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)
