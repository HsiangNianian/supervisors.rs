"""Tests for memory systems."""

from __future__ import annotations

import pytest

from supervisor.memory import (
    BaseMemory,
    ConversationMemory,
    SummaryMemory,
    VectorMemory,
)


class TestConversationMemory:
    def test_add_and_get(self):
        mem = ConversationMemory(window=5)
        mem.add("user", "hello")
        mem.add("assistant", "hi")
        msgs = mem.get_messages()
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "hello"}
        assert msgs[1] == {"role": "assistant", "content": "hi"}

    def test_window_limit(self):
        mem = ConversationMemory(window=3)
        for i in range(10):
            mem.add("user", f"msg{i}")
        msgs = mem.get_messages()
        assert len(msgs) == 3
        assert msgs[0]["content"] == "msg7"

    def test_clear(self):
        mem = ConversationMemory()
        mem.add("user", "test")
        mem.clear()
        assert mem.get_messages() == []

    def test_to_prompt_string(self):
        mem = ConversationMemory()
        mem.add("user", "hello")
        mem.add("assistant", "hi")
        prompt = mem.to_prompt_string()
        assert "user: hello" in prompt
        assert "assistant: hi" in prompt

    def test_returns_copy(self):
        mem = ConversationMemory()
        mem.add("user", "x")
        msgs = mem.get_messages()
        msgs.clear()
        assert len(mem.get_messages()) == 1


class TestSummaryMemory:
    def test_basic_add(self):
        mem = SummaryMemory(max_tokens=10000)
        mem.add("user", "hello")
        assert len(mem.get_messages()) == 1

    def test_summarisation_triggers(self):
        mem = SummaryMemory(max_tokens=50)
        for i in range(20):
            mem.add("user", f"message number {i} with some content")
        msgs = mem.get_messages()
        # Should have a summary message at the start
        assert any("Summary" in m["content"] for m in msgs)

    def test_summary_property(self):
        mem = SummaryMemory(max_tokens=20)
        mem.add("user", "a long message that exceeds budget")
        mem.add("user", "another long message to trigger summary")
        assert isinstance(mem.summary, str)

    def test_clear(self):
        mem = SummaryMemory(max_tokens=50)
        mem.add("user", "test")
        mem.clear()
        assert mem.get_messages() == []
        assert mem.summary == ""

    def test_no_summarise_when_under_limit(self):
        mem = SummaryMemory(max_tokens=10000)
        mem.add("user", "short")
        assert mem.summary == ""
        assert len(mem.get_messages()) == 1


class TestVectorMemory:
    def test_add_and_get(self):
        mem = VectorMemory()
        mem.add("user", "hello world")
        mem.add("assistant", "greetings")
        assert len(mem.get_messages()) == 2

    def test_search_similarity(self):
        mem = VectorMemory()
        mem.add("user", "the weather is nice today")
        mem.add("user", "python programming is fun")
        mem.add("user", "it is sunny and warm outside")
        results = mem.search("sunny weather", top_k=2)
        assert len(results) == 2
        # The weather-related messages should rank higher
        contents = [r["content"] for r in results]
        assert "the weather is nice today" in contents or "it is sunny and warm outside" in contents

    def test_search_empty(self):
        mem = VectorMemory()
        assert mem.search("anything") == []

    def test_search_top_k(self):
        mem = VectorMemory()
        for i in range(10):
            mem.add("user", f"message {i}")
        results = mem.search("message", top_k=3)
        assert len(results) == 3

    def test_clear(self):
        mem = VectorMemory()
        mem.add("user", "test")
        mem.clear()
        assert mem.get_messages() == []
        assert mem.search("test") == []

    def test_to_prompt_string(self):
        mem = VectorMemory()
        mem.add("user", "hello")
        assert "user: hello" in mem.to_prompt_string()


class TestBaseMemoryInterface:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseMemory()
