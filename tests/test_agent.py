"""Tests for the Agent base class and extension plugin system."""

import pytest

from supervisors._core import Message, Supervisor
from supervisors.agent import Agent
from supervisors.ext import Extension
from supervisors.ext.rag import RAGExtension
from supervisors.ext.function_calling import FunctionCallingExtension, ToolSpec
from supervisors.ext.mcp import MCPExtension, MCPClient
from supervisors.ext.skills import SkillsExtension, Skill
from supervisors.ext.a2a import A2AExtension

# ---------------------------------------------------------------------------
# Agent base class
# ---------------------------------------------------------------------------


class TestAgent:
    def test_create_agent(self):
        agent = Agent("test")
        assert agent.name == "test"
        assert agent.extensions == {}
        assert agent.supervisor is None

    def test_register_with_supervisor(self):
        sup = Supervisor()
        agent = Agent("myagent")
        agent.register(sup)
        assert "myagent" in sup.agent_names()
        assert agent.supervisor is sup

    def test_unregister(self):
        sup = Supervisor()
        agent = Agent("myagent")
        agent.register(sup)
        result = agent.unregister()
        assert result is True
        assert "myagent" not in sup.agent_names()
        assert agent.supervisor is None

    def test_unregister_without_supervisor(self):
        agent = Agent("myagent")
        assert agent.unregister() is False

    def test_subclass_handle_message(self):
        received = []

        class MyAgent(Agent):
            def handle_message(self, msg):
                received.append(msg.content)

        sup = Supervisor()
        agent = MyAgent("sub")
        agent.register(sup)
        sup.send(Message("src", "sub", "hello"))
        sup.run_once()
        assert received == ["hello"]

    def test_a2a_send(self):
        received = []

        class Receiver(Agent):
            def handle_message(self, msg):
                received.append((msg.sender, msg.content))

        sup = Supervisor()
        sender = Agent("sender")
        receiver = Receiver("receiver")
        sender.register(sup)
        receiver.register(sup)
        sender.send("receiver", "hi from sender")
        sup.run_once()
        assert received == [("sender", "hi from sender")]

    def test_a2a_send_without_supervisor_raises(self):
        agent = Agent("lonely")
        with pytest.raises(RuntimeError, match="not registered"):
            agent.send("nobody", "msg")

    def test_repr(self):
        agent = Agent("test")
        r = repr(agent)
        assert "test" in r

    def test_use_returns_self(self):
        agent = Agent("a")
        result = agent.use(Extension())
        assert result is agent

    def test_chained_use(self):
        ext1 = Extension()
        ext1.name = "ext1"
        ext2 = Extension()
        ext2.name = "ext2"
        agent = Agent("a").use(ext1).use(ext2)
        assert "ext1" in agent.extensions
        assert "ext2" in agent.extensions


# ---------------------------------------------------------------------------
# Extension base class
# ---------------------------------------------------------------------------


class TestExtension:
    def test_default_name(self):
        ext = Extension()
        assert ext.name == ""

    def test_auto_name_from_subclass(self):
        class MyCustomExtension(Extension):
            pass

        ext = MyCustomExtension()
        assert ext.name == "MyCustomExtension"

    def test_explicit_name(self):
        class MyExt(Extension):
            name = "custom_name"

        ext = MyExt()
        assert ext.name == "custom_name"

    def test_on_load_called(self):
        loaded = []

        class TrackingExt(Extension):
            name = "tracking"

            def on_load(self, agent):
                loaded.append(agent.name)

        agent = Agent("a")
        agent.use(TrackingExt())
        assert loaded == ["a"]

    def test_on_unload_called(self):
        unloaded = []

        class TrackingExt(Extension):
            name = "tracking"

            def on_unload(self, agent):
                unloaded.append(agent.name)

        agent = Agent("a")
        agent.use(TrackingExt())
        agent.remove_extension("tracking")
        assert unloaded == ["a"]

    def test_replace_extension_calls_unload(self):
        events = []

        class Ext1(Extension):
            name = "shared"

            def on_unload(self, agent):
                events.append("unload_1")

        class Ext2(Extension):
            name = "shared"

            def on_load(self, agent):
                events.append("load_2")

        agent = Agent("a")
        agent.use(Ext1())
        agent.use(Ext2())
        assert events == ["unload_1", "load_2"]

    def test_on_message_passthrough(self):
        ext = Extension()
        agent = Agent("a")
        msg = Message("x", "a", "hello")
        assert ext.on_message(agent, msg) is None

    def test_on_message_modifies(self):
        class UpperExt(Extension):
            name = "upper"

            def on_message(self, agent, msg):
                return Message(msg.sender, msg.recipient, msg.content.upper())

        received = []

        class MyAgent(Agent):
            def handle_message(self, msg):
                received.append(msg.content)

        sup = Supervisor()
        agent = MyAgent("a")
        agent.use(UpperExt())
        agent.register(sup)
        sup.send(Message("x", "a", "hello"))
        sup.run_once()
        assert received == ["HELLO"]

    def test_on_message_swallow(self):
        class SwallowExt(Extension):
            name = "swallow"

            def on_message(self, agent, msg):
                raise StopIteration

        received = []

        class MyAgent(Agent):
            def handle_message(self, msg):
                received.append(msg.content)

        sup = Supervisor()
        agent = MyAgent("a")
        agent.use(SwallowExt())
        agent.register(sup)
        sup.send(Message("x", "a", "hello"))
        sup.run_once()
        assert received == []

    def test_remove_nonexistent_extension(self):
        agent = Agent("a")
        assert agent.remove_extension("nope") is False

    def test_repr(self):
        ext = Extension()
        ext.name = "test"
        r = repr(ext)
        assert "test" in r


# ---------------------------------------------------------------------------
# RAG Extension
# ---------------------------------------------------------------------------


class TestRAGExtension:
    def test_retrieve_not_implemented(self):
        rag = RAGExtension()
        with pytest.raises(NotImplementedError):
            rag.retrieve("query")

    def test_add_documents_not_implemented(self):
        rag = RAGExtension()
        with pytest.raises(NotImplementedError):
            rag.add_documents(["doc"])

    def test_custom_rag_implementation(self):
        class InMemoryRAG(RAGExtension):
            def __init__(self):
                super().__init__(auto_retrieve=False)
                self._docs = []

            def retrieve(self, query, top_k=None):
                k = top_k or self.top_k
                return [d for d in self._docs if query.lower() in d.lower()][:k]

            def add_documents(self, docs, **kwargs):
                self._docs.extend(docs)

        rag = InMemoryRAG()
        rag.add_documents(["Hello world", "Python docs", "Hello Python"])
        results = rag.retrieve("hello")
        assert len(results) == 2
        assert "Hello world" in results
        assert "Hello Python" in results

    def test_auto_retrieve_enriches_message(self):
        class SimpleRAG(RAGExtension):
            def __init__(self):
                super().__init__(auto_retrieve=True, top_k=2)

            def retrieve(self, query, top_k=None):
                return ["relevant doc 1", "relevant doc 2"]

            def add_documents(self, docs, **kwargs):
                pass

        received = []

        class MyAgent(Agent):
            def handle_message(self, msg):
                received.append(msg.content)

        sup = Supervisor()
        agent = MyAgent("a")
        agent.use(SimpleRAG())
        agent.register(sup)
        sup.send(Message("x", "a", "search query"))
        sup.run_once()
        assert len(received) == 1
        assert "search query" in received[0]
        assert "[RAG context]" in received[0]
        assert "relevant doc 1" in received[0]

    def test_auto_retrieve_disabled(self):
        class SimpleRAG(RAGExtension):
            def __init__(self):
                super().__init__(auto_retrieve=False)

            def retrieve(self, query, top_k=None):
                return ["should not appear"]

            def add_documents(self, docs, **kwargs):
                pass

        received = []

        class MyAgent(Agent):
            def handle_message(self, msg):
                received.append(msg.content)

        sup = Supervisor()
        agent = MyAgent("a")
        agent.use(SimpleRAG())
        agent.register(sup)
        sup.send(Message("x", "a", "original"))
        sup.run_once()
        assert received == ["original"]

    def test_name(self):
        rag = RAGExtension()
        assert rag.name == "rag"


# ---------------------------------------------------------------------------
# Function Calling Extension
# ---------------------------------------------------------------------------


class TestFunctionCallingExtension:
    def test_register_and_call_tool(self):
        fc = FunctionCallingExtension()

        def add(a: int, b: int) -> int:
            return a + b

        fc.register_tool(add)
        assert fc.call_tool("add", a=1, b=2) == 3

    def test_tool_decorator(self):
        fc = FunctionCallingExtension()

        @fc.tool
        def multiply(a: int, b: int) -> int:
            return a * b

        assert fc.call_tool("multiply", a=3, b=4) == 12

    def test_tool_decorator_with_args(self):
        fc = FunctionCallingExtension()

        @fc.tool(name="my_add", description="Add numbers")
        def add(a: int, b: int) -> int:
            return a + b

        specs = fc.get_tools_spec()
        assert len(specs) == 1
        assert specs[0]["name"] == "my_add"
        assert specs[0]["description"] == "Add numbers"

    def test_call_unknown_tool_raises(self):
        fc = FunctionCallingExtension()
        with pytest.raises(KeyError, match="nope"):
            fc.call_tool("nope")

    def test_list_tools(self):
        fc = FunctionCallingExtension()
        fc.register_tool(lambda: None, name="t1")
        fc.register_tool(lambda: None, name="t2")
        tools = fc.list_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"t1", "t2"}

    def test_auto_parameters(self):
        fc = FunctionCallingExtension()

        def typed_func(a: int, b: str, c: float, d: bool) -> None:
            pass

        fc.register_tool(typed_func)
        spec = fc.get_tools_spec()[0]
        props = spec["parameters"]["properties"]
        assert props["a"]["type"] == "integer"
        assert props["b"]["type"] == "string"
        assert props["c"]["type"] == "number"
        assert props["d"]["type"] == "boolean"

    def test_name(self):
        fc = FunctionCallingExtension()
        assert fc.name == "function_calling"

    def test_tool_spec_repr(self):
        spec = ToolSpec("test", lambda: None, "desc")
        assert "test" in repr(spec)


# ---------------------------------------------------------------------------
# MCP Extension
# ---------------------------------------------------------------------------


class TestMCPExtension:
    def test_mcp_tool_decorator(self):
        mcp = MCPExtension()

        @mcp.mcp_tool(description="Echo text")
        def echo(text: str) -> str:
            return text

        tools = mcp.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "echo"

    def test_server_handle_request(self):
        mcp = MCPExtension()

        @mcp.mcp_tool
        def double(n: int) -> int:
            return n * 2

        result = mcp.server.handle_request({"tool": "double", "args": {"n": 5}})
        assert result == {"result": 10}

    def test_server_unknown_tool(self):
        mcp = MCPExtension()
        result = mcp.server.handle_request({"tool": "nope", "args": {}})
        assert "error" in result

    def test_client_connect_disconnect(self):
        client = MCPClient("http://localhost:8080")
        assert not client.connected
        client.connect()
        assert client.connected
        client.disconnect()
        assert not client.connected

    def test_client_call_without_connect_raises(self):
        client = MCPClient("http://localhost:8080")
        with pytest.raises(RuntimeError, match="not connected"):
            client.call("tool")

    def test_on_load_connects_client(self):
        mcp = MCPExtension(server_url="http://localhost:8080")
        agent = Agent("a")
        mcp.on_load(agent)
        assert mcp.client.connected

    def test_on_unload_disconnects(self):
        mcp = MCPExtension(server_url="http://localhost:8080")
        agent = Agent("a")
        mcp.on_load(agent)
        mcp.on_unload(agent)
        assert not mcp.client.connected

    def test_call_remote_without_client_raises(self):
        mcp = MCPExtension()
        with pytest.raises(RuntimeError, match="No MCPClient"):
            mcp.call_remote("tool")

    def test_name(self):
        mcp = MCPExtension()
        assert mcp.name == "mcp"


# ---------------------------------------------------------------------------
# Skills Extension
# ---------------------------------------------------------------------------


class TestSkillsExtension:
    def test_register_and_invoke(self):
        skills = SkillsExtension()

        def greet(agent, msg):
            return f"Hello, {msg.sender}!"

        skills.register_skill(greet)
        agent = Agent("a")
        msg = Message("bob", "a", "hi")
        result = skills.invoke("greet", agent, msg)
        assert result == "Hello, bob!"

    def test_skill_decorator(self):
        skills = SkillsExtension()

        @skills.skill
        def reverse(agent, msg):
            return msg.content[::-1]

        agent = Agent("a")
        msg = Message("x", "a", "hello")
        assert skills.invoke("reverse", agent, msg) == "olleh"

    def test_invoke_unknown_raises(self):
        skills = SkillsExtension()
        with pytest.raises(KeyError, match="nope"):
            skills.invoke("nope", Agent("a"), Message("x", "a", "m"))

    def test_list_skills(self):
        skills = SkillsExtension()
        skills.register_skill(lambda a, m: None, name="s1")
        skills.register_skill(lambda a, m: None, name="s2")
        assert len(skills.list_skills()) == 2

    def test_name(self):
        sk = SkillsExtension()
        assert sk.name == "skills"

    def test_skill_repr(self):
        sk = Skill("test", lambda a, m: None)
        assert "test" in repr(sk)


# ---------------------------------------------------------------------------
# A2A Extension
# ---------------------------------------------------------------------------


class TestA2AExtension:
    def test_broadcast(self):
        received = {"a": [], "b": [], "c": []}

        class Collector(Agent):
            def handle_message(self, msg):
                received[self.name].append(msg.content)

        sup = Supervisor()
        agents = [Collector(n) for n in "abc"]
        for ag in agents:
            ag.use(A2AExtension())
            ag.register(sup)

        agents[0].extensions["a2a"].broadcast(agents[0], "hello all")
        sup.run_once()
        assert received["a"] == []  # sender excluded
        assert received["b"] == ["hello all"]
        assert received["c"] == ["hello all"]

    def test_discover_agents(self):
        sup = Supervisor()
        a = Agent("a")
        b = Agent("b")
        a2a = A2AExtension()
        a.use(a2a)
        a.register(sup)
        b.register(sup)
        names = a2a.discover_agents(a)
        assert set(names) == {"a", "b"}

    def test_broadcast_without_supervisor_raises(self):
        a2a = A2AExtension()
        agent = Agent("lonely")
        with pytest.raises(RuntimeError, match="not registered"):
            a2a.broadcast(agent, "msg")

    def test_discover_without_supervisor_raises(self):
        a2a = A2AExtension()
        agent = Agent("lonely")
        with pytest.raises(RuntimeError, match="not registered"):
            a2a.discover_agents(agent)

    def test_request_reply(self):
        a2a = A2AExtension()
        handler_called = []

        def on_reply(msg):
            handler_called.append(msg)

        sup = Supervisor()
        a = Agent("a")
        b = Agent("b")
        a.use(a2a)
        a.register(sup)
        b.register(sup)
        a2a.request(a, "b", "question", on_reply)
        # Check reply handler is stored
        h = a2a.get_reply_handler("b")
        assert h is on_reply
        # Second call returns None
        assert a2a.get_reply_handler("b") is None

    def test_name(self):
        a2a = A2AExtension()
        assert a2a.name == "a2a"


# ---------------------------------------------------------------------------
# Integration: multiple extensions on one agent
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_multiple_extensions(self):
        fc = FunctionCallingExtension()

        @fc.tool
        def add(a: int, b: int) -> int:
            return a + b

        skills = SkillsExtension()

        @skills.skill
        def greet(agent, msg):
            return f"Hi {msg.sender}"

        agent = Agent("multi")
        agent.use(fc).use(skills)
        assert "function_calling" in agent.extensions
        assert "skills" in agent.extensions

    def test_extension_chain_modifies_message(self):
        """Two extensions both modify the message before the agent sees it."""

        class PrefixExt(Extension):
            name = "prefix"

            def on_message(self, agent, msg):
                return Message(msg.sender, msg.recipient, "PREFIX:" + msg.content)

        class SuffixExt(Extension):
            name = "suffix"

            def on_message(self, agent, msg):
                return Message(msg.sender, msg.recipient, msg.content + ":SUFFIX")

        received = []

        class MyAgent(Agent):
            def handle_message(self, msg):
                received.append(msg.content)

        sup = Supervisor()
        agent = MyAgent("a")
        agent.use(PrefixExt())
        agent.use(SuffixExt())
        agent.register(sup)
        sup.send(Message("x", "a", "hello"))
        sup.run_once()
        assert received == ["PREFIX:hello:SUFFIX"]

    def test_agent_with_all_extensions(self):
        """Smoke test: load every extension type onto one agent."""

        class DummyRAG(RAGExtension):
            def __init__(self):
                super().__init__(auto_retrieve=False)

            def retrieve(self, query, top_k=None):
                return []

            def add_documents(self, docs, **kwargs):
                pass

        agent = Agent("full")
        agent.use(DummyRAG())
        agent.use(FunctionCallingExtension())
        agent.use(MCPExtension())
        agent.use(SkillsExtension())
        agent.use(A2AExtension())
        assert len(agent.extensions) == 5
