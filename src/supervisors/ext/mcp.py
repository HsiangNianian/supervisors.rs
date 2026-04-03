"""MCP (Model Context Protocol) extension.

Provides :class:`MCPExtension` with a lightweight MCP client, server, and
a ``mcp_tool`` decorator for exposing tools over the protocol.  Users may
alternatively integrate the official MCP SDK.

Example::

    from supervisors.ext.mcp import MCPExtension, mcp_tool

    mcp = MCPExtension(server_url="http://localhost:8080")

    @mcp.mcp_tool(description="Echo a message")
    def echo(text: str) -> str:
        return text

    agent.use(mcp)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from supervisors.ext import Extension

if TYPE_CHECKING:
    from supervisors.agent import Agent


class MCPToolSpec:
    """Descriptor for a tool exposed via MCP."""

    def __init__(
        self,
        name: str,
        func: Callable[..., Any],
        description: str = "",
    ) -> None:
        self.name = name
        self.func = func
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
        }

    def __repr__(self) -> str:
        return f"MCPToolSpec(name={self.name!r})"


class MCPClient:
    """Lightweight MCP client for connecting to an MCP server.

    This provides a minimal reference implementation.  Users who need
    full protocol support should integrate the official MCP SDK instead.

    Parameters:
        server_url: Base URL of the MCP server.
    """

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url
        self.connected = False

    def connect(self) -> None:
        """Establish a connection to the MCP server."""
        self.connected = True

    def disconnect(self) -> None:
        """Close the connection."""
        self.connected = False

    def call(self, tool_name: str, **kwargs: Any) -> Any:
        """Call a remote tool on the MCP server.

        In this reference implementation the call is a no-op stub.
        Replace with real HTTP/WebSocket logic or use the official SDK.
        """
        if not self.connected:
            raise RuntimeError("MCPClient is not connected")
        # Stub: real implementation would make an HTTP/WS request.
        return {"tool": tool_name, "args": kwargs, "result": None}


class MCPServer:
    """Lightweight MCP server that hosts tools for remote agents.

    Parameters:
        host: Bind address.
        port: Bind port.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        self.host = host
        self.port = port
        self._tools: Dict[str, MCPToolSpec] = {}
        self.running = False

    def register_tool(self, spec: MCPToolSpec) -> None:
        self._tools[spec.name] = spec

    def list_tools(self) -> List[MCPToolSpec]:
        return list(self._tools.values())

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming MCP request (reference implementation).

        Parameters:
            request: A dict with ``"tool"`` and ``"args"`` keys.

        Returns:
            A dict with the tool's return value under ``"result"``.
        """
        tool_name = request.get("tool", "")
        args = request.get("args", {})
        if tool_name not in self._tools:
            return {"error": f"Unknown tool '{tool_name}'"}
        try:
            result = self._tools[tool_name].func(**args)
        except Exception as exc:
            return {"error": str(exc)}
        return {"result": result}

    def start(self) -> None:
        """Start the server (stub)."""
        self.running = True

    def stop(self) -> None:
        """Stop the server (stub)."""
        self.running = False


class MCPExtension(Extension):
    """MCP extension for agents.

    Embeds an :class:`MCPClient` (for calling remote tools) and an
    :class:`MCPServer` (for exposing local tools).  Tools can be
    registered with :meth:`mcp_tool`.

    Parameters:
        server_url: URL of a remote MCP server to connect to.
        host: Address to bind the local MCP server.
        port: Port to bind the local MCP server.
    """

    name: str = "mcp"

    def __init__(
        self,
        server_url: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> None:
        self.client: Optional[MCPClient] = None
        if server_url:
            self.client = MCPClient(server_url)
        self.server = MCPServer(host=host, port=port)

    # -- decorator -----------------------------------------------------------

    def mcp_tool(
        self,
        func: Optional[Callable[..., Any]] = None,
        *,
        name: Optional[str] = None,
        description: str = "",
    ) -> Any:
        """Decorator to register a function as an MCP tool.

        Usage::

            @mcp.mcp_tool(description="Reverse a string")
            def reverse(text: str) -> str:
                return text[::-1]
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            spec = MCPToolSpec(
                name=name or fn.__name__,
                func=fn,
                description=description or (fn.__doc__ or "").strip(),
            )
            self.server.register_tool(spec)
            return fn

        if func is not None:
            return decorator(func)
        return decorator

    # -- lifecycle -----------------------------------------------------------

    def on_load(self, agent: "Agent") -> None:
        if self.client is not None:
            self.client.connect()

    def on_unload(self, agent: "Agent") -> None:
        if self.client is not None:
            self.client.disconnect()
        self.server.stop()

    # -- convenience ---------------------------------------------------------

    def call_remote(self, tool_name: str, **kwargs: Any) -> Any:
        """Call a tool on the remote MCP server via the client."""
        if self.client is None:
            raise RuntimeError("No MCPClient configured (server_url not set)")
        return self.client.call(tool_name, **kwargs)

    def list_tools(self) -> List[MCPToolSpec]:
        """Return all tools registered on the local server."""
        return self.server.list_tools()


__all__ = [
    "MCPExtension",
    "MCPClient",
    "MCPServer",
    "MCPToolSpec",
]
