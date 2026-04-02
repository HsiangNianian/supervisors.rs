"""Agent base class with inheritance and extension plugin support.

Users create custom agents by subclassing :class:`Agent` and overriding
:meth:`handle_message`.  Extensions (RAG, Function Calling, MCP, Skills,
A2A, …) can be loaded onto any agent via :meth:`Agent.use`.

Example::

    from supervisor import Agent, Supervisor, Message

    class GreeterAgent(Agent):
        def handle_message(self, msg: Message) -> None:
            print(f"Hello from {self.name}! You said: {msg.content}")

    sup = Supervisor()
    greeter = GreeterAgent("greeter")
    greeter.register(sup)
    sup.send(Message("main", "greeter", "Hi!"))
    sup.run_once()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from supervisor._core import Message, Supervisor
from supervisor.ext import Extension

if TYPE_CHECKING:
    pass


class Agent:
    """Base class for all agents.

    Subclass and override :meth:`handle_message` to define custom behaviour.
    Load extensions with :meth:`use` and register with a :class:`Supervisor`
    via :meth:`register`.

    Parameters:
        name: Unique name for the agent within its supervisor.

    Attributes:
        name: The agent's name.
        extensions: Mapping of loaded extension names to instances.
        supervisor: The :class:`Supervisor` this agent is registered with
                    (or ``None``).
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.extensions: dict[str, Extension] = {}
        self.supervisor: Optional[Supervisor] = None

    # -- extension management ------------------------------------------------

    def use(self, extension: Extension) -> "Agent":
        """Load an *extension* plugin onto this agent.

        The extension's :meth:`~Extension.on_load` hook is called immediately.
        If an extension with the same :attr:`~Extension.name` is already
        loaded, it is replaced (after calling :meth:`~Extension.on_unload` on
        the old one).

        Returns *self* so calls can be chained::

            agent.use(RAGExtension()).use(FunctionCallingExtension())
        """
        key = extension.name
        if key in self.extensions:
            self.extensions[key].on_unload(self)
        self.extensions[key] = extension
        extension.on_load(self)
        return self

    def remove_extension(self, name: str) -> bool:
        """Remove a loaded extension by *name*.

        Returns ``True`` if the extension existed and was removed.
        """
        ext = self.extensions.pop(name, None)
        if ext is not None:
            ext.on_unload(self)
            return True
        return False

    # -- message handling ----------------------------------------------------

    def handle_message(self, msg: Message) -> None:
        """Process an incoming *msg*.

        Override in subclasses to implement custom logic.  The default
        implementation does nothing.
        """

    def _dispatch(self, msg: Message) -> None:
        """Internal handler registered with the :class:`Supervisor`.

        Runs each loaded extension's :meth:`~Extension.on_message` hook in
        load order.  If any hook raises :exc:`StopIteration` the message is
        swallowed.  Otherwise the (possibly modified) message is forwarded to
        :meth:`handle_message`.
        """
        current: Message = msg
        for ext in self.extensions.values():
            try:
                result = ext.on_message(self, current)
            except StopIteration:
                return  # message swallowed
            if result is not None:
                current = result
        self.handle_message(current)

    # -- supervisor integration ----------------------------------------------

    def register(self, supervisor: Supervisor) -> None:
        """Register this agent with a *supervisor*.

        The agent's internal dispatcher (which chains extension hooks before
        :meth:`handle_message`) is used as the message handler.
        """
        self.supervisor = supervisor
        supervisor.register(self.name, self._dispatch)

    def unregister(self) -> bool:
        """Remove this agent from its supervisor.

        Returns ``True`` if the agent was registered and successfully removed.
        """
        if self.supervisor is None:
            return False
        result = self.supervisor.unregister(self.name)
        if result:
            self.supervisor = None
        return result

    # -- A2A (agent-to-agent) convenience ------------------------------------

    def send(self, recipient: str, content: str) -> None:
        """Send a message to another agent registered on the same supervisor.

        This is the built-in *Agent-to-Agent* (A2A) communication that every
        agent inherits automatically.

        Raises:
            RuntimeError: If the agent is not registered with a supervisor.
        """
        if self.supervisor is None:
            raise RuntimeError(
                f"Agent '{self.name}' is not registered with a supervisor."
            )
        self.supervisor.send(Message(self.name, recipient, content))

    # -- dunder --------------------------------------------------------------

    def __repr__(self) -> str:
        ext_names = ", ".join(self.extensions) or "none"
        return f"Agent(name={self.name!r}, extensions=[{ext_names}])"


__all__ = ["Agent"]
