from supervisors import Agent, Supervisor, Message
from supervisors.ext.function_calling import FunctionCallingExtension


class EchoAgent(Agent):
    """Agent that echoes received messages."""

    def __init__(self) -> None:
        super().__init__("echo")
        self.log: list[str] = []

    def handle_message(self, msg: Message) -> None:
        line = f"[{self.name}] received from '{msg.sender}': {msg.content}"
        self.log.append(line)
        print(line)


class ReverseAgent(Agent):
    """Agent that reverses the message content."""

    def __init__(self) -> None:
        super().__init__("reverse")
        self.log: list[str] = []

    def handle_message(self, msg: Message) -> None:
        line = f"[{self.name}] reversed: {msg.content[::-1]}"
        self.log.append(line)
        print(line)


class ToolAgent(Agent):
    """Agent with function-calling tools."""

    def __init__(self) -> None:
        super().__init__("tools")
        fc = FunctionCallingExtension()

        @fc.tool(description="Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        self.use(fc)
        self.log: list[str] = []

    def handle_message(self, msg: Message) -> None:
        fc = self.extensions["function_calling"]
        # Demo: call the registered tool
        result = fc.call_tool("add", a=2, b=3)
        line = f"[{self.name}] called add(2, 3) = {result}"
        self.log.append(line)
        print(line)


def main() -> None:
    """Run a short demonstration of the supervisor agent framework."""
    sup = Supervisor()

    echo = EchoAgent()
    reverse = ReverseAgent()
    tools = ToolAgent()

    echo.register(sup)
    reverse.register(sup)
    tools.register(sup)

    sup.send(Message("main", "echo", "Hello, Agent!"))
    sup.send(Message("main", "reverse", "supervisor.rs"))
    sup.send(Message("echo", "echo", "Self-message"))
    sup.send(Message("main", "tools", "call a tool"))

    # Demonstrate A2A: echo sends to reverse
    echo.send("reverse", "A2A message from echo")

    processed = sup.run_once()
    print(f"\nProcessed {processed} message(s) across {sup.agent_count()} agent(s).")
    print(f"Registered agents: {sup.agent_names()}")
    print(f"Echo agent extensions: {echo.extensions}")
    print(f"Tool agent extensions: {tools.extensions}")


if __name__ == "__main__":
    main()
