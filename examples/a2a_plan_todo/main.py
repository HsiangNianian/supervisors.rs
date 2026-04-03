"""A2A Plan/Todo Multi-Agent Example — DeepSeek Reasoner.

A main planning agent with three sub-agents:

* **planner_agent** — main agent with ``todo`` tool, delegates to sub-agents.
* **write_file_agent** — writes content to files.
* **read_file_agent** — reads file contents.
* **shell_agent** — runs ``pwd`` and ``ls`` commands.

All agents use the DeepSeek ``deepseek-reasoner`` model via the OpenAI SDK.

Agent-to-agent communication uses ``send_task()`` which wraps a ``Message``
object and routes it through the supervisor.

Usage::

    cd examples/a2a_plan_todo
    cp .env.example .env   # fill in OPENAI_API_KEY
    uv run python main.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from supervisors import Agent, Message, Supervisor

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")

_api_key = os.getenv("OPENAI_API_KEY", "")
if not _api_key:
    print(
        "Error: OPENAI_API_KEY is not set.\n"
        "    Copy .env.example -> .env and fill in your key."
    )
    sys.exit(1)

_base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
client = OpenAI(api_key=_api_key, base_url=_base_url)
MODEL = os.getenv("MODEL", "deepseek-reasoner")

# ---------------------------------------------------------------------------
# Shared todo list (mutable state visible to planner_agent)
# ---------------------------------------------------------------------------

_todo_list: list[dict[str, str]] = []


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def todo(action: str, item: str = "", index: int = 0) -> str:
    """Manage a todo list. Actions: add, list, complete, remove."""
    action = action.lower().strip()
    if action == "add":
        if not item:
            return "Error: 'item' is required for 'add' action."
        _todo_list.append({"task": item, "status": "pending"})
        return f"Added todo: {item}"
    elif action == "list":
        if not _todo_list:
            return "Todo list is empty."
        lines = []
        for i, t in enumerate(_todo_list):
            lines.append(f"  [{i}] [{t['status']}] {t['task']}")
        return "Todo list:\n" + "\n".join(lines)
    elif action == "complete":
        if 0 <= index < len(_todo_list):
            _todo_list[index]["status"] = "done"
            return f"Marked todo '{_todo_list[index]['task']}' as done."
        return f"Error: index {index} out of range."
    elif action == "remove":
        if 0 <= index < len(_todo_list):
            removed = _todo_list.pop(index)
            return f"Removed todo: {removed['task']}"
        return f"Error: index {index} out of range."
    else:
        return f"Error: unknown action '{action}'. Use: add, list, complete, remove."


def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Successfully wrote {len(content)} bytes to {path}"
    except Exception as exc:
        return f"Error writing file: {exc}"


def read_file(path: str) -> str:
    """Read content from a file."""
    try:
        content = Path(path).read_text()
        return content
    except Exception as exc:
        return f"Error reading file: {exc}"


def pwd() -> str:
    """Return the current working directory."""
    return str(Path.cwd())


def ls(path: str = ".") -> str:
    """List directory contents."""
    try:
        entries = sorted(os.listdir(path))
        return "\n".join(entries)
    except Exception as exc:
        return f"Error listing directory: {exc}"


# ---------------------------------------------------------------------------
# OpenAI tool schemas
# ---------------------------------------------------------------------------

_TODO_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "todo",
        "description": (
            "Manage a todo list. "
            "Actions: 'add' (requires 'item'), 'list', 'complete' (requires 'index'), 'remove' (requires 'index')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "One of: add, list, complete, remove",
                },
                "item": {
                    "type": "string",
                    "description": "Todo item text (required for 'add')",
                },
                "index": {
                    "type": "integer",
                    "description": "Index in the list (required for 'complete' and 'remove')",
                },
            },
            "required": ["action"],
        },
    },
}

_WRITE_FILE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write content to a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to write to",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write",
                },
            },
            "required": ["path", "content"],
        },
    },
}

_READ_FILE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read and return the content of a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to read",
                },
            },
            "required": ["path"],
        },
    },
}

_PWD_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "pwd",
        "description": "Return the current working directory.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

_LS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ls",
        "description": "List the contents of a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (default: '.')",
                },
            },
            "required": [],
        },
    },
}

# Delegation schemas — each takes a message string to send to the sub-agent
_DELEGATE_WRITE: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "send_task_write",
        "description": "Send a task message to the write_file_agent via A2A communication.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message/task to send to write_file_agent",
                },
            },
            "required": ["message"],
        },
    },
}

_DELEGATE_READ: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "send_task_read",
        "description": "Send a task message to the read_file_agent via A2A communication.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message/task to send to read_file_agent",
                },
            },
            "required": ["message"],
        },
    },
}

_DELEGATE_SHELL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "send_task_shell",
        "description": "Send a task message to the shell_agent via A2A communication.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message/task to send to shell_agent",
                },
            },
            "required": ["message"],
        },
    },
}

# Tool sets per agent
MAIN_TOOLS = [_TODO_SCHEMA, _DELEGATE_WRITE, _DELEGATE_READ, _DELEGATE_SHELL]
WRITE_FILE_TOOLS = [_WRITE_FILE_SCHEMA]
READ_FILE_TOOLS = [_READ_FILE_SCHEMA]
SHELL_TOOLS = [_PWD_SCHEMA, _LS_SCHEMA]


# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------


class ReActAgent(Agent):
    """Agent that uses ReAct (Reason + Act) via OpenAI function-calling."""

    def __init__(
        self,
        name: str,
        *,
        system_prompt: str,
        tools: list[dict[str, Any]],
        tool_map: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name)
        self.system_prompt = system_prompt
        self.tools = tools
        self.tool_map: dict[str, Any] = tool_map or {}
        self.last_response: str | None = None

    def react(self, user_message: str, *, max_steps: int = 10) -> str:
        """Run a ReAct reasoning loop and return the final text answer."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        for _step in range(max_steps):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=self.tools or None,
                tool_choice="auto" if self.tools else None,
            )

            choice = response.choices[0]
            msg = choice.message
            assistant_turn: dict[str, Any] = {"role": "assistant"}
            if msg.content:
                assistant_turn["content"] = msg.content
            if msg.tool_calls:
                assistant_turn["tool_calls"] = [
                    tc.model_dump() for tc in msg.tool_calls
                ]
            messages.append(assistant_turn)

            if not msg.tool_calls:
                self.last_response = msg.content or ""
                return self.last_response

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args: dict[str, Any] = json.loads(tc.function.arguments)

                print(f"  [{self.name}] tool: {fn_name}({fn_args})")

                if fn_name in self.tool_map:
                    result = str(self.tool_map[fn_name](**fn_args))
                else:
                    result = f"Unknown tool: {fn_name}"

                preview = result[:300] + ("..." if len(result) > 300 else "")
                print(f"  [{self.name}] result: {preview}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

        return "(max reasoning steps reached)"

    def handle_message(self, msg: Message) -> None:
        """Handle an incoming supervisor message via a ReAct loop."""
        response = self.react(msg.content)
        self.last_response = response
        if msg.sender and self.supervisor:
            try:
                self.send(msg.sender, response)
            except (KeyError, RuntimeError):
                pass


# A2A send_task helpers
# ---------------------------------------------------------------------------


def _make_send_task(sub_agent: ReActAgent, tool_name: str):
    """Return a tool callable that sends a Message to *sub_agent* via the supervisor.

    The tool receives a ``message`` string, wraps it in a ``Message`` object,
    sends it through the supervisor, then runs the supervisor to process it
    and returns the sub-agent's response.
    """

    def _send_task(message: str) -> str:
        preview = message[:80] + ("..." if len(message) > 80 else "")
        print(f'    -> {tool_name}: [{sub_agent.name}] "{preview}"')

        msg = Message(
            sender="planner_agent",
            recipient=sub_agent.name,
            content=message,
        )
        sub_agent.supervisor.send(msg)
        sub_agent.supervisor.run_once()

        return sub_agent.last_response or "(no response)"

    return _send_task


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("A2A Plan/Todo Multi-Agent System (supervisor.rs)")
    print(f"Model: {MODEL}")
    print(f"API:   {_base_url}")
    print("=" * 55)

    sup = Supervisor()

    # -- Create sub-agents ---------------------------------------------------
    write_file_agent = ReActAgent(
        "write_file_agent",
        system_prompt=(
            "You are a file-writing assistant. Use the write_file tool to write "
            "content to files. Be helpful and confirm what was written."
        ),
        tools=WRITE_FILE_TOOLS,
    )

    read_file_agent = ReActAgent(
        "read_file_agent",
        system_prompt=(
            "You are a file-reading assistant. Use the read_file tool to read "
            "and display file contents. Be concise."
        ),
        tools=READ_FILE_TOOLS,
    )

    shell_agent = ReActAgent(
        "shell_agent",
        system_prompt=(
            "You are a shell assistant. Use pwd to show the current directory "
            "and ls to list directory contents. Be concise and helpful."
        ),
        tools=SHELL_TOOLS,
    )

    # -- Create main planner agent -------------------------------------------
    planner_agent = ReActAgent(
        "planner_agent",
        system_prompt=(
            "You are a planning assistant that manages tasks and delegates to specialists.\n"
            "Tools available:\n"
            "  - todo: manage a todo list (add, list, complete, remove tasks)\n"
            "  - send_task (to write_file_agent): send a message to the file-writing specialist\n"
            "  - send_task (to read_file_agent): send a message to the file-reading specialist\n"
            "  - send_task (to shell_agent): send a message to the directory specialist\n\n"
            "Use ReAct reasoning: think step-by-step, use the todo tool to plan, "
            "delegate to sub-agents via send_task when needed, then give a final answer."
        ),
        tools=MAIN_TOOLS,
    )

    # -- Register with supervisor --------------------------------------------
    write_file_agent.register(sup)
    read_file_agent.register(sup)
    shell_agent.register(sup)
    planner_agent.register(sup)

    # -- Wire up tool maps ---------------------------------------------------
    write_file_agent.tool_map = {
        "write_file": write_file,
    }
    read_file_agent.tool_map = {
        "read_file": read_file,
    }
    shell_agent.tool_map = {
        "pwd": pwd,
        "ls": ls,
    }
    planner_agent.tool_map = {
        "todo": todo,
        "send_task_write": _make_send_task(write_file_agent, "send_task_write"),
        "send_task_read": _make_send_task(read_file_agent, "send_task_read"),
        "send_task_shell": _make_send_task(shell_agent, "send_task_shell"),
    }

    print(f"[OK] {sup.agent_count()} agents registered: {', '.join(sup.agent_names())}")
    print("[OK] Planner tools: todo + send_task_write/read/shell (A2A message-based)")
    print("[OK] Sub-agents: write_file, read_file, shell (pwd+ls)")
    print("\nType a message to the planner agent.")
    print("Commands: 'quit' to exit, 'todos' to show current todo list.")
    print(f"{'-' * 55}")

    # -- Interactive loop ----------------------------------------------------
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                break
            if user_input.lower() == "todos":
                print(todo("list"))
                continue

            print()
            response = planner_agent.react(user_input)
            print(f"\nAgent: {response}")

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as exc:
            print(f"Error: {exc}")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
