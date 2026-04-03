# A2A Plan/Todo Multi-Agent Example

This example demonstrates an **Agent-to-Agent (A2A)** multi-agent system with
a planning/todo workflow built with `supervisors.rs`. Four agents collaborate
via ReAct-style reasoning powered by DeepSeek:

| Agent | Role | Tools |
|-------|------|-------|
| **planner_agent** | Main orchestrator — plans tasks via `todo` tool | `todo`, delegates to all sub-agents |
| **write_file_agent** | File writing specialist | `write_file` |
| **read_file_agent** | File reading specialist | `read_file` |
| **shell_agent** | Directory specialist | `pwd`, `ls` |

## Quick Start

```bash
cd examples/a2a_plan_todo

# 1. Create your .env with an OpenAI API key
cp .env.example .env
# Edit .env and fill in your OPENAI_API_KEY

# 2. Run with uv (installs deps + builds supervisor automatically)
uv run python main.py
```

## How It Works

1. The user types a message in the terminal.
2. **planner_agent** receives it and enters a ReAct loop (Thought → Action → Observation).
3. The planner can use the `todo` tool to add/list/complete/remove tasks.
4. If the task involves file writing, planner delegates to **write_file_agent**.
5. If the task involves file reading, planner delegates to **read_file_agent**.
6. If the task involves directory operations, planner delegates to **shell_agent**.
7. Sub-agents run their own ReAct loops and return results.
8. **planner_agent** synthesises the final answer and prints it.

## Architecture

```
User ←→ planner_agent (todo tool)
              ├──→ write_file_agent (write_file tool)
              ├──→ read_file_agent (read_file tool)
              └──→ shell_agent (pwd + ls tools)
```

All agents use the DeepSeek `deepseek-reasoner` model via the OpenAI SDK
with a configurable base URL.

## Commands

- Type a message to interact with the planner agent
- `todos` — show the current todo list
- `quit` / `exit` / `q` — exit the program
