# Multi-Agent Example

A simulated collaborative research team that uses the **MultiAgent** pattern
to coordinate peer agents working together on a research task.

## Architecture

```
Research Query
    |
    v
+---------------------+
|    MultiAgent       |
|    (ResearchTeam)   |
|                     |
|  +--------+  +--------+  +---------+
|  |Gatherer|  |Analyst |  |Writer   |
|  +--------+  +--------+  +---------+
|       |           |            |
|       +--- collaborate --------+
+---------------------+
    |
    v
Research Report
```

## Running

```bash
cd examples/multi_agent
python main.py
```

No external dependencies required -- all research is simulated.
