# Supervisor + SubAgent Example

A simulated content moderation system that uses the **SupervisorAgent**
pattern to delegate specialised tasks to child agents.

## Architecture

```
User Content
    |
    v
+---------------------+
|  SupervisorAgent    |
|  (ModerationMgr)    |
|                     |
|  router: classifies |
|  content and routes |
|  to the right agent |
+-----+---------+-----+
      |         |
      v         v
+----------+ +----------+
| TextMod  | | ImageMod |
| Agent    | | Agent    |
+----------+ +----------+
```

## Running

```bash
cd examples/supervisor_subagent
python main.py
```

No external dependencies required -- all moderation is rule-based.
