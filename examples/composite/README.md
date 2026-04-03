# Composite Example

A simulated DevOps incident response system that combines the
**SupervisorAgent** and **MultiAgent** patterns.  A top-level supervisor
routes incidents to specialised teams, where each team is a collaborative
multi-agent group.

## Architecture

```
Incident Alert
    |
    v
+-------------------------+
|   SupervisorAgent       |
|   (IncidentCommander)   |
|                         |
|   router: classifies    |
|   incident and routes   |
|   to the right team     |
+----+----------+---------+
     |          |
     v          v
+---------+  +---------+
|MultiAgent|  |MultiAgent|
|(InfraTeam)|  |(AppTeam) |
|           |  |          |
| monitor   |  | logger   |
| fixer     |  | deployer |
| verifier  |  | tester   |
+-----------+  +----------+
```

## Running

```bash
cd examples/composite
python main.py
```

No external dependencies required -- all actions are simulated.
