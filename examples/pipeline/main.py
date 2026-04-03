"""Pipeline Agent Example -- log processing pipeline.

Demonstrates the PipelineAgent pattern where raw log entries are processed
through sequential stages: parsing, enrichment, filtering, and formatting.

No external dependencies required.

Usage::

    cd examples/pipeline
    python main.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from supervisor import PipelineAgent, Message, Supervisor


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def parse_log(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a raw log line into structured fields.

    Expected format: ``LEVEL TIMESTAMP SOURCE message``
    """
    raw = ctx["input"]
    parts = raw.split(None, 3)
    if len(parts) >= 4:
        ctx["level"] = parts[0].upper()
        ctx["timestamp"] = parts[1]
        ctx["source"] = parts[2]
        ctx["message"] = parts[3]
    else:
        ctx["level"] = "UNKNOWN"
        ctx["timestamp"] = datetime.now(timezone.utc).isoformat()
        ctx["source"] = "unknown"
        ctx["message"] = raw
    print(f"  [parse] level={ctx['level']} source={ctx['source']}")
    return ctx


def enrich(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Add metadata such as severity score and processing timestamp."""
    severity_map = {"ERROR": 3, "WARN": 2, "INFO": 1, "DEBUG": 0, "UNKNOWN": 1}
    ctx["severity"] = severity_map.get(ctx["level"], 1)
    ctx["processed_at"] = datetime.now(timezone.utc).isoformat()
    # Simulated geo-lookup based on source.
    geo_map = {
        "web-server-01": "us-east-1",
        "api-gateway": "eu-west-1",
        "db-primary": "ap-southeast-1",
    }
    ctx["region"] = geo_map.get(ctx["source"], "unknown")
    print(f"  [enrich] severity={ctx['severity']} region={ctx['region']}")
    return ctx


def filter_stage(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Mark low-severity entries as filtered (non-alertable)."""
    ctx["alertable"] = ctx.get("severity", 0) >= 2
    status = "alertable" if ctx["alertable"] else "filtered"
    print(f"  [filter] {status}")
    return ctx


def format_output(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a structured JSON alert if the entry is alertable."""
    if ctx.get("alertable"):
        alert = {
            "type": "log_alert",
            "level": ctx["level"],
            "source": ctx["source"],
            "region": ctx.get("region", "unknown"),
            "message": ctx["message"],
            "severity": ctx["severity"],
            "processed_at": ctx.get("processed_at", ""),
        }
        ctx["result"] = json.dumps(alert, indent=2)
    else:
        ctx["result"] = None
    print(f"  [format] {'alert generated' if ctx['result'] else 'no alert'}")
    return ctx


# ---------------------------------------------------------------------------
# Custom pipeline agent with hooks
# ---------------------------------------------------------------------------

class LogPipelineAgent(PipelineAgent):
    """Pipeline agent that processes log entries with stage-level logging."""

    def on_pipeline_start(self, msg, ctx):
        print(f"\n--- Processing log from '{msg.sender}' ---")

    def on_pipeline_end(self, msg, ctx):
        result = ctx.get("result")
        if result:
            print(f"  Alert:\n{result}")
        else:
            print("  (no alert generated)")
        print("--- Pipeline complete ---")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Log Processing Pipeline Agent")
    print("=" * 50)

    sup = Supervisor()
    agent = LogPipelineAgent(
        "log_processor",
        stages=[parse_log, enrich, filter_stage, format_output],
    )
    agent.register(sup)

    # Simulated log entries.
    logs = [
        "ERROR 2025-01-15T10:30:00Z web-server-01 Connection refused to upstream",
        "WARN 2025-01-15T10:30:05Z api-gateway Rate limit exceeded for client 192.168.1.1",
        "INFO 2025-01-15T10:30:10Z db-primary Routine checkpoint completed successfully",
        "ERROR 2025-01-15T10:30:15Z db-primary Replication lag exceeds threshold (5s)",
    ]

    for i, log_line in enumerate(logs):
        sup.send(Message(f"log_source_{i}", "log_processor", log_line))

    processed = sup.run_once()
    print(f"\nProcessed {processed} log entries through pipeline.")
    print(f"Pipeline has {agent.stage_count} stages: "
          f"{', '.join(s.__name__ for s in agent.stages)}")


if __name__ == "__main__":
    main()
