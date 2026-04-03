"""Composite Example -- DevOps incident response.

Combines the SupervisorAgent and MultiAgent patterns.  A top-level
supervisor (the Incident Commander) routes incoming incidents to
specialised teams.  Each team is a MultiAgent group whose members
collaborate to resolve the incident.

All actions are simulated -- no external dependencies required.

Usage::

    cd examples/composite
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from supervisor import Agent, MultiAgent, SupervisorAgent, Message, Supervisor


# ---------------------------------------------------------------------------
# Infrastructure team members
# ---------------------------------------------------------------------------

class InfraMonitor(Agent):
    """Monitors infrastructure health metrics."""

    def __init__(self) -> None:
        super().__init__("infra_monitor")
        self.findings: List[str] = []

    def handle_message(self, msg: Message) -> None:
        # Simulate metric checks.
        findings = [
            f"CPU usage: 92% (critical)",
            f"Memory: 78% (warning)",
            f"Disk I/O: elevated latency detected",
        ]
        self.findings.extend(findings)
        print(f"  [InfraMonitor] Checked {len(findings)} metrics")
        for f in findings:
            print(f"    - {f}")


class InfraFixer(Agent):
    """Applies infrastructure remediation actions."""

    def __init__(self) -> None:
        super().__init__("infra_fixer")
        self.actions: List[str] = []

    def handle_message(self, msg: Message) -> None:
        actions = [
            "Scaled up auto-scaling group by 2 instances",
            "Triggered garbage collection on affected nodes",
            "Redirected traffic away from degraded region",
        ]
        self.actions.extend(actions)
        print(f"  [InfraFixer] Applied {len(actions)} remediation actions")
        for a in actions:
            print(f"    + {a}")


class InfraVerifier(Agent):
    """Verifies that infrastructure remediation was successful."""

    def __init__(self) -> None:
        super().__init__("infra_verifier")
        self.verified = False

    def handle_message(self, msg: Message) -> None:
        self.verified = True
        print(f"  [InfraVerifier] Verification complete: systems recovering")


# ---------------------------------------------------------------------------
# Application team members
# ---------------------------------------------------------------------------

class AppLogger(Agent):
    """Analyses application logs for error patterns."""

    def __init__(self) -> None:
        super().__init__("app_logger")
        self.errors_found: List[str] = []

    def handle_message(self, msg: Message) -> None:
        errors = [
            "NullPointerException in PaymentService.process()",
            "TimeoutError in DatabasePool.getConnection()",
            "RetryExhausted in ExternalAPI.call()",
        ]
        self.errors_found.extend(errors)
        print(f"  [AppLogger] Found {len(errors)} error patterns")
        for e in errors:
            print(f"    ! {e}")


class AppDeployer(Agent):
    """Deploys hotfixes or rollbacks for application issues."""

    def __init__(self) -> None:
        super().__init__("app_deployer")
        self.deployments: List[str] = []

    def handle_message(self, msg: Message) -> None:
        deployment = "Rolled back to last stable release v2.3.1"
        self.deployments.append(deployment)
        print(f"  [AppDeployer] {deployment}")


class AppTester(Agent):
    """Runs smoke tests after deployment changes."""

    def __init__(self) -> None:
        super().__init__("app_tester")
        self.test_results: List[Dict[str, Any]] = []

    def handle_message(self, msg: Message) -> None:
        results = {
            "total": 42,
            "passed": 41,
            "failed": 1,
            "status": "mostly_passing",
        }
        self.test_results.append(results)
        print(f"  [AppTester] Smoke tests: {results['passed']}/{results['total']} passed")


# ---------------------------------------------------------------------------
# Multi-agent teams
# ---------------------------------------------------------------------------

class InfraTeam(MultiAgent):
    """Infrastructure response team."""

    def on_group_start(self, msg):
        print(f"\n  --- Infrastructure Team activated ---")

    def on_group_end(self, msg, total):
        print(f"  --- Infrastructure Team done ({total} actions) ---")


class AppTeam(MultiAgent):
    """Application response team."""

    def on_group_start(self, msg):
        print(f"\n  --- Application Team activated ---")

    def on_group_end(self, msg, total):
        print(f"  --- Application Team done ({total} actions) ---")


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------

def incident_router(msg: Message) -> str:
    """Route incidents to the appropriate team.

    Simple keyword-based routing:
    - Infrastructure keywords -> infra_team
    - Application keywords -> app_team
    """
    content = msg.content.lower()
    infra_keywords = {"cpu", "memory", "disk", "network", "server", "node", "infra"}
    app_keywords = {"error", "exception", "timeout", "deploy", "api", "service", "app"}

    infra_score = sum(1 for kw in infra_keywords if kw in content)
    app_score = sum(1 for kw in app_keywords if kw in content)

    if infra_score >= app_score:
        return "infra_team"
    return "app_team"


# ---------------------------------------------------------------------------
# Incident Commander (SupervisorAgent)
# ---------------------------------------------------------------------------

class IncidentCommander(SupervisorAgent):
    """Top-level supervisor that routes incidents to specialised teams."""

    def on_delegate(self, msg, target):
        print(f"\n  [Commander] Routing incident to '{target}'")

    def on_sub_agents_complete(self, processed):
        print(f"\n  [Commander] Team processed {processed} task(s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("DevOps Incident Response System (Composite)")
    print("=" * 55)

    sup = Supervisor()

    # Build infrastructure team.
    infra_monitor = InfraMonitor()
    infra_fixer = InfraFixer()
    infra_verifier = InfraVerifier()

    infra_team = InfraTeam(
        "infra_team",
        members=[infra_monitor, infra_fixer, infra_verifier],
        max_rounds=3,
    )

    # Build application team.
    app_logger = AppLogger()
    app_deployer = AppDeployer()
    app_tester = AppTester()

    app_team = AppTeam(
        "app_team",
        members=[app_logger, app_deployer, app_tester],
        max_rounds=3,
    )

    # Create the incident commander.
    commander = IncidentCommander(
        "incident_commander",
        router=incident_router,
    )
    commander.add_sub_agent(infra_team)
    commander.add_sub_agent(app_team)
    commander.register(sup)

    print(f"Teams: {', '.join(commander.sub_agent_names)}")
    print(f"  Infra team: {', '.join(infra_team.member_names)}")
    print(f"  App team:   {', '.join(app_team.member_names)}")
    print("-" * 55)

    # Simulate incidents.
    incidents = [
        (
            "monitoring_system",
            "CRITICAL: CPU usage at 95% on production server cluster, "
            "memory pressure increasing, disk I/O latency spike detected"
        ),
        (
            "error_tracker",
            "HIGH: Multiple TimeoutError exceptions in PaymentService, "
            "API error rate at 5%, service degradation reported"
        ),
    ]

    for sender, description in incidents:
        print(f"\n{'#' * 55}")
        print(f"INCIDENT from '{sender}':")
        print(f"  {description[:70]}...")
        sup.send(Message(sender, "incident_commander", description))
        sup.run_once()

    # Final summary.
    print(f"\n{'=' * 55}")
    print("Incident Response Summary")
    print("-" * 55)
    print(f"  Infrastructure findings:  {len(infra_monitor.findings)}")
    print(f"  Infrastructure actions:   {len(infra_fixer.actions)}")
    print(f"  Infrastructure verified:  {infra_verifier.verified}")
    print(f"  Application errors found: {len(app_logger.errors_found)}")
    print(f"  Application deployments:  {len(app_deployer.deployments)}")
    print(f"  Application test runs:    {len(app_tester.test_results)}")


if __name__ == "__main__":
    main()
