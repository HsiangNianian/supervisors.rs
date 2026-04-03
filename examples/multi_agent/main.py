"""Multi-Agent Example -- collaborative research team.

Demonstrates the MultiAgent pattern where a group of peer agents
collaborate on a research task.  Each agent has a specialised role
(data gatherer, analyst, report writer) and they work together to
produce a final research report.

All research data is simulated -- no external dependencies required.

Usage::

    cd examples/multi_agent
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from supervisor import Agent, MultiAgent, Message, Supervisor


# ---------------------------------------------------------------------------
# Simulated knowledge base
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE: Dict[str, List[str]] = {
    "renewable energy": [
        "Solar capacity grew 26% globally in 2024.",
        "Wind energy now supplies 8% of world electricity.",
        "Battery storage costs dropped 15% year-over-year.",
        "Hydrogen fuel cells are gaining traction in heavy transport.",
    ],
    "artificial intelligence": [
        "Large language models reached 1 trillion parameters in 2024.",
        "AI adoption in healthcare diagnostics grew 40%.",
        "Regulatory frameworks for AI emerged in 30+ countries.",
        "Edge AI deployment doubled in manufacturing sectors.",
    ],
    "default": [
        "The global economy grew 3.1% in 2024.",
        "Technology sector investment reached record highs.",
        "Cross-border data flows increased by 25%.",
    ],
}


def lookup_data(topic: str) -> List[str]:
    """Simulate a data lookup from the knowledge base."""
    topic_lower = topic.lower()
    for key, facts in KNOWLEDGE_BASE.items():
        if key in topic_lower:
            return facts
    return KNOWLEDGE_BASE["default"]


# ---------------------------------------------------------------------------
# Member agents
# ---------------------------------------------------------------------------

class DataGatherer(Agent):
    """Collects relevant data points for the research topic."""

    def __init__(self) -> None:
        super().__init__("gatherer")
        self.gathered_data: List[str] = []

    def handle_message(self, msg: Message) -> None:
        facts = lookup_data(msg.content)
        self.gathered_data.extend(facts)
        print(f"  [Gatherer] Found {len(facts)} data points")
        for fact in facts:
            print(f"    - {fact}")


class Analyst(Agent):
    """Analyses gathered data and produces insights."""

    def __init__(self) -> None:
        super().__init__("analyst")
        self.insights: List[str] = []

    def handle_message(self, msg: Message) -> None:
        # Simulate analysis based on the query topic.
        topic = msg.content.lower()
        insights = []
        if "energy" in topic:
            insights = [
                "Trend: Accelerating shift from fossil fuels to renewables.",
                "Risk: Supply chain constraints for rare-earth minerals.",
                "Opportunity: Grid-scale storage solutions are investment-ready.",
            ]
        elif "intelligence" in topic or "ai" in topic:
            insights = [
                "Trend: Rapid scaling of foundation model capabilities.",
                "Risk: Regulatory uncertainty may slow enterprise adoption.",
                "Opportunity: AI-native startups are disrupting legacy sectors.",
            ]
        else:
            insights = [
                "Trend: Global markets show moderate growth trajectory.",
                "Risk: Geopolitical tensions affect trade patterns.",
                "Opportunity: Emerging markets present new growth vectors.",
            ]
        self.insights.extend(insights)
        print(f"  [Analyst] Generated {len(insights)} insights")
        for insight in insights:
            print(f"    * {insight}")


class ReportWriter(Agent):
    """Compiles findings and insights into a structured report."""

    def __init__(self) -> None:
        super().__init__("writer")
        self.reports: List[str] = []

    def handle_message(self, msg: Message) -> None:
        # Simulate report generation.
        report = (
            f"Research Report: {msg.content}\n"
            f"{'=' * 40}\n"
            f"This report provides a comprehensive analysis of {msg.content}.\n"
            f"Key findings and recommendations are summarised below.\n"
            f"(Full report would integrate data from gatherer and analyst.)\n"
        )
        self.reports.append(report)
        print(f"  [Writer] Report compiled ({len(report)} chars)")


# ---------------------------------------------------------------------------
# Custom MultiAgent with hooks
# ---------------------------------------------------------------------------

class ResearchTeam(MultiAgent):
    """Research team that coordinates data gathering, analysis, and writing."""

    def on_group_start(self, msg):
        print(f"\n{'=' * 50}")
        print(f"Research Team activated for: {msg.content}")
        print(f"{'=' * 50}")

    def on_group_end(self, msg, total_processed):
        print(f"\nTeam completed: {total_processed} tasks processed")
        print(f"{'=' * 50}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Collaborative Research Team (Multi-Agent)")
    print("=" * 55)

    sup = Supervisor()

    # Create the research team.
    gatherer = DataGatherer()
    analyst = Analyst()
    writer = ReportWriter()

    team = ResearchTeam(
        "research_team",
        members=[gatherer, analyst, writer],
        max_rounds=5,
    )
    team.register(sup)

    print(f"Team members: {', '.join(team.member_names)}")

    # Simulate research queries.
    queries = [
        "Renewable Energy Trends 2025",
        "Artificial Intelligence Market Outlook",
    ]

    for query in queries:
        sup.send(Message("research_director", "research_team", query))
        sup.run_once()

    # Print final summary.
    print("\n" + "=" * 55)
    print("Research Summary")
    print("-" * 55)
    print(f"  Data points gathered: {len(gatherer.gathered_data)}")
    print(f"  Insights produced:    {len(analyst.insights)}")
    print(f"  Reports written:      {len(writer.reports)}")

    if writer.reports:
        print(f"\nLatest report preview:")
        print(f"  {writer.reports[-1][:200]}...")


if __name__ == "__main__":
    main()
