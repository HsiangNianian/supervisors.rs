"""Loop Agent Example -- simulated customer support with iterative reasoning.

Demonstrates the LoopAgent pattern where an agent processes each incoming
ticket through multiple reasoning steps: parsing, classifying, drafting a
response, reviewing, and finalising.

All LLM calls are simulated locally -- no API keys required.

Usage::

    cd examples/loop
    python main.py
"""

from __future__ import annotations
from supervisors import LoopAgent, Message, Supervisor

import sys
from pathlib import Path
from typing import Any, Dict

# Ensure the package is importable when running from the examples directory.
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root / "src"))


# ---------------------------------------------------------------------------
# Simulated LLM helper
# ---------------------------------------------------------------------------


def simulated_llm(prompt: str) -> str:
    """Simulate an LLM response based on keyword matching.

    In a real application this would call an OpenAI, Anthropic, or local
    model endpoint.
    """
    prompt_lower = prompt.lower()
    if "classify" in prompt_lower:
        if "refund" in prompt_lower or "money" in prompt_lower:
            return "billing"
        if "bug" in prompt_lower or "error" in prompt_lower or "crash" in prompt_lower:
            return "technical"
        return "general"
    if prompt_lower.startswith("review"):
        return "approved"
    if "draft" in prompt_lower:
        return (
            "Thank you for reaching out. We have reviewed your request and "
            "will take appropriate action. Our team typically responds within "
            "24 hours for detailed follow-up."
        )
    return "acknowledged"


# ---------------------------------------------------------------------------
# Support Agent (LoopAgent subclass)
# ---------------------------------------------------------------------------


class SupportAgent(LoopAgent):
    """Customer support agent that reasons through tickets step by step.

    Each iteration of the loop advances through the following phases:
    1. parse   -- extract key information from the ticket
    2. classify -- determine the ticket category
    3. draft   -- formulate a response
    4. review  -- quality-check the draft
    5. done    -- finalise and output the response
    """

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        phase = state.get("phase", "parse")

        if phase == "parse":
            # Extract basic information from the raw input.
            raw = state.get("input", "")
            state["ticket_text"] = raw.strip()
            state["phase"] = "classify"
            print(f"  [parse] Extracted ticket text ({len(raw)} chars)")

        elif phase == "classify":
            category = simulated_llm(
                f"Classify this support ticket: {state['ticket_text']}"
            )
            state["category"] = category
            state["phase"] = "draft"
            print(f"  [classify] Category: {category}")

        elif phase == "draft":
            draft = simulated_llm(
                f"Draft a response for a {state['category']} ticket: "
                f"{state['ticket_text']}"
            )
            state["draft"] = draft
            state["phase"] = "review"
            print(f"  [draft] Response drafted ({len(draft)} chars)")

        elif phase == "review":
            verdict = simulated_llm(f"Review this draft: {state['draft']}")
            state["review_result"] = verdict
            if verdict == "approved":
                state["phase"] = "done"
                print(f"  [review] Draft approved")
            else:
                # Rejected -- go back to drafting.
                state["phase"] = "draft"
                print(f"  [review] Draft rejected, redrafting...")

        elif phase == "done":
            state["done"] = True
            state["final_response"] = state["draft"]
            print(f"  [done] Final response ready")

        return state

    def on_loop_start(self, msg: "Message", state: Dict[str, Any]) -> None:
        print(f"\n--- Processing ticket from '{msg.sender}' ---")

    def on_loop_end(
        self, msg: "Message", state: Dict[str, Any], iterations: int
    ) -> None:
        print(f"--- Completed in {iterations} iterations ---")
        response = state.get("final_response", "(no response)")
        print(f"\nFinal response to '{msg.sender}':\n  {response}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Customer Support Loop Agent")
    print("=" * 50)

    sup = Supervisor()
    agent = SupportAgent("support", max_iterations=10)
    agent.register(sup)

    # Simulate incoming support tickets.
    tickets = [
        ("alice", "I was charged twice for my subscription. Please refund."),
        ("bob", "The app crashes when I try to upload a photo. Error code 500."),
        ("charlie", "How do I change my notification settings?"),
    ]

    for sender, content in tickets:
        sup.send(Message(sender, "support", content))

    processed = sup.run_once()
    print(f"\nProcessed {processed} ticket(s).")


if __name__ == "__main__":
    main()
