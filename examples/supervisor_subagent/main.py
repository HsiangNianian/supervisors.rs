"""Supervisor + SubAgent Example -- content moderation system.

Demonstrates the SupervisorAgent pattern where a parent agent routes
incoming content to specialised child agents (text moderator, image
moderator, spam detector) based on content type.

All moderation logic is rule-based -- no external dependencies required.

Usage::

    cd examples/supervisor_subagent
    python main.py
"""

from __future__ import annotations
from supervisors import Agent, SupervisorAgent, Message, Supervisor

import sys
from pathlib import Path
from typing import Any, Dict

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root / "src"))


# ---------------------------------------------------------------------------
# Sub-agent implementations
# ---------------------------------------------------------------------------


class TextModerator(Agent):
    """Moderates text content using rule-based keyword matching."""

    # Simulated list of flagged keywords.
    FLAGGED_KEYWORDS = {"spam", "scam", "offensive", "banned"}

    def __init__(self) -> None:
        super().__init__("text_moderator")
        self.results: list[Dict[str, Any]] = []

    def handle_message(self, msg: Message) -> None:
        text = msg.content.lower()
        flagged = [kw for kw in self.FLAGGED_KEYWORDS if kw in text]
        verdict = "rejected" if flagged else "approved"
        result = {
            "type": "text",
            "verdict": verdict,
            "flagged_keywords": flagged,
            "original": msg.content[:80],
        }
        self.results.append(result)
        print(f"  [TextMod] {verdict} (flagged: {
              flagged if flagged else 'none'})")


class ImageModerator(Agent):
    """Moderates image URLs using simulated analysis."""

    def __init__(self) -> None:
        super().__init__("image_moderator")
        self.results: list[Dict[str, Any]] = []

    def handle_message(self, msg: Message) -> None:
        # Simulated image moderation based on URL patterns.
        url = msg.content.strip()
        is_suspicious = any(
            pattern in url.lower() for pattern in ["unsafe", "explicit", "banned"]
        )
        verdict = "rejected" if is_suspicious else "approved"
        result = {
            "type": "image",
            "verdict": verdict,
            "url": url[:80],
        }
        self.results.append(result)
        print(f"  [ImageMod] {verdict} -- {url[:60]}")


class SpamDetector(Agent):
    """Detects spam content using simple heuristics."""

    def __init__(self) -> None:
        super().__init__("spam_detector")
        self.results: list[Dict[str, Any]] = []

    def handle_message(self, msg: Message) -> None:
        text = msg.content.lower()
        spam_signals = 0
        if text.count("!") > 3:
            spam_signals += 1
        if "buy now" in text or "free money" in text:
            spam_signals += 2
        if len(text) > 200 and text.upper() == text:
            spam_signals += 1
        is_spam = spam_signals >= 2
        verdict = "spam" if is_spam else "not_spam"
        result = {
            "type": "spam_check",
            "verdict": verdict,
            "score": spam_signals,
        }
        self.results.append(result)
        print(f"  [SpamDet] {verdict} (score: {spam_signals})")


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------


def content_router(msg: Message) -> str:
    """Route content to the appropriate moderation sub-agent.

    Determines the content type from the message and returns the name
    of the sub-agent that should handle it.
    """
    content = msg.content.strip().lower()
    if content.startswith(("http://", "https://")) and any(
        ext in content for ext in [".jpg", ".png", ".gif", ".webp"]
    ):
        return "image_moderator"
    if content.count("!") > 3 or "buy now" in content or "free money" in content:
        return "spam_detector"
    return "text_moderator"


# ---------------------------------------------------------------------------
# Custom supervisor with hooks
# ---------------------------------------------------------------------------


class ModerationManager(SupervisorAgent):
    """Supervisor that manages content moderation sub-agents."""

    def on_delegate(self, msg, target):
        print(f"\n  [Router] Routing to '{target}'")

    def on_sub_agents_complete(self, processed):
        print(f"  [Manager] Sub-agents processed {processed} item(s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Content Moderation System (Supervisor + SubAgent)")
    print("=" * 55)

    sup = Supervisor()

    # Create sub-agents.
    text_mod = TextModerator()
    image_mod = ImageModerator()
    spam_det = SpamDetector()

    # Create supervisor agent with routing.
    manager = ModerationManager("moderation_manager", router=content_router)
    manager.add_sub_agent(text_mod)
    manager.add_sub_agent(image_mod)
    manager.add_sub_agent(spam_det)
    manager.register(sup)

    print(f"\nSub-agents: {', '.join(manager.sub_agent_names)}")
    print("-" * 55)

    # Simulate incoming content for moderation.
    content_items = [
        ("user_1", "This is a normal comment about the product."),
        ("user_2", "BUY NOW!!! Free money!!! Click here!!! Amazing deal!!!!"),
        ("user_3", "https://cdn.example.com/images/photo.jpg"),
        ("user_4", "This post contains offensive language that should be banned."),
        ("user_5", "https://cdn.example.com/unsafe_explicit_content.png"),
        ("user_6", "Great product, highly recommend it to everyone."),
    ]

    for sender, content in content_items:
        print(f"\n--- Content from '{sender}' ---")
        print(f"  Content: {content[:70]}{'...' if len(content) > 70 else ''}")
        sup.send(Message(sender, "moderation_manager", content))
        sup.run_once()

    # Print summary.
    print("\n" + "=" * 55)
    print("Moderation Summary")
    print("-" * 55)
    print(f"  Text moderation:  {len(text_mod.results)} items")
    for r in text_mod.results:
        print(f"    {r['verdict']:10} | {r['original'][:50]}")
    print(f"  Image moderation: {len(image_mod.results)} items")
    for r in image_mod.results:
        print(f"    {r['verdict']:10} | {r['url'][:50]}")
    print(f"  Spam detection:   {len(spam_det.results)} items")
    for r in spam_det.results:
        print(f"    {r['verdict']:10} | score={r['score']}")


if __name__ == "__main__":
    main()
