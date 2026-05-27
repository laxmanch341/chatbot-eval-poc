# tests/test_multi_turn.py

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.metrics import TurnRelevancyMetric,ConversationCompletenessMetric
from gemini_judge import GeminiJudge

judge = GeminiJudge()

# ============================================================
# TEST 1 — Good conversation
# Bot remembers order number and name throughout.
# Every response follows naturally from what came before.
# This test should PASS with a HIGH score.
# ============================================================

def test_good_conversation_retains_context():
    """
    A well-behaved bot remembers what the user said earlier.
    It never asks for information the user already provided.
    The judge reads all 6 turns and scores coherence across them.
    """

    conversation = ConversationalTestCase(
        turns=[
            # Turn 1 — user introduces themselves and gives order number
            Turn(
                role="user",
                content="Hi, my name is Laxman and my order number is 45231."
            ),
            # Turn 2 — bot acknowledges both pieces of information
            Turn(
                role="assistant",
                content="Hi Laxman! I can see your order #45231. How can I help you today?"
            ),
            # Turn 3 — user explains the problem
            Turn(
                role="user",
                content="The item arrived damaged. The box was completely crushed."
            ),
            # Turn 4 — CRITICAL TURN
            # Good bot uses the order number it already has.
            # It does NOT ask for the order number again.
            # It directly moves to resolving the issue.
            Turn(
                role="assistant",
                content=(
                    "I'm really sorry to hear that Laxman. "
                    "I've pulled up order #45231 and I can see it was "
                    "delivered yesterday. I'm initiating a replacement "
                    "right away. Can you upload a photo of the damage?"
                )
            ),
            # Turn 5 — user confirms
            Turn(
                role="user",
                content="Yes I'll upload the photo now."
            ),
            # Turn 6 — bot closes the loop, still remembers context
            Turn(
                role="assistant",
                content=(
                    "Thank you Laxman. Once I receive the photo for "
                    "order #45231 I'll process the replacement within "
                    "24 hours. Is there anything else I can help you with?"
                )
            ),
        ]
    )

    metric = TurnRelevancyMetric(threshold=0.7, model=judge)
    metric.measure(conversation)

    print(f"\nScore: {metric.score}")
    print(f"Reason: {metric.reason}")

    assert metric.score >= 0.7, (
        f"Good conversation should retain context. "
        f"Score: {metric.score}. Reason: {metric.reason}"
    )


# ============================================================
# TEST 2 — Bad conversation — context retention failure
# This is EXACTLY what you described:
# Bot asks for order number again after user already gave it.
# This test should PASS when metric gives a LOW score —
# proving the metric correctly detected the failure.
# ============================================================

def test_bad_conversation_fails_context_retention():
    """
    The bot asks for the order number in Turn 4 even though
    the user provided it in Turn 1. This is a context retention
    failure. The judge should score this LOW because Turn 4
    makes no sense given what happened in Turn 1.

    This test PASSES when the score is LOW —
    meaning the metric successfully detected the problem.
    """

    conversation = ConversationalTestCase(
        turns=[
            # Turn 1 — user gives name and order number
            Turn(
                role="user",
                content="Hi, my name is Laxman and my order number is 45231."
            ),
            # Turn 2 — bot acknowledges
            Turn(
                role="assistant",
                content="Hi Laxman! I'm here to help. What seems to be the issue?"
            ),
            # Turn 3 — user explains the problem
            Turn(
                role="user",
                content="The item arrived damaged. The box was completely crushed."
            ),
            # Turn 4 — FAILURE POINT
            # Bot asks for the order number again.
            # User already gave this in Turn 1.
            # This is the exact failure you described.
            Turn(
                role="assistant",
                content=(
                    "I'm sorry to hear that. "
                    "Could you please provide your order number "
                    "so I can look into this for you?"
                )
            ),
            # Turn 5 — frustrated user repeats themselves
            Turn(
                role="user",
                content="I already told you, it's 45231."
            ),
            # Turn 6 — bot continues as if nothing happened
            Turn(
                role="assistant",
                content="Thank you for providing that. Let me check order #45231 for you."
            ),
        ]
    )

    metric = TurnRelevancyMetric(threshold=0.7, model=judge)
    metric.measure(conversation)

    print(f"\nScore: {metric.score}")
    print(f"Reason: {metric.reason}")

    # We EXPECT a low score here —
    # the metric should detect that Turn 4 ignored Turn 1
    assert metric.score < 0.7, (
        f"Context retention failure should score low. "
        f"Score: {metric.score}. Reason: {metric.reason}"
    )


# ============================================================
# TEST 3 — Edge case — partial context retention
# Bot remembers the order number but forgets the user's name.
# Scores somewhere in the middle — tests metric sensitivity.
# ============================================================

def test_edge_case_partial_context_retention():
    """
    The bot remembers the order number but keeps forgetting
    the user's name — addressing them generically instead.
    This is a softer failure — not catastrophic but noticeable.
    We expect a middling score — not great, not terrible.
    """

    conversation = ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="Hi, my name is Laxman and my order number is 45231."
            ),
            Turn(
                role="assistant",
                content="Hello! I can see order #45231. What is the issue?"
                # Already forgot the name — didn't say "Hi Laxman"
            ),
            Turn(
                role="user",
                content="The item arrived damaged."
            ),
            Turn(
                role="assistant",
                content=(
                    "I'm sorry to hear about the damage to order #45231. "
                    "Could you please share a photo of the damaged item?"
                    # Remembers order number — good.
                    # Still not using the name — partial failure.
                )
            ),
            Turn(
                role="user",
                content="Sure, uploading now."
            ),
            Turn(
                role="assistant",
                content=(
                    "Thank you. I'll process a replacement for order #45231 "
                    "within 24 hours."
                    # Order number correct throughout.
                    # Name never used after Turn 1 — consistent partial failure.
                )
            ),
        ]
    )

    metric = TurnRelevancyMetric(threshold=0.7, model=judge)
    metric.measure(conversation)

    print(f"\nScore: {metric.score}")
    print(f"Reason: {metric.reason}")

    # We just observe the score here — no strict pass/fail assertion.
    # This is intentional — edge cases teach you where your threshold
    # should be set. If this scores 0.75, maybe 0.7 is too lenient.
    # If it scores 0.4, the metric is being harsh about name usage.
    print(f"\nEdge case score: {metric.score} — use this to calibrate your threshold")