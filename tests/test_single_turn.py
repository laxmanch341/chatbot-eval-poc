# tests/test_single_turn.py
import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemini_judge import GeminiJudge
from mock_chatbot import MockChatbot

bot = MockChatbot()
judge = GeminiJudge()

# --- Test 1: Good response should pass relevancy ---

def test_damaged_item_response_is_relevant():
    """
    The bot's response to a damaged item report should be
    relevant to the user's actual question.
    Good scenario should score above threshold.
    """
    result = bot.respond("My item arrived damaged", scenario="good")

    test_case = LLMTestCase(
        input="My item arrived damaged",
        actual_output=result["response"]
    )

    metric = AnswerRelevancyMetric(threshold=0.7, model=judge)
    metric.measure(test_case)

    print(f"\nScore: {metric.score}")
    print(f"Reason: {metric.reason}")

    assert metric.score >= 0.7, f"Expected relevant response. Got score: {metric.score}. Reason: {metric.reason}"


# --- Test 2: Bad response should fail relevancy ---

def test_damaged_item_hallucination_fails_relevancy():
    """
    The bad scenario hallucinates a successful delivery.
    This should score LOW on relevancy because it ignores
    the damage report entirely.
    This test PASSES when the metric correctly identifies a BAD response.
    """
    result = bot.respond("My item arrived damaged", scenario="bad")

    test_case = LLMTestCase(
        input="My item arrived damaged",
        actual_output=result["response"]
    )

    metric = AnswerRelevancyMetric(threshold=0.7, model=judge)
    metric.measure(test_case)

    print(f"\nScore: {metric.score}")
    print(f"Reason: {metric.reason}")

    # We EXPECT this to fail — the score should be LOW
    # This validates that our metric can detect bad responses
    assert metric.score < 0.7, f"Expected low score for hallucinated response. Got: {metric.score}"


# --- Test 3: Faithfulness — did the bot invent facts? ---

def test_order_status_faithfulness():
    """
    FaithfulnessMetric checks whether the response stays faithful
    to the provided context (retrieval_context).
    
    retrieval_context = the facts the bot SHOULD have used.
    If the bot says things not in the context, it's hallucinating.
    """
    result = bot.respond("Where is my order?", scenario="good")

    test_case = LLMTestCase(
        input="Where is my order?",
        actual_output=result["response"],
        # This is the ground truth context the bot should base its answer on
        # In a real system this would come from your order database / retrieval layer
        retrieval_context=[
            "Order #45231 is out for delivery.",
            "Estimated delivery time is 6pm today.",
            "A live tracking link is available."
        ]
    )

    metric = FaithfulnessMetric(threshold=0.7, model=judge)
    metric.measure(test_case)

    print(f"\nScore: {metric.score}")
    print(f"Reason: {metric.reason}")

    assert metric.score >= 0.7, f"Response not faithful to context. Reason: {metric.reason}"


# --- Test 4: Faithfulness failure — bad response invents facts ---

def test_hallucination_detected_in_bad_response():
    """
    The bad scenario for missing order claims delivery happened.
    That claim is NOT in the retrieval context.
    FaithfulnessMetric should detect this as hallucination.
    """
    result = bot.respond("My order never arrived", scenario="bad")

    test_case = LLMTestCase(
        input="My order never arrived",
        actual_output=result["response"],
        retrieval_context=[
            "Order #45231 shows a delivery exception.",
            "Last scan was at the sorting facility 3 days ago.",
            "No confirmed delivery on record."
        ]
    )

    metric = FaithfulnessMetric(threshold=0.7, model=judge)
    metric.measure(test_case)

    print(f"\nScore: {metric.score}")
    print(f"Reason: {metric.reason}")

    # Expect LOW score — bot claimed delivery happened, context says it didn't
    assert metric.score < 0.7, f"Expected hallucination to be detected. Got score: {metric.score}"