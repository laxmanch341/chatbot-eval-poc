from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from gemini_judge import GeminiJudge

# Instantiate the judge once and reuse it across tests
judge = GeminiJudge()

test_case = LLMTestCase(
    input="My order is damaged, what should I do?",
    actual_output="Please contact our support team with your order number and photos of the damage.",
)

# Pass model=judge to tell DeepEval to use Gemini instead of OpenAI
metric = AnswerRelevancyMetric(threshold=0.7, model=judge)
metric.measure(test_case)

print(f"Score:  {metric.score}")
print(f"Passed: {metric.score >= 0.7}")
print(f"Reason: {metric.reason}")