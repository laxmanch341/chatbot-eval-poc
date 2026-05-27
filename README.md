# Chatbot Evaluation POC — DeepEval + Gemini

A proof-of-concept LLM evaluation framework for testing conversational AI
workflows using [DeepEval](https://deepeval.com) with Google Gemini 1.5 Flash
as the judge model.

Built as a practical exploration of **LLM-as-a-Judge** evaluation methodology —
the approach of using an LLM to evaluate another LLM's outputs on dimensions
like relevance, faithfulness, and contextual accuracy.

---

## Why this project exists

Traditional test automation relies on deterministic assertions:
assert response == "expected string"

This breaks for LLM outputs because the same input can produce semantically
correct but textually different responses every time.

This project explores how to evaluate LLM responses using **metric-based
evaluation** instead — scoring responses on properties like:

- Is this response relevant to what was asked?
- Did the bot invent facts not present in the context (hallucination)?
- Does the bot remember context across a multi-turn conversation?
- Did the agent call the right tools in the right order?
- Did escalation trigger correctly when required?

---

## Evaluation coverage

| Test File | What it evaluates | Metrics used |
|---|---|---|
| `test_single_turn.py` | Relevancy and hallucination detection | AnswerRelevancy, Faithfulness |
| `test_multi_turn.py` | Context retention across conversation turns | ConversationalRelevancy *(in progress)* |
| `test_tool_calling.py` | Agent tool invocation correctness | Custom assertion *(in progress)* |
| `test_escalation.py` | Escalation routing logic | Custom assertion *(in progress)* |

---

## Tech stack

| Layer | Tool |
|---|---|
| Evaluation framework | DeepEval 4.x |
| LLM judge | Google Gemini 1.5 Flash |
| Test runner | pytest |
| Language | Python 3.12 |
| Mock chatbot | Custom (simulates e-commerce support AI) |

---

## Mock chatbot

The project includes a mock chatbot (`mock_chatbot.py`) that simulates an
e-commerce customer support AI assistant handling:

- Damaged item reports
- Order status queries
- Missing order escalations
- Human agent handoff requests

Each scenario has three response variants — **good**, **bad**, and **edge** —
so tests can validate both that good responses pass and that hallucinated or
incorrect responses are correctly flagged.

This separation is intentional: if an evaluator gives a high score to a
hallucinated response, the evaluator itself is broken. The bad/edge scenarios
exist to **test the test framework**.

---

## Setup

```bash
# Clone
git clone https://github.com/laxmanch341/chatbot-eval-poc.git
cd chatbot-eval-poc

# Virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux

# Dependencies
pip install -r requirements.txt

# API key (get free key at https://aistudio.google.com/apikey)
set GOOGLE_API_KEY=your-key-here     # Windows
export GOOGLE_API_KEY=your-key-here  # Mac/Linux
```

## Running the tests

```bash
# All tests
python -m pytest tests/ -v -s

# Single file
python -m pytest tests\test_single_turn.py -v -s
```

---

## Key concepts demonstrated

**LLM-as-a-Judge** — using an LLM (Gemini) to evaluate another LLM's output
instead of hardcoded assertions. The judge scores responses and provides a
human-readable reason for its score.

**Hallucination detection** — using `FaithfulnessMetric` to check whether a
response stays faithful to provided context, or invents facts not present in
the source.

**Adapter pattern** — `GeminiJudge` wraps the Google Gemini API to conform to
DeepEval's `DeepEvalBaseLLM` interface, making any model pluggable as the judge.

**Good/bad/edge test design** — each scenario has three response variants.
Tests that should pass *and* tests that should fail are both included —
validating the metric's ability to distinguish quality levels.

---

## Project status

Active — building out evaluation coverage progressively.

- [x] Environment setup and Gemini integration
- [x] Mock chatbot with good/bad/edge scenarios
- [x] Single-turn evaluation (relevancy + faithfulness)
- [ ] Multi-turn context retention tests
- [ ] Tool-calling validation tests
- [ ] Escalation routing tests
- [ ] GitHub Actions CI pipeline
