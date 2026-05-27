"""
Mock Chatbot — simulates an e-commerce customer support AI assistant.

In a real project this would call the actual chatbot API (Salesforce Agentforce,
or any conversational AI endpoint). Here we return controlled responses so we can:
  1. Test that our DeepEval metrics correctly score good responses highly
  2. Test that our metrics correctly flag bad/hallucinated responses
  3. Reproduce specific edge cases reliably

Each method returns a dict with:
  - response: what the chatbot said
  - escalated: whether it routed to a human agent
  - tools_called: which internal tools/actions the bot invoked
"""

class MockChatbot:

    def respond(self, user_input: str, scenario: str = "good") -> dict:
        """
        scenario = "good"  → correct, helpful, well-routed response
        scenario = "bad"   → hallucinated or irrelevant response  
        scenario = "edge"  → ambiguous or borderline response
        """
        handlers = {
            "damaged_item": self._damaged_item,
            "order_status": self._order_status,
            "missing_order": self._missing_order,
            "escalation":    self._escalation,
        }

        # Detect intent from the input (simple keyword matching — 
        # in a real bot this would be an NLP classifier or LLM routing)
        intent = self._detect_intent(user_input)
        handler = handlers.get(intent, self._default)
        return handler(scenario)

    def _detect_intent(self, text: str) -> str:
        text = text.lower()
        if any(word in text for word in ["damaged", "broken", "cracked", "ruined"]):
            return "damaged_item"
        if any(word in text for word in ["where is", "status", "tracking"]):
            return "order_status"
        if any(word in text for word in ["missing", "not arrived", "never received"]):
            return "missing_order"
        if any(word in text for word in ["agent", "human", "speak to someone", "urgent"]):
            return "escalation"
        return "default"

    # --- Damaged item scenarios ---

    def _damaged_item(self, scenario: str) -> dict:
        if scenario == "good":
            return {
                "response": (
                    "I'm sorry to hear your item arrived damaged. "
                    "Please upload photos of the damage and your order number "
                    "and I'll arrange a replacement or refund right away. "
                    "If the damage looks severe, I'll connect you with a specialist."
                ),
                "escalated": False,
                "tools_called": ["fetch_order_details", "initiate_return_flow"]
            }
        if scenario == "bad":
            return {
                "response": (
                    "Your order was delivered successfully on Monday at 2pm "
                    "and our records show it was in perfect condition."
                    # Hallucination — the bot invented delivery details
                    # and ignored the damage report entirely
                ),
                "escalated": False,
                "tools_called": []
            }
        if scenario == "edge":
            return {
                "response": "Can you tell me more about the issue?",
                # Ambiguous — not wrong, but unhelpful and missing next steps
                "escalated": False,
                "tools_called": []
            }

    # --- Order status scenarios ---

    def _order_status(self, scenario: str) -> dict:
        if scenario == "good":
            return {
                "response": (
                    "Your order #45231 is currently out for delivery and "
                    "should arrive by 6pm today. You can track it live here: "
                    "[tracking link]. Let me know if anything changes."
                ),
                "escalated": False,
                "tools_called": ["fetch_order_details", "fetch_tracking_info"]
            }
        if scenario == "bad":
            return {
                "response": "I don't have access to order information right now.",
                # Bad — the bot should have called fetch_order_details
                # Failing to use available tools is a real agentic failure mode
                "escalated": False,
                "tools_called": []
            }
        if scenario == "edge":
            return {
                "response": "Your order is being processed.",
                # Vague — technically not wrong but not helpful
                "escalated": False,
                "tools_called": ["fetch_order_details"]
            }

    # --- Missing order scenarios ---

    def _missing_order(self, scenario: str) -> dict:
        if scenario == "good":
            return {
                "response": (
                    "I'm sorry your order hasn't arrived. I've checked your tracking "
                    "and it appears there may be a delivery exception. I'm escalating "
                    "this to our fulfilment team who will contact you within 2 hours."
                ),
                "escalated": True,
                "tools_called": ["fetch_order_details", "fetch_tracking_info", "escalate_to_agent"]
            }
        if scenario == "bad":
            return {
                "response": "Your order was delivered. Please check with your neighbours.",
                # Hallucination — invented a delivery confirmation
                "escalated": False,
                "tools_called": []
            }
        if scenario == "edge":
            return {
                "response": (
                    "I can see your order left our warehouse 8 days ago. "
                    "Delivery usually takes 5-7 days. Please wait another day."
                ),
                # Edge case — order is already overdue, should escalate but didn't
                "escalated": False,
                "tools_called": ["fetch_order_details"]
            }

    # --- Escalation scenarios ---

    def _escalation(self, scenario: str) -> dict:
        if scenario == "good":
            return {
                "response": (
                    "I understand this is urgent. Let me connect you with a "
                    "human agent right away. Average wait time is 3 minutes."
                ),
                "escalated": True,
                "tools_called": ["escalate_to_agent"]
            }
        if scenario == "bad":
            return {
                "response": "I can help you with that. What is your issue?",
                # Bad — user explicitly asked for a human, bot ignored it
                "escalated": False,
                "tools_called": []
            }
        if scenario == "edge":
            return {
                "response": "Let me try to help you first before escalating.",
                # Edge — deflecting escalation request, debatable whether acceptable
                "escalated": False,
                "tools_called": []
            }

    def _default(self, scenario: str) -> dict:
        return {
            "response": "I'm here to help with order issues, damages, and delivery queries.",
            "escalated": False,
            "tools_called": []
        }