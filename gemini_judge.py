import os
from google import genai
from google.genai import types
from deepeval.models.base_model import DeepEvalBaseLLM

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

class GeminiJudge(DeepEvalBaseLLM):
    """
    Adapter that lets DeepEval use Gemini as its evaluation judge.
    Implements the interface DeepEval expects: generate() and a_generate().
    """

    def load_model(self):
        return "gemini-3.5-flash"

    def generate(self, prompt: str) -> str:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )
        return response.text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "gemini-3.5-flash"