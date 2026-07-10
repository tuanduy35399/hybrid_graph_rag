from llm.gemini import GeminiLLM
from llm.prompts import PROMPT, build_prompt


class AnswerGenerator:

    def __init__(self):
        self.llm = GeminiLLM()

    def generate(self, question: str, context: str) -> str:

        prompt = build_prompt(
            context=context,
            question=question,
        )

        full_prompt = (
            PROMPT
            + "\n\n"
            + prompt
        )

        return self.llm.generate(full_prompt)