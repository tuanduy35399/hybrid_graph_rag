from src.rag.answering.prompt import build_grounded_answer_prompt


class GeminiAnswerGeneratorClient:
    def __init__(self, model):
        self.model = model

    def generate(self, answer_context) -> str:
        prompt = build_grounded_answer_prompt(answer_context)
        response = self.model.generate_content(prompt)
        return response.text
