class PlannerLLMClient:
    def __init__(self, model):
        self.model = model

    def generate(self, prompt: str) -> object:
        response = self.model.generate_content(prompt)
        return getattr(response, "text", None)
