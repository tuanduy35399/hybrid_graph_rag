from .client import LLMClient
from .parser import parse_normalization_response
from .prompt import build_normalization_prompt
from .types import NormalizationResult, fallback_normalization


class LLMNormalizer:
    def __init__(self, client: LLMClient):
        self.client = client

    def normalize(self, raw_text: str) -> NormalizationResult:
        prompt = build_normalization_prompt(raw_text)

        try:
            llm_text = self.client.generate(prompt)
        except Exception:
            return fallback_normalization(raw_text)

        if not isinstance(llm_text, str):
            return fallback_normalization(raw_text)

        return parse_normalization_response(raw_text, llm_text)
