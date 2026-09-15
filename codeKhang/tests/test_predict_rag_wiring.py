import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.integration.types import BranchType, PreparedBranchInput
from src.rag.normalization.types import NormalizationResult, QueryType


class StubPreprocessor:
    def __init__(self, prepared):
        self.prepared = prepared
        self.calls = []

    def prepare(self, raw_text: str, has_image: bool):
        self.calls.append((raw_text, has_image))
        return self.prepared


def test_gemini_llm_client_wraps_model_generate_content_result():
    from src.rag.normalization.gemini_client import GeminiLLMClient

    class StubResponse:
        text = '{"normalized_query":"cach tuoi mai","query_type":"care_query","normalization_notes":["no_major_change"]}'

    class StubModel:
        def __init__(self):
            self.calls = []

        def generate_content(self, prompt: str):
            self.calls.append(prompt)
            return StubResponse()

    client = GeminiLLMClient(StubModel())
    output = client.generate("normalize this")

    assert output == StubResponse.text
    assert client.model.calls == ["normalize this"]


def test_prepare_image_prompt_uses_has_image_true_and_returns_normalized_prompt():
    from src.rag.integration.runtime import prepare_image_prompt

    prepared = PreparedBranchInput(
        branch=BranchType.IMAGE_BRANCH,
        normalization=NormalizationResult(
            normalized_query="la mai bi dom den la benh gi",
            query_type=QueryType.DISEASE_QUERY,
            normalization_notes=("rewritten_for_retrieval",),
        ),
    )
    preprocessor = StubPreprocessor(prepared)

    runtime_input = prepare_image_prompt("la mai bi dom den la gi", preprocessor)

    assert preprocessor.calls == [("la mai bi dom den la gi", True)]
    assert runtime_input.normalized_prompt == "la mai bi dom den la benh gi"
    assert runtime_input.prepared.branch is BranchType.IMAGE_BRANCH


def test_read_gemini_api_key_from_env_map_returns_expected_key():
    from src.rag.integration.runtime import read_gemini_api_key_from_env_map

    env_map = {"GEMINI_API_KEY": "secret-key"}

    assert read_gemini_api_key_from_env_map(env_map) == "secret-key"


def test_build_fallback_image_preprocessor_produces_safe_image_branch_result():
    from src.rag.integration.runtime import build_fallback_image_preprocessor

    preprocessor = build_fallback_image_preprocessor()
    prepared = preprocessor.prepare("  la mai bi dom den  ", has_image=True)

    assert prepared.branch.value == "image_branch"
    assert prepared.normalization.normalized_query == "la mai bi dom den"
    assert prepared.normalization.normalization_notes == ("fallback_raw_query",)


def test_build_gemini_image_preprocessor_uses_factory_outputs():
    from src.rag.integration.runtime import build_gemini_image_preprocessor

    class StubResponse:
        text = '{"normalized_query":"cach tuoi mai","query_type":"care_query","normalization_notes":["no_major_change"]}'

    class StubModel:
        def __init__(self, model_name: str):
            self.model_name = model_name
            self.prompts = []

        def generate_content(self, prompt: str):
            self.prompts.append(prompt)
            return StubResponse()

    class StubGenAI:
        def __init__(self):
            self.configured = []
            self.models = []

        def configure(self, api_key: str):
            self.configured.append(api_key)

        def GenerativeModel(self, model_name: str):
            model = StubModel(model_name)
            self.models.append(model)
            return model

    genai = StubGenAI()
    env_map = {"GEMINI_API_KEY": "secret-key"}
    preprocessor = build_gemini_image_preprocessor(
        env_map=env_map,
        genai_module=genai,
        model_name="gemini-2.5-flash",
    )

    prepared = preprocessor.prepare("cach tuoi mai", has_image=True)

    assert genai.configured == ["secret-key"]
    assert len(genai.models) == 1
    assert genai.models[0].model_name == "gemini-2.5-flash"
    assert prepared.branch.value == "image_branch"
    assert prepared.normalization.normalized_query == "cach tuoi mai"


def test_select_prompt_for_text_transform_returns_normalized_prompt():
    from predict import select_prompt_for_text_transform
    from src.rag.integration.runtime import PreparedImagePrompt

    prepared = PreparedImagePrompt(
        raw_prompt="la mai bi dom den la gi",
        normalized_prompt="la mai bi dom den la benh gi",
        prepared=None,
    )

    assert select_prompt_for_text_transform(prepared) == "la mai bi dom den la benh gi"


def test_prepare_image_prompt_preserves_mixed_query_without_changing_branch():
    from src.rag.integration.runtime import prepare_image_prompt

    prepared = PreparedBranchInput(
        branch=BranchType.IMAGE_BRANCH,
        normalization=NormalizationResult(
            normalized_query="la mai bi dom den la benh gi va xu ly sao",
            query_type=QueryType.MIXED_QUERY,
            normalization_notes=("rewritten_for_retrieval",),
        ),
    )
    preprocessor = StubPreprocessor(prepared)

    runtime_input = prepare_image_prompt(
        "la mai bi dom den la gi va xu ly sao",
        preprocessor,
    )

    assert runtime_input.prepared.branch is BranchType.IMAGE_BRANCH
    assert runtime_input.prepared.normalization.query_type is QueryType.MIXED_QUERY
