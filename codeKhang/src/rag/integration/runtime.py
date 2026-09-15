from dataclasses import dataclass
from pathlib import Path

from src.rag.integration.service import SharedBranchPreprocessor
from src.rag.integration.types import PreparedBranchInput
from src.rag.normalization.gemini_client import GeminiLLMClient
from src.rag.normalization.service import LLMNormalizer
from src.rag.normalization.types import fallback_normalization


@dataclass(frozen=True)
class PreparedImagePrompt:
    raw_prompt: str
    normalized_prompt: str
    prepared: PreparedBranchInput | None


class FallbackOnlyNormalizer:
    def normalize(self, raw_text: str):
        return fallback_normalization(raw_text)


def read_gemini_api_key_from_env_map(env_map: dict[str, str]) -> str:
    return env_map["GEMINI_API_KEY"]


def load_env_file(env_path: Path) -> dict[str, str]:
    env_map: dict[str, str] = {}
    if not env_path.exists():
        return env_map

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        env_map[key.strip()] = value

    return env_map


def prepare_image_prompt(
    raw_prompt: str,
    preprocessor: SharedBranchPreprocessor,
) -> PreparedImagePrompt:
    prepared = preprocessor.prepare(raw_prompt, has_image=True)
    return PreparedImagePrompt(
        raw_prompt=raw_prompt,
        normalized_prompt=prepared.normalization.normalized_query,
        prepared=prepared,
    )


def build_fallback_image_preprocessor() -> SharedBranchPreprocessor:
    return SharedBranchPreprocessor(FallbackOnlyNormalizer())


def build_gemini_image_preprocessor(
    env_map: dict[str, str],
    genai_module,
    model_name: str = "gemini-2.5-flash",
) -> SharedBranchPreprocessor:
    api_key = read_gemini_api_key_from_env_map(env_map)
    genai_module.configure(api_key=api_key)
    model = genai_module.GenerativeModel(model_name)
    client = GeminiLLMClient(model)
    normalizer = LLMNormalizer(client)
    return SharedBranchPreprocessor(normalizer)


def build_runtime_image_preprocessor() -> SharedBranchPreprocessor:
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError(
            "google-generativeai is required for Gemini normalization."
        ) from exc

    env_path = Path("src/.env")
    env_map = load_env_file(env_path)
    return build_gemini_image_preprocessor(env_map=env_map, genai_module=genai)
