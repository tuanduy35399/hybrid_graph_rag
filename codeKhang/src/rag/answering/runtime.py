from pathlib import Path

from codeKhang.src.rag.answering.gemini_client import GeminiAnswerGeneratorClient
from codeKhang.src.rag.answering.resolver import build_manifest_chunk_resolver
from codeKhang.src.rag.answering.selector import ScorePreservingReranker
from src.rag.answering.service import TextAnsweringService
from src.rag.planning.runtime import load_env_file, read_gemini_api_key_from_env_map

try:
    import google.generativeai as google_generativeai
except ImportError:
    google_generativeai = None


def build_runtime_text_answering_service(
    *,
    manifest_path: Path = Path("manifest.jsonl"),
    model_name: str = "gemini-2.5-flash",
):
    if google_generativeai is None:
        raise RuntimeError("google-generativeai is required for answer synthesis.")

    env_map = load_env_file(Path("src/.env"))
    api_key = read_gemini_api_key_from_env_map(env_map)
    google_generativeai.configure(api_key=api_key)
    model = google_generativeai.GenerativeModel(model_name)

    return TextAnsweringService(
        reranker=ScorePreservingReranker(),
        chunk_resolver=build_manifest_chunk_resolver(manifest_path),
        answer_generator=GeminiAnswerGeneratorClient(model),
    )
