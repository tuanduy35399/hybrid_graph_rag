from src.rag.runtime.runtime import build_runtime_service
from src.rag.runtime.types import TextRuntimeResult


def prepare_text_runtime_probe(
    raw_text: str,
    runtime_service=None,
) -> TextRuntimeResult:
    service = runtime_service or build_runtime_service()
    return service.prepare_text_runtime_result(raw_text)
