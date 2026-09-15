from pathlib import Path

from src.rag.answering.runtime import build_runtime_text_answering_service
from src.rag.retrieval.runtime import (
    build_runtime_graph_chunk_provider,
    build_runtime_text_retrieval_service,
)
from src.rag.runtime.runtime import build_runtime_service


def build_text_only_rag_runtime_service(
    *,
    manifest_path: Path = Path("manifest.jsonl"),
    collection_name: str | None = None,
    dense_vector_name: str | None = None,
    sparse_vector_name: str | None = None,
    model_name: str = "gemini-2.5-flash",
    prefetch_limit: int = 12,
    top_k: int = 8,
):
    return build_runtime_service(
        text_retrieval_service_factory=lambda: build_runtime_text_retrieval_service(
            collection_name=collection_name,
            dense_vector_name=dense_vector_name,
            sparse_vector_name=sparse_vector_name,
            prefetch_limit=prefetch_limit,
            top_k=top_k,
        ),
        graph_chunk_provider_factory=lambda: build_runtime_graph_chunk_provider(
            manifest_path=manifest_path,
        ),
        text_answering_service_factory=lambda: build_runtime_text_answering_service(
            manifest_path=manifest_path,
            model_name=model_name,
        ),
    )


def answer_text_query(
    raw_text: str,
    runtime_service=None,
):
    service = runtime_service or build_text_only_rag_runtime_service()
    return service.answer_text_query(raw_text)
