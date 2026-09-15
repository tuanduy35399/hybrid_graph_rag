from pathlib import Path

from src.rag.retrieval.config import build_runtime_provider_config
from src.rag.retrieval.graph_provider import build_manifest_graph_chunk_provider
from src.rag.retrieval.graph_retriever import GraphRetriever
from src.rag.retrieval.hybrid_runtime import HybridRuntimeRetriever
from src.rag.retrieval.hybrid_service import HybridRetrievalService
from src.rag.retrieval.local_embedding import build_local_embedding_client
from src.rag.retrieval.provider_types import BM25DocumentQuery, Phase5RuntimeConfig
from src.rag.retrieval.qdrant_store import QdrantVectorStoreClient


def build_phase5_runtime_config(
    *,
    collection_name: str,
    dense_vector_name: str,
    sparse_vector_name: str,
) -> Phase5RuntimeConfig:
    if not dense_vector_name.strip():
        raise ValueError(
            "Phase 5 dense vector name must be configured for named-vector hybrid retrieval."
        )
    if not sparse_vector_name.strip():
        raise ValueError(
            "Phase 5 sparse vector name must be configured for dense+sparse hybrid retrieval."
        )

    return Phase5RuntimeConfig(
        collection_name=collection_name,
        dense_vector_name=dense_vector_name,
        sparse_vector_name=sparse_vector_name,
    )


def build_runtime_embedding_client():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required to build the runtime embedding client."
        ) from exc

    config = build_runtime_provider_config()
    return build_local_embedding_client(
        config=config,
        sentence_transformer_cls=SentenceTransformer,
    )


def build_runtime_vector_store_client(
    *,
    collection_name: str | None = None,
    hybrid_ready: bool = False,
):
    try:
        from qdrant_client import QdrantClient, models as qdrant_models
    except ImportError as exc:
        raise RuntimeError(
            "qdrant-client is required to build the runtime vector store client."
        ) from exc

    config = build_runtime_provider_config()
    return QdrantVectorStoreClient(
        qdrant_client=QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
        ),
        qdrant_models=qdrant_models,
        collection_name=collection_name or config.qdrant_collection_name,
        hybrid_ready=hybrid_ready,
    )


def build_runtime_hybrid_retriever(
    *,
    sparse_query_builder,
    runtime_config,
    vector_store_client=None,
    embedding_client=None,
    prefetch_limit: int = 12,
    top_k: int = 8,
):
    return HybridRuntimeRetriever(
        embedding_client=embedding_client or build_runtime_embedding_client(),
        sparse_query_builder=sparse_query_builder,
        vector_store_client=vector_store_client
        or build_runtime_vector_store_client(
            collection_name=runtime_config.collection_name,
            hybrid_ready=True,
        ),
        runtime_config=runtime_config,
        prefetch_limit=prefetch_limit,
        top_k=top_k,
    )


def build_runtime_graph_retriever():
    return GraphRetriever()


def build_runtime_hybrid_retrieval_service(*, hybrid_retriever):
    return HybridRetrievalService(
        graph_retriever=build_runtime_graph_retriever(),
        hybrid_retriever=hybrid_retriever,
    )


class RuntimeBM25SparseQueryBuilder:
    def build(self, request):
        parts = [request.query_text]
        parts.extend(request.query_terms)
        parts.extend(hint.value for hint in request.entity_hints if hint.value.strip())
        parts.extend(request.subqueries)
        text = "\n".join(str(part).strip() for part in parts if str(part).strip())
        return BM25DocumentQuery(text=text, model="qdrant/bm25")


def build_runtime_sparse_query_builder():
    return RuntimeBM25SparseQueryBuilder()


def build_runtime_graph_chunk_provider(
    manifest_path: Path = Path("manifest.jsonl"),
):
    return build_manifest_graph_chunk_provider(manifest_path)


def build_runtime_text_retrieval_service(
    *,
    collection_name: str | None = None,
    dense_vector_name: str | None = None,
    sparse_vector_name: str | None = None,
    prefetch_limit: int = 12,
    top_k: int = 8,
):
    provider_config = build_runtime_provider_config()
    runtime_config = build_phase5_runtime_config(
        collection_name=collection_name or provider_config.qdrant_collection_name,
        dense_vector_name=dense_vector_name
        or getattr(provider_config, "qdrant_dense_vector_name", None)
        or "",
        sparse_vector_name=sparse_vector_name
        or getattr(provider_config, "qdrant_sparse_vector_name", None)
        or "",
    )
    vector_store_client = build_runtime_vector_store_client(
        collection_name=runtime_config.collection_name,
        hybrid_ready=True,
    )
    vector_store_client.assert_hybrid_collection_ready(
        dense_vector_name=runtime_config.dense_vector_name,
        sparse_vector_name=runtime_config.sparse_vector_name,
    )
    sparse_query_builder = build_runtime_sparse_query_builder()
    hybrid_retriever = build_runtime_hybrid_retriever(
        sparse_query_builder=sparse_query_builder,
        runtime_config=runtime_config,
        vector_store_client=vector_store_client,
        prefetch_limit=prefetch_limit,
        top_k=top_k,
    )
    return build_runtime_hybrid_retrieval_service(
        hybrid_retriever=hybrid_retriever,
    )
