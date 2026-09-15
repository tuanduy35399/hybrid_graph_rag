from src.rag.retrieval.dense_retriever import (
    _extract_singleton_query_vector,
    build_dense_query_text,
)
from src.rag.retrieval.hybrid_types import HybridCandidate


class HybridRuntimeRetriever:
    def __init__(
        self,
        *,
        embedding_client,
        sparse_query_builder,
        vector_store_client,
        runtime_config,
        prefetch_limit: int = 12,
        top_k: int = 8,
    ):
        self.embedding_client = embedding_client
        self.sparse_query_builder = sparse_query_builder
        self.vector_store_client = vector_store_client
        self.runtime_config = runtime_config
        self.prefetch_limit = prefetch_limit
        self.top_k = top_k

    def retrieve(self, request):
        dense_query_text = build_dense_query_text(request)
        embedding_rows = self.embedding_client.embed_texts((dense_query_text,))
        query_vector = _extract_singleton_query_vector(embedding_rows)
        sparse_query = self.sparse_query_builder.build(request)
        chunks = self.vector_store_client.search_hybrid(
            dense_vector=query_vector,
            sparse_query=sparse_query,
            filters=request.filters,
            dense_vector_name=self.runtime_config.dense_vector_name,
            sparse_vector_name=self.runtime_config.sparse_vector_name,
            prefetch_limit=self.prefetch_limit,
            top_k=self.top_k,
        )
        return tuple(
            HybridCandidate(
                chunk_id=chunk.chunk_id,
                sources=("dense", "sparse"),
                raw_scores={"hybrid": float(chunk.score)},
            )
            for chunk in chunks
        )
