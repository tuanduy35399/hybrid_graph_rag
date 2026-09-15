from src.rag.retrieval.provider_types import BM25DocumentQuery
from src.rag.retrieval.sparse_types import SparseQuery


def _build_sparse_query_payload(sparse_query):
    if isinstance(sparse_query, SparseQuery):
        return {
            "indices": [int(index) for index in sparse_query.indices],
            "values": [float(value) for value in sparse_query.values],
        }

    if isinstance(sparse_query, BM25DocumentQuery):
        return {
            "text": sparse_query.text,
            "model": sparse_query.model,
        }

    raise TypeError("Unsupported sparse query input for hybrid Qdrant payload.")


def build_hybrid_qdrant_query_payload(
    dense_vector: tuple[float, ...],
    sparse_query,
    dense_vector_name: str,
    sparse_vector_name: str,
    prefetch_limit: int,
    result_limit: int,
) -> dict[str, object]:
    return {
        "prefetch": [
            {
                "using": dense_vector_name,
                "query": [float(value) for value in dense_vector],
                "limit": prefetch_limit,
            },
            {
                "using": sparse_vector_name,
                "query": _build_sparse_query_payload(sparse_query),
                "limit": prefetch_limit,
            },
        ],
        "query": {"rrf": {}},
        "limit": result_limit,
    }
