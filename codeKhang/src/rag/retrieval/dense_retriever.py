from numbers import Real

from src.rag.retrieval.types import RetrievalRequest


def build_dense_query_text(request: RetrievalRequest) -> str:
    lines = []

    query_text = _clean_line(request.query_text)
    if query_text:
        lines.append(query_text)

    for term in request.query_terms:
        cleaned = _clean_line(term)
        if cleaned and cleaned not in lines:
            lines.append(cleaned)

    if not lines:
        raise ValueError("Dense query text cannot be empty.")

    return "\n".join(lines)


class DenseRetriever:
    def __init__(self, embedding_client, vector_store_client, top_k: int = 8):
        self.embedding_client = embedding_client
        self.vector_store_client = vector_store_client
        self.top_k = top_k

    def retrieve(self, request: RetrievalRequest):
        dense_query_text = build_dense_query_text(request)
        embedding_rows = self.embedding_client.embed_texts((dense_query_text,))
        query_vector = _extract_singleton_query_vector(embedding_rows)
        return self.vector_store_client.search(
            query_vector=query_vector,
            filters=request.filters,
            top_k=self.top_k,
        )


def _clean_line(value: str) -> str:
    return " ".join(value.strip().split())


def _extract_singleton_query_vector(embedding_rows) -> tuple[float, ...]:
    if _looks_like_numeric_vector(embedding_rows):
        raise ValueError(
            "Embedding client must return a batched singleton result, not a raw vector."
        )

    if len(embedding_rows) != 1:
        raise ValueError(
            "Embedding client must return exactly one vector for a singleton dense query."
        )

    row = embedding_rows[0]
    if isinstance(row, Real):
        raise ValueError(
            "Embedding client must return a batched singleton result, not a raw vector."
        )

    try:
        vector = tuple(float(value) for value in row)
    except TypeError as exc:
        raise ValueError(
            "Embedding client must return a batched singleton result, not a raw vector."
        ) from exc
    except ValueError as exc:
        raise ValueError("Embedding vector must contain only numeric values.") from exc

    return vector


def _looks_like_numeric_vector(value) -> bool:
    if isinstance(value, (str, bytes)):
        return False

    try:
        items = tuple(value)
    except TypeError:
        return False

    return bool(items) and all(isinstance(item, Real) for item in items)
