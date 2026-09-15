import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.integration.types import BranchType
from src.rag.normalization.types import QueryType
from src.rag.planning.types import EntityHint, QueryIntent
from src.rag.retrieval.types import RetrievalFilter, RetrievalRequest


def _make_request(
    *,
    query_text="la mai bi dom vang la benh gi",
    query_terms=("mai", "dom vang"),
    filters=(),
):
    return RetrievalRequest(
        query_text=query_text,
        branch=BranchType.TEXT_ONLY_BRANCH,
        query_type=QueryType.DISEASE_QUERY,
        intent=QueryIntent.DISEASE_LOOKUP,
        query_terms=query_terms,
        entity_hints=(EntityHint(kind="symptom", value="dom vang"),),
        topic_hints=("symptoms", "treatment"),
        filters=filters,
        subqueries=("la mai bi dom vang la benh gi",),
    )


def test_build_dense_query_text_keeps_query_and_semantic_terms_together():
    from src.rag.retrieval.dense_retriever import build_dense_query_text

    request = _make_request(query_terms=("mai", "dom vang", "benh"))

    dense_text = build_dense_query_text(request)

    assert dense_text == "la mai bi dom vang la benh gi\nmai\ndom vang\nbenh"


def test_build_dense_query_text_deduplicates_cleaned_terms():
    from src.rag.retrieval.dense_retriever import build_dense_query_text

    request = _make_request(query_terms=("  mai  ", "dom   vang", "mai", "", " benh "))

    dense_text = build_dense_query_text(request)

    assert dense_text == "la mai bi dom vang la benh gi\nmai\ndom vang\nbenh"


def test_build_dense_query_text_uses_terms_when_query_text_is_whitespace_only():
    from src.rag.retrieval.dense_retriever import build_dense_query_text

    request = _make_request(
        query_text=" \n\t ",
        query_terms=("  mai  ", "dom   vang", "mai"),
    )

    dense_text = build_dense_query_text(request)

    assert dense_text == "mai\ndom vang"


def test_build_dense_query_text_rejects_fully_blank_input():
    from src.rag.retrieval.dense_retriever import build_dense_query_text

    request = _make_request(
        query_text=" \n\t ",
        query_terms=(" ", "\t"),
    )

    try:
        build_dense_query_text(request)
    except ValueError as exc:
        assert str(exc) == "Dense query text cannot be empty."
    else:
        raise AssertionError("Expected ValueError for fully blank dense query input")


def test_dense_retriever_embeds_request_and_calls_vector_store():
    from src.rag.indexing.models import ChunkMetadata
    from src.rag.retrieval.dense_retriever import DenseRetriever
    from src.rag.retrieval.provider_types import RetrievedChunk

    class StubEmbeddingClient:
        def __init__(self):
            self.calls = []

        def embed_texts(self, texts: tuple[str, ...]):
            self.calls.append(texts)
            return ((0.1, 0.2),)

    class StubVectorStore:
        def __init__(self):
            self.calls = []
            self.result = (
                RetrievedChunk(
                    chunk_id="chunk-1",
                    content="La mai co dom vang.",
                    score=0.9,
                    metadata=ChunkMetadata(
                        source_file="benh.md",
                        source_path="data/knowledge_base/benh.md",
                        heading_path=("Benh",),
                        doc_type="disease_guide",
                        topic="symptoms",
                        disease_name="ri_sat",
                        semantic_graph=(),
                    ),
                ),
            )

        def search(self, query_vector, filters, top_k: int):
            self.calls.append((query_vector, filters, top_k))
            return self.result

    request = _make_request(
        filters=(RetrievalFilter(field="doc_type", value="disease_guide"),),
    )

    embedding_client = StubEmbeddingClient()
    vector_store_client = StubVectorStore()
    retriever = DenseRetriever(
        embedding_client=embedding_client,
        vector_store_client=vector_store_client,
        top_k=4,
    )

    hits = retriever.retrieve(request)

    assert hits is vector_store_client.result
    assert embedding_client.calls == [("la mai bi dom vang la benh gi\nmai\ndom vang",)]
    assert vector_store_client.calls == [((0.1, 0.2), request.filters, 4)]


def test_dense_retriever_rejects_empty_embedding_output():
    from src.rag.retrieval.dense_retriever import DenseRetriever

    class StubEmbeddingClient:
        def embed_texts(self, texts: tuple[str, ...]):
            return ()

    class StubVectorStore:
        def search(self, query_vector, filters, top_k: int):
            raise AssertionError("vector store should not be called")

    retriever = DenseRetriever(
        embedding_client=StubEmbeddingClient(),
        vector_store_client=StubVectorStore(),
    )

    try:
        retriever.retrieve(_make_request())
    except ValueError as exc:
        assert str(exc) == "Embedding client must return exactly one vector for a singleton dense query."
    else:
        raise AssertionError("Expected ValueError for empty embedding output")


def test_dense_retriever_rejects_multi_row_embedding_output_for_singleton_query():
    from src.rag.retrieval.dense_retriever import DenseRetriever

    class StubEmbeddingClient:
        def embed_texts(self, texts: tuple[str, ...]):
            return ((0.1, 0.2), (0.3, 0.4))

    class StubVectorStore:
        def search(self, query_vector, filters, top_k: int):
            raise AssertionError("vector store should not be called")

    retriever = DenseRetriever(
        embedding_client=StubEmbeddingClient(),
        vector_store_client=StubVectorStore(),
    )

    try:
        retriever.retrieve(_make_request())
    except ValueError as exc:
        assert str(exc) == "Embedding client must return exactly one vector for a singleton dense query."
    else:
        raise AssertionError("Expected ValueError for multi-row embedding output")


def test_dense_retriever_rejects_non_batched_singleton_embedding_output():
    from src.rag.retrieval.dense_retriever import DenseRetriever

    class StubEmbeddingClient:
        def embed_texts(self, texts: tuple[str, ...]):
            return (0.1, 0.2)

    class StubVectorStore:
        def search(self, query_vector, filters, top_k: int):
            raise AssertionError("vector store should not be called")

    retriever = DenseRetriever(
        embedding_client=StubEmbeddingClient(),
        vector_store_client=StubVectorStore(),
    )

    try:
        retriever.retrieve(_make_request())
    except ValueError as exc:
        assert str(exc) == "Embedding client must return a batched singleton result, not a raw vector."
    else:
        raise AssertionError("Expected ValueError for non-batched singleton embedding output")
