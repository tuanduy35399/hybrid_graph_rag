import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.indexing.models import ChunkMetadata, SemanticRelation


def test_phase4_provider_config_reads_existing_qdrant_env_keys():
    from src.rag.retrieval.config import build_provider_config

    config = build_provider_config(
        {
            "QDRANT_API_KEY": "secret",
            "QDRANT_URL": "https://example.qdrant.io",
            "QDRANT_COLLECTION_NAME": "mai-kb",
            "QDRANT_DENSE_VECTOR_NAME": "dense-default",
            "QDRANT_SPARSE_VECTOR_NAME": "sparse-bm25",
        }
    )

    assert config.qdrant_api_key == "secret"
    assert config.qdrant_url == "https://example.qdrant.io"
    assert config.qdrant_collection_name == "mai-kb"
    assert config.qdrant_dense_vector_name == "dense-default"
    assert config.qdrant_sparse_vector_name == "sparse-bm25"
    assert config.embedding_model_name == "bkai-foundation-models/vietnamese-bi-encoder"


def test_build_runtime_provider_config_reads_explicit_env_file():
    import tempfile

    from src.rag.retrieval.config import build_runtime_provider_config

    temp_dir = Path(tempfile.mkdtemp())
    env_path = temp_dir / "phase4.env"
    env_path.write_text(
        "\n".join(
            [
                "QDRANT_API_KEY=runtime-secret",
                "QDRANT_URL=https://runtime.qdrant.io",
                "QDRANT_COLLECTION_NAME=runtime-kb",
                "QDRANT_DENSE_VECTOR_NAME=dense-runtime",
                "QDRANT_SPARSE_VECTOR_NAME=sparse-runtime",
            ]
        ),
        encoding="utf-8",
    )

    config = build_runtime_provider_config(env_path)

    assert config.qdrant_api_key == "runtime-secret"
    assert config.qdrant_url == "https://runtime.qdrant.io"
    assert config.qdrant_collection_name == "runtime-kb"
    assert config.qdrant_dense_vector_name == "dense-runtime"
    assert config.qdrant_sparse_vector_name == "sparse-runtime"


def test_build_runtime_provider_config_uses_module_relative_default_env_path():
    from src.rag.retrieval import config as config_module

    expected_env_path = Path(config_module.__file__).resolve().parents[2] / ".env"
    captured = {}

    def fake_load_env_file(env_path: Path):
        captured["env_path"] = env_path
        return {
            "QDRANT_API_KEY": "default-secret",
            "QDRANT_URL": "https://default.qdrant.io",
            "QDRANT_COLLECTION_NAME": "default-kb",
            "QDRANT_DENSE_VECTOR_NAME": "dense-default",
            "QDRANT_SPARSE_VECTOR_NAME": "sparse-default",
        }

    with patch.object(config_module, "_load_env_file", side_effect=fake_load_env_file):
        config = config_module.build_runtime_provider_config()

    assert captured["env_path"] == expected_env_path
    assert config.qdrant_collection_name == "default-kb"
    assert config.qdrant_dense_vector_name == "dense-default"
    assert config.qdrant_sparse_vector_name == "sparse-default"


def test_config_module_owns_env_loader_without_planning_runtime_dependency():
    import src.rag.retrieval.config as config_module

    assert not hasattr(config_module, "load_env_file")
    assert callable(config_module._load_env_file)


def test_build_provider_config_rejects_missing_required_env_key():
    from src.rag.retrieval.config import build_provider_config

    try:
        build_provider_config(
            {
                "QDRANT_API_KEY": "secret",
                "QDRANT_URL": "https://example.qdrant.io",
            }
        )
    except ValueError as exc:
        assert str(exc) == "Missing required environment variable: QDRANT_COLLECTION_NAME"
    else:
        raise AssertionError("Expected build_provider_config() to reject a missing env key")


def test_build_provider_config_rejects_blank_required_env_key():
    from src.rag.retrieval.config import build_provider_config

    try:
        build_provider_config(
            {
                "QDRANT_API_KEY": "secret",
                "QDRANT_URL": "   ",
                "QDRANT_COLLECTION_NAME": "mai-kb",
            }
        )
    except ValueError as exc:
        assert str(exc) == "Blank environment variable: QDRANT_URL"
    else:
        raise AssertionError("Expected build_provider_config() to reject a blank env value")


def test_local_env_loader_parses_comments_and_quoted_values():
    import tempfile

    from src.rag.retrieval.config import _load_env_file

    temp_dir = Path(tempfile.mkdtemp())
    env_path = temp_dir / "phase4.env"
    env_path.write_text(
        "\n".join(
            [
                "# comment",
                "",
                "QDRANT_API_KEY='quoted-secret'",
                'QDRANT_URL="https://quoted.qdrant.io"',
                "QDRANT_COLLECTION_NAME=quoted-kb",
                "NOT_A_PAIR",
            ]
        ),
        encoding="utf-8",
    )

    env_map = _load_env_file(env_path)

    assert env_map == {
        "QDRANT_API_KEY": "quoted-secret",
        "QDRANT_URL": "https://quoted.qdrant.io",
        "QDRANT_COLLECTION_NAME": "quoted-kb",
    }


def test_vector_point_preserves_vector_and_payload():
    from src.rag.retrieval.provider_types import VectorPoint

    point = VectorPoint(
        point_id="chunk-1",
        vector=(0.1, 0.2, 0.3),
        payload={"chunk_id": "chunk-1", "topic": "symptoms"},
    )

    assert point.point_id == "chunk-1"
    assert point.vector == (0.1, 0.2, 0.3)
    assert point.payload["topic"] == "symptoms"


def test_vector_point_copies_vector_as_immutable_tuple():
    from src.rag.retrieval.provider_types import VectorPoint

    original_vector = [0.1, 0.2, 0.3]
    point = VectorPoint(
        point_id="chunk-1",
        vector=original_vector,
        payload={"chunk_id": "chunk-1"},
    )

    original_vector[0] = 9.9

    assert point.vector == (0.1, 0.2, 0.3)
    assert isinstance(point.vector, tuple)


def test_vector_point_copies_payload_as_read_only_mapping():
    from src.rag.retrieval.provider_types import VectorPoint

    original_payload = {"chunk_id": "chunk-1", "topic": "symptoms"}
    point = VectorPoint(
        point_id="chunk-1",
        vector=(0.1, 0.2, 0.3),
        payload=original_payload,
    )

    original_payload["topic"] = "mutated"

    assert point.payload["topic"] == "symptoms"

    try:
        point.payload["new_field"] = "blocked"
    except TypeError:
        pass
    else:
        raise AssertionError("Expected VectorPoint.payload to be read-only")


def test_vector_point_deep_freezes_nested_payload_values():
    from src.rag.retrieval.provider_types import VectorPoint

    original_payload = {
        "chunk_id": "chunk-1",
        "metadata": {
            "topic": "symptoms",
            "tags": ["yellow_spot", {"severity": "high"}],
        },
    }
    point = VectorPoint(
        point_id="chunk-1",
        vector=(0.1, 0.2, 0.3),
        payload=original_payload,
    )

    original_payload["metadata"]["topic"] = "mutated"
    original_payload["metadata"]["tags"][0] = "changed"
    original_payload["metadata"]["tags"][1]["severity"] = "low"

    assert point.payload["metadata"]["topic"] == "symptoms"
    assert point.payload["metadata"]["tags"][0] == "yellow_spot"
    assert point.payload["metadata"]["tags"][1]["severity"] == "high"
    assert isinstance(point.payload["metadata"]["tags"], tuple)

    try:
        point.payload["metadata"]["tags"][1]["severity"] = "blocked"
    except TypeError:
        pass
    else:
        raise AssertionError("Expected nested payload mappings to be read-only")


def test_vector_point_deep_freezes_tuple_descendants():
    from src.rag.retrieval.provider_types import VectorPoint

    original_payload = {
        "chunk_id": "chunk-1",
        "metadata": (
            {"topic": "symptoms"},
            ["yellow_spot", {"severity": "high"}],
        ),
    }
    point = VectorPoint(
        point_id="chunk-1",
        vector=(0.1, 0.2, 0.3),
        payload=original_payload,
    )

    original_payload["metadata"][0]["topic"] = "mutated"
    original_payload["metadata"][1][0] = "changed"
    original_payload["metadata"][1][1]["severity"] = "low"

    assert point.payload["metadata"][0]["topic"] == "symptoms"
    assert point.payload["metadata"][1][0] == "yellow_spot"
    assert point.payload["metadata"][1][1]["severity"] == "high"
    assert isinstance(point.payload["metadata"], tuple)
    assert isinstance(point.payload["metadata"][1], tuple)

    try:
        point.payload["metadata"][0]["topic"] = "blocked"
    except TypeError:
        pass
    else:
        raise AssertionError("Expected tuple-descendant mappings to be read-only")


def test_retrieved_chunk_preserves_metadata_score_and_content():
    from src.rag.retrieval.provider_types import RetrievedChunk

    metadata = ChunkMetadata(
        source_file="benh.md",
        source_path="data/knowledge_base/benh.md",
        heading_path=("Benh", "Trieu chung"),
        doc_type="disease_guide",
        topic="symptoms",
        disease_name="ri_sat",
        semantic_graph=(
            SemanticRelation(
                source="dom vang",
                type="symptom",
                relation="indicates",
                target="ri_sat",
            ),
        ),
    )

    hit = RetrievedChunk(
        chunk_id="chunk-1",
        content="La mai co dom vang.",
        score=0.87,
        metadata=metadata,
    )

    assert hit.chunk_id == "chunk-1"
    assert hit.content == "La mai co dom vang."
    assert hit.score == 0.87
    assert hit.metadata.disease_name == "ri_sat"


def test_local_embedding_client_returns_tuple_vectors_in_input_order():
    from src.rag.retrieval.local_embedding import LocalEmbeddingClient

    class StubModel:
        def __init__(self):
            self.calls = []

        def encode(self, texts, normalize_embeddings=True):
            self.calls.append((tuple(texts), normalize_embeddings))
            return [
                [0.1, 0.2],
                [0.3, 0.4],
            ]

    client = LocalEmbeddingClient(model=StubModel())
    vectors = client.embed_texts(("alpha", "beta"))

    assert vectors == ((0.1, 0.2), (0.3, 0.4))
    assert client.model.calls == [(("alpha", "beta"), True)]


def test_local_embedding_client_returns_empty_tuple_without_calling_model_for_empty_batch():
    from src.rag.retrieval.local_embedding import LocalEmbeddingClient

    class StubModel:
        def __init__(self):
            self.calls = []

        def encode(self, texts, normalize_embeddings=True):
            self.calls.append((tuple(texts), normalize_embeddings))
            return [[9.9]]

    client = LocalEmbeddingClient(model=StubModel())

    assert client.embed_texts(()) == ()
    assert client.model.calls == []


def test_local_embedding_client_rejects_row_count_mismatch():
    from src.rag.retrieval.local_embedding import LocalEmbeddingClient

    class StubModel:
        def encode(self, texts, normalize_embeddings=True):
            return [[0.1, 0.2]]

    client = LocalEmbeddingClient(model=StubModel())

    try:
        client.embed_texts(("alpha", "beta"))
    except ValueError as exc:
        assert (
            str(exc)
            == "LocalEmbeddingClient.encode() returned 1 embedding rows for 2 input texts."
        )
    else:
        raise AssertionError("Expected LocalEmbeddingClient to reject mismatched row counts")


def test_local_embedding_client_rejects_singleton_non_batched_output():
    from src.rag.retrieval.local_embedding import LocalEmbeddingClient

    class StubModel:
        def encode(self, texts, normalize_embeddings=True):
            return [0.1, 0.2]

    client = LocalEmbeddingClient(model=StubModel())

    try:
        client.embed_texts(("alpha",))
    except ValueError as exc:
        assert (
            str(exc)
            == "LocalEmbeddingClient.encode() must return one embedding row per input text; got a 1-D vector for a singleton batch."
        )
    else:
        raise AssertionError("Expected LocalEmbeddingClient to reject singleton 1-D output")


def test_local_embedding_client_rejects_singleton_array_like_non_batched_output():
    from src.rag.retrieval.local_embedding import LocalEmbeddingClient

    class NumericLike:
        def __float__(self):
            return 0.1

    class StubVector:
        def __init__(self, values):
            self._values = tuple(values)

        def __len__(self):
            return len(self._values)

        def __getitem__(self, index):
            return self._values[index]

    class StubModel:
        def encode(self, texts, normalize_embeddings=True):
            return StubVector((NumericLike(), NumericLike()))

    client = LocalEmbeddingClient(model=StubModel())

    try:
        client.embed_texts(("alpha",))
    except ValueError as exc:
        assert (
            str(exc)
            == "LocalEmbeddingClient.encode() must return one embedding row per input text; got a 1-D vector for a singleton batch."
        )
    else:
        raise AssertionError("Expected LocalEmbeddingClient to reject singleton array-like output")


def test_build_local_embedding_client_uses_provider_config_model_name():
    from src.rag.retrieval.local_embedding import build_local_embedding_client
    from src.rag.retrieval.provider_types import ProviderConfig

    class StubSentenceTransformer:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def encode(self, texts, normalize_embeddings=True):
            return [[0.5]]

    config = ProviderConfig(
        qdrant_api_key="secret",
        qdrant_url="https://example.qdrant.io",
        qdrant_collection_name="mai-kb",
    )

    client = build_local_embedding_client(
        config=config,
        sentence_transformer_cls=StubSentenceTransformer,
    )

    assert client.__class__.__name__ == "LocalEmbeddingClient"
    assert client.model.model_name == "bkai-foundation-models/vietnamese-bi-encoder"


def test_qdrant_vector_store_upsert_translates_vector_points():
    from src.rag.retrieval.provider_types import VectorPoint
    from src.rag.retrieval.qdrant_store import QdrantVectorStoreClient

    class StubModels:
        class Distance:
            COSINE = "cosine"

        class VectorParams:
            def __init__(self, size: int, distance):
                self.size = size
                self.distance = distance

        class PointStruct:
            def __init__(self, id, vector, payload):
                self.id = id
                self.vector = vector
                self.payload = payload

        class MatchValue:
            def __init__(self, value):
                self.value = value

        class FieldCondition:
            def __init__(self, key, match):
                self.key = key
                self.match = match

        class Filter:
            def __init__(self, must=None, should=None):
                self.must = must
                self.should = should

    class StubQdrantClient:
        def __init__(self):
            self.upserts = []
            self.collections = []

        def recreate_collection(self, collection_name: str, vectors_config):
            self.collections.append(
                (collection_name, vectors_config.size, vectors_config.distance)
            )

        def upsert(self, collection_name: str, points):
            self.upserts.append((collection_name, points))

        def search(self, collection_name: str, query_vector, query_filter, limit: int):
            return []

    client = QdrantVectorStoreClient(
        qdrant_client=StubQdrantClient(),
        qdrant_models=StubModels,
        collection_name="mai-kb",
    )
    client.ensure_collection(vector_size=2)
    client.upsert_points(
        (
            VectorPoint(
                point_id="chunk-1",
                vector=(0.1, 0.2),
                payload={"chunk_id": "chunk-1", "content": "abc", "topic": "symptoms"},
            ),
        )
    )

    assert client.qdrant_client.collections == [("mai-kb", 2, "cosine")]
    assert client.qdrant_client.upserts[0][0] == "mai-kb"
    assert client.qdrant_client.upserts[0][1][0].id == "chunk-1"
    assert client.qdrant_client.upserts[0][1][0].vector == (0.1, 0.2)
    assert client.qdrant_client.upserts[0][1][0].payload["topic"] == "symptoms"


def test_qdrant_vector_store_can_create_hybrid_ready_named_collection_and_points():
    from src.rag.retrieval.provider_types import BM25DocumentQuery, VectorPoint
    from src.rag.retrieval.qdrant_store import QdrantVectorStoreClient

    class StubModels:
        class Distance:
            COSINE = "cosine"

        class VectorParams:
            def __init__(self, size: int, distance):
                self.size = size
                self.distance = distance

        class SparseVectorParams:
            pass

        class PointStruct:
            def __init__(self, id, vector, payload):
                self.id = id
                self.vector = vector
                self.payload = payload

        class MatchValue:
            def __init__(self, value):
                self.value = value

        class FieldCondition:
            def __init__(self, key, match):
                self.key = key
                self.match = match

        class Filter:
            def __init__(self, must=None, should=None):
                self.must = must
                self.should = should

    class StubQdrantClient:
        def __init__(self):
            self.collections = []
            self.upserts = []

        def recreate_collection(
            self,
            collection_name: str,
            vectors_config,
            sparse_vectors_config=None,
        ):
            self.collections.append(
                {
                    "collection_name": collection_name,
                    "vectors_config": vectors_config,
                    "sparse_vectors_config": sparse_vectors_config,
                }
            )

        def upsert(self, collection_name: str, points):
            self.upserts.append((collection_name, points))

        def search(self, collection_name: str, query_vector, query_filter, limit: int):
            return []

        def get_collection(self, collection_name: str):
            return {
                "config": {
                    "params": {
                        "vectors": {"dense-default": {"size": 2}},
                        "sparse_vectors": {"sparse-bm25": {}},
                    }
                }
            }

    client = QdrantVectorStoreClient(
        qdrant_client=StubQdrantClient(),
        qdrant_models=StubModels,
        collection_name="mai-kb",
    )

    client.ensure_collection(
        vector_size=2,
        dense_vector_name="dense-default",
        sparse_vector_name="sparse-bm25",
    )
    client.upsert_points(
        (
            VectorPoint(
                point_id="chunk-1",
                vector=(0.1, 0.2),
                dense_vector_name="dense-default",
                sparse_vector_name="sparse-bm25",
                sparse_document=BM25DocumentQuery(text="abc", model="qdrant/bm25"),
                payload={"chunk_id": "chunk-1", "content": "abc", "topic": "symptoms"},
            ),
        )
    )

    collection_call = client.qdrant_client.collections[0]
    assert collection_call["collection_name"] == "mai-kb"
    assert "dense-default" in collection_call["vectors_config"]
    assert collection_call["vectors_config"]["dense-default"].size == 2
    assert "sparse-bm25" in collection_call["sparse_vectors_config"]
    point = client.qdrant_client.upserts[0][1][0]
    assert point.vector == {
        "dense-default": (0.1, 0.2),
        "sparse-bm25": {"text": "abc", "model": "qdrant/bm25"},
    }


def test_qdrant_vector_store_upsert_normalizes_nested_payload_to_plain_containers():
    from src.rag.retrieval.provider_types import VectorPoint
    from src.rag.retrieval.qdrant_store import QdrantVectorStoreClient

    class StubModels:
        class Distance:
            COSINE = "cosine"

        class VectorParams:
            def __init__(self, size: int, distance):
                self.size = size
                self.distance = distance

        class PointStruct:
            def __init__(self, id, vector, payload):
                self.id = id
                self.vector = vector
                self.payload = payload

        class MatchValue:
            def __init__(self, value):
                self.value = value

        class FieldCondition:
            def __init__(self, key, match):
                self.key = key
                self.match = match

        class Filter:
            def __init__(self, must=None, should=None):
                self.must = must
                self.should = should

    class StubQdrantClient:
        def __init__(self):
            self.upserts = []

        def recreate_collection(self, collection_name: str, vectors_config):
            pass

        def upsert(self, collection_name: str, points):
            self.upserts.append((collection_name, points))

        def search(self, collection_name: str, query_vector, query_filter, limit: int):
            return []

    client = QdrantVectorStoreClient(
        qdrant_client=StubQdrantClient(),
        qdrant_models=StubModels,
        collection_name="mai-kb",
    )
    client.upsert_points(
        (
            VectorPoint(
                point_id="chunk-1",
                vector=(0.1, 0.2),
                payload={
                    "chunk_id": "chunk-1",
                    "metadata": (
                        {"topic": "symptoms"},
                        ["yellow_spot", {"severity": "high"}],
                    ),
                },
            ),
        )
    )

    payload = client.qdrant_client.upserts[0][1][0].payload
    assert isinstance(payload, dict)
    assert isinstance(payload["metadata"], list)
    assert isinstance(payload["metadata"][0], dict)
    assert isinstance(payload["metadata"][1], list)
    assert isinstance(payload["metadata"][1][1], dict)
    assert payload["metadata"][1][1]["severity"] == "high"


def test_qdrant_vector_store_search_omits_query_filter_when_filters_are_empty():
    from src.rag.retrieval.qdrant_store import QdrantVectorStoreClient

    class StubModels:
        class Distance:
            COSINE = "cosine"

        class VectorParams:
            def __init__(self, size: int, distance):
                self.size = size
                self.distance = distance

        class MatchValue:
            def __init__(self, value):
                self.value = value

        class FieldCondition:
            def __init__(self, key, match):
                self.key = key
                self.match = match

        class Filter:
            def __init__(self, must=None, should=None):
                self.must = must
                self.should = should

        class PointStruct:
            def __init__(self, id, vector, payload):
                self.id = id
                self.vector = vector
                self.payload = payload

    class StubQdrantClient:
        def recreate_collection(self, collection_name: str, vectors_config):
            pass

        def upsert(self, collection_name: str, points):
            pass

        def search(self, collection_name: str, query_vector, query_filter, limit: int):
            assert query_filter is None
            return []

    client = QdrantVectorStoreClient(
        qdrant_client=StubQdrantClient(),
        qdrant_models=StubModels,
        collection_name="mai-kb",
    )
    hits = client.search(
        query_vector=(0.1, 0.2),
        filters=(),
        top_k=3,
    )

    assert hits == ()


def test_qdrant_vector_store_search_groups_repeated_field_filters_with_or():
    from src.rag.retrieval.qdrant_store import QdrantVectorStoreClient
    from src.rag.retrieval.types import RetrievalFilter

    class StubModels:
        class Distance:
            COSINE = "cosine"

        class VectorParams:
            def __init__(self, size: int, distance):
                self.size = size
                self.distance = distance

        class MatchValue:
            def __init__(self, value):
                self.value = value

        class FieldCondition:
            def __init__(self, key, match):
                self.key = key
                self.match = match

        class Filter:
            def __init__(self, must=None, should=None):
                self.must = must
                self.should = should

        class PointStruct:
            def __init__(self, id, vector, payload):
                self.id = id
                self.vector = vector
                self.payload = payload

    class StubQdrantClient:
        def recreate_collection(self, collection_name: str, vectors_config):
            pass

        def upsert(self, collection_name: str, points):
            pass

        def search(self, collection_name: str, query_vector, query_filter, limit: int):
            assert query_filter is not None
            assert len(query_filter.must) == 2
            assert query_filter.must[0].key == "doc_type"
            assert query_filter.must[0].match.value == "disease_guide"
            assert query_filter.must[1].must is None
            assert len(query_filter.must[1].should) == 2
            assert [item.key for item in query_filter.must[1].should] == [
                "disease_name",
                "disease_name",
            ]
            assert [item.match.value for item in query_filter.must[1].should] == [
                "ri_sat",
                "thoi_nhun",
            ]
            return []

    client = QdrantVectorStoreClient(
        qdrant_client=StubQdrantClient(),
        qdrant_models=StubModels,
        collection_name="mai-kb",
    )
    hits = client.search(
        query_vector=(0.1, 0.2),
        filters=(
            RetrievalFilter(field="doc_type", value="disease_guide"),
            RetrievalFilter(field="disease_name", value="ri_sat"),
            RetrievalFilter(field="disease_name", value="thoi_nhun"),
        ),
        top_k=3,
    )

    assert hits == ()


def test_qdrant_vector_store_search_parses_payload_back_to_retrieved_chunk():
    from src.rag.retrieval.qdrant_store import QdrantVectorStoreClient
    from src.rag.retrieval.types import RetrievalFilter

    class StubHit:
        def __init__(self):
            self.id = "chunk-1"
            self.score = 0.77
            self.payload = {
                "chunk_id": "chunk-1",
                "content": "La mai co dom vang.",
                "source_file": "benh.md",
                "source_path": "data/knowledge_base/benh.md",
                "heading_path": ["Benh", "Trieu chung"],
                "doc_type": "disease_guide",
                "topic": "symptoms",
                "disease_name": "ri_sat",
                "semantic_graph": [
                    {
                        "source": "dom vang",
                        "type": "symptom",
                        "relation": "indicates",
                        "target": "ri_sat",
                    }
                ],
            }

    class StubModels:
        class Distance:
            COSINE = "cosine"

        class VectorParams:
            def __init__(self, size: int, distance):
                self.size = size
                self.distance = distance

        class MatchValue:
            def __init__(self, value):
                self.value = value

        class FieldCondition:
            def __init__(self, key, match):
                self.key = key
                self.match = match

        class Filter:
            def __init__(self, must=None, should=None):
                self.must = must
                self.should = should

        class PointStruct:
            def __init__(self, id, vector, payload):
                self.id = id
                self.vector = vector
                self.payload = payload

    class StubQdrantClient:
        def recreate_collection(self, collection_name: str, vectors_config):
            pass

        def upsert(self, collection_name: str, points):
            pass

        def search(self, collection_name: str, query_vector, query_filter, limit: int):
            assert collection_name == "mai-kb"
            assert tuple(query_vector) == (0.1, 0.2)
            assert limit == 3
            assert len(query_filter.must) == 1
            assert query_filter.must[0].key == "disease_name"
            assert query_filter.must[0].match.value == "ri_sat"
            return [StubHit()]

    client = QdrantVectorStoreClient(
        qdrant_client=StubQdrantClient(),
        qdrant_models=StubModels,
        collection_name="mai-kb",
    )
    hits = client.search(
        query_vector=(0.1, 0.2),
        filters=(RetrievalFilter(field="disease_name", value="ri_sat"),),
        top_k=3,
    )

    assert len(hits) == 1
    assert hits[0].chunk_id == "chunk-1"
    assert hits[0].content == "La mai co dom vang."
    assert hits[0].score == 0.77
    assert hits[0].metadata.source_file == "benh.md"
    assert hits[0].metadata.heading_path == ("Benh", "Trieu chung")
    assert hits[0].metadata.disease_name == "ri_sat"
    assert hits[0].metadata.semantic_graph[0].target == "ri_sat"


def test_qdrant_vector_store_search_rejects_string_heading_path_payload():
    from src.rag.retrieval.qdrant_store import QdrantVectorStoreClient

    class StubHit:
        def __init__(self):
            self.id = "chunk-1"
            self.score = 0.77
            self.payload = {
                "chunk_id": "chunk-1",
                "content": "La mai co dom vang.",
                "source_file": "benh.md",
                "source_path": "data/knowledge_base/benh.md",
                "heading_path": "Benh/Trieu chung",
            }

    class StubModels:
        class Distance:
            COSINE = "cosine"

        class VectorParams:
            def __init__(self, size: int, distance):
                self.size = size
                self.distance = distance

        class MatchValue:
            def __init__(self, value):
                self.value = value

        class FieldCondition:
            def __init__(self, key, match):
                self.key = key
                self.match = match

        class Filter:
            def __init__(self, must=None, should=None):
                self.must = must
                self.should = should

        class PointStruct:
            def __init__(self, id, vector, payload):
                self.id = id
                self.vector = vector
                self.payload = payload

    class StubQdrantClient:
        def recreate_collection(self, collection_name: str, vectors_config):
            pass

        def upsert(self, collection_name: str, points):
            pass

        def search(self, collection_name: str, query_vector, query_filter, limit: int):
            return [StubHit()]

    client = QdrantVectorStoreClient(
        qdrant_client=StubQdrantClient(),
        qdrant_models=StubModels,
        collection_name="mai-kb",
    )

    try:
        client.search(
            query_vector=(0.1, 0.2),
            filters=(),
            top_k=1,
        )
    except ValueError as exc:
        assert str(exc) == "Qdrant hit payload field 'heading_path' must be a sequence of strings, not a string."
    else:
        raise AssertionError("Expected malformed heading_path payload to raise ValueError")


def test_retrieval_package_exports_phase4_public_surface():
    import src.rag.retrieval as retrieval
    from src.rag.retrieval.config import build_provider_config, build_runtime_provider_config
    from src.rag.retrieval.dense_retriever import DenseRetriever, build_dense_query_text
    from src.rag.retrieval.local_embedding import (
        LocalEmbeddingClient,
        build_local_embedding_client,
    )
    from src.rag.retrieval.manifest_indexer import (
        index_manifest_records,
        load_manifest_records,
        prepare_vector_points,
    )
    from src.rag.retrieval.provider_types import (
        EmbeddingClient,
        ProviderConfig,
        RetrievedChunk,
        VectorPoint,
        VectorStoreClient,
    )
    from src.rag.retrieval.qdrant_store import (
        QdrantVectorStoreClient,
        build_qdrant_filter_payload,
    )
    from src.rag.retrieval.request_builder import build_retrieval_request
    from src.rag.retrieval.runtime import (
        build_runtime_embedding_client,
        build_runtime_vector_store_client,
    )
    from src.rag.retrieval.types import RetrievalFilter, RetrievalRequest

    assert retrieval.EmbeddingClient is EmbeddingClient
    assert retrieval.VectorStoreClient is VectorStoreClient
    assert retrieval.ProviderConfig is ProviderConfig
    assert retrieval.VectorPoint is VectorPoint
    assert retrieval.RetrievedChunk is RetrievedChunk
    assert retrieval.LocalEmbeddingClient is LocalEmbeddingClient
    assert retrieval.QdrantVectorStoreClient is QdrantVectorStoreClient
    assert retrieval.DenseRetriever is DenseRetriever
    assert retrieval.RetrievalFilter is RetrievalFilter
    assert retrieval.RetrievalRequest is RetrievalRequest
    assert retrieval.build_provider_config is build_provider_config
    assert retrieval.build_runtime_provider_config is build_runtime_provider_config
    assert retrieval.build_local_embedding_client is build_local_embedding_client
    assert retrieval.build_qdrant_filter_payload is build_qdrant_filter_payload
    assert retrieval.build_retrieval_request is build_retrieval_request
    assert retrieval.load_manifest_records is load_manifest_records
    assert retrieval.prepare_vector_points is prepare_vector_points
    assert retrieval.index_manifest_records is index_manifest_records
    assert retrieval.build_dense_query_text is build_dense_query_text
    assert retrieval.build_runtime_embedding_client is build_runtime_embedding_client
    assert retrieval.build_runtime_vector_store_client is build_runtime_vector_store_client
    assert retrieval.__all__ == [
        "DenseRetriever",
        "EmbeddingClient",
        "LocalEmbeddingClient",
        "ProviderConfig",
        "QdrantVectorStoreClient",
        "RetrievedChunk",
        "RetrievalFilter",
        "RetrievalRequest",
        "VectorPoint",
        "VectorStoreClient",
        "build_dense_query_text",
        "build_local_embedding_client",
        "build_provider_config",
        "build_qdrant_filter_payload",
        "build_retrieval_request",
        "build_runtime_embedding_client",
        "build_runtime_provider_config",
        "build_runtime_vector_store_client",
        "index_manifest_records",
        "load_manifest_records",
        "prepare_vector_points",
    ]
