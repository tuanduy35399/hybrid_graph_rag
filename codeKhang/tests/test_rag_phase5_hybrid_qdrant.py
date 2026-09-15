import os
import sys
import unittest
from unittest.mock import ANY


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class SparseQueryTests(unittest.TestCase):
    def test_sparse_query_coerces_sequences_to_immutable_tuples(self):
        from src.rag.retrieval.sparse_types import SparseQuery

        indices = [3, 7, 11]
        values = [0.25, 0.5, 0.75]

        query = SparseQuery(indices=indices, values=values)

        indices[0] = 99
        values[1] = 9.9

        self.assertEqual(query.indices, (3, 7, 11))
        self.assertEqual(query.values, (0.25, 0.5, 0.75))
        self.assertIsInstance(query.indices, tuple)
        self.assertIsInstance(query.values, tuple)


class HybridPayloadBuilderTests(unittest.TestCase):
    def test_build_hybrid_qdrant_query_payload_uses_named_dense_and_sparse_prefetch(self):
        from src.rag.retrieval.hybrid_qdrant import build_hybrid_qdrant_query_payload
        from src.rag.retrieval.sparse_types import SparseQuery

        payload = build_hybrid_qdrant_query_payload(
            dense_vector=(0.1, 0.2, 0.3),
            sparse_query=SparseQuery(indices=(1, 4), values=(0.7, 0.2)),
            dense_vector_name="dense-default",
            sparse_vector_name="sparse-bm25",
            prefetch_limit=25,
            result_limit=7,
        )

        self.assertEqual(payload["query"], {"rrf": {}})
        self.assertEqual(payload["limit"], 7)
        self.assertEqual(len(payload["prefetch"]), 2)
        self.assertEqual(
            payload["prefetch"][0],
            {
                "using": "dense-default",
                "query": [0.1, 0.2, 0.3],
                "limit": 25,
            },
        )
        self.assertEqual(
            payload["prefetch"][1],
            {
                "using": "sparse-bm25",
                "query": {
                    "indices": [1, 4],
                    "values": [0.7, 0.2],
                },
                "limit": 25,
            },
        )

    def test_build_hybrid_qdrant_query_payload_supports_bm25_document_query(self):
        from src.rag.retrieval.hybrid_qdrant import build_hybrid_qdrant_query_payload
        from src.rag.retrieval.provider_types import BM25DocumentQuery

        payload = build_hybrid_qdrant_query_payload(
            dense_vector=(0.1, 0.2),
            sparse_query=BM25DocumentQuery(
                text="la mai bi dom vang",
                model="qdrant/bm25",
            ),
            dense_vector_name="dense",
            sparse_vector_name="sparse",
            prefetch_limit=10,
            result_limit=4,
        )

        self.assertEqual(payload["prefetch"][1]["using"], "sparse")
        self.assertEqual(
            payload["prefetch"][1]["query"],
            {
                "text": "la mai bi dom vang",
                "model": "qdrant/bm25",
            },
        )
        self.assertEqual(payload["query"], {"rrf": {}})


class QdrantVectorStoreHybridTests(unittest.TestCase):
    def test_search_dense_path_still_uses_qdrant_search(self):
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
            def __init__(self):
                self.search_calls = []
                self.query_points_calls = []

            def recreate_collection(self, collection_name: str, vectors_config):
                return None

            def upsert(self, collection_name: str, points):
                return None

            def search(self, collection_name: str, query_vector, query_filter, limit: int):
                self.search_calls.append(
                    {
                        "collection_name": collection_name,
                        "query_vector": query_vector,
                        "query_filter": query_filter,
                        "limit": limit,
                    }
                )
                return []

            def query_points(self, **kwargs):
                self.query_points_calls.append(kwargs)
                return []

        qdrant_client = StubQdrantClient()
        client = QdrantVectorStoreClient(
            qdrant_client=qdrant_client,
            qdrant_models=StubModels,
            collection_name="mai-kb",
        )

        hits = client.search(
            query_vector=(0.1, 0.2),
            filters=(),
            top_k=3,
        )

        self.assertEqual(hits, ())
        self.assertEqual(
            qdrant_client.search_calls,
            [
                {
                    "collection_name": "mai-kb",
                    "query_vector": (0.1, 0.2),
                    "query_filter": None,
                    "limit": 3,
                }
            ],
        )
        self.assertEqual(qdrant_client.query_points_calls, [])

    def test_search_hybrid_passes_built_payload_to_query_points(self):
        from src.rag.retrieval.qdrant_store import QdrantVectorStoreClient
        from src.rag.retrieval.sparse_types import SparseQuery
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

        class StubHit:
            def __init__(self):
                self.id = "chunk-1"
                self.score = 0.88
                self.payload = {
                    "chunk_id": "chunk-1",
                    "content": "Hybrid result",
                    "source_file": "benh.md",
                    "source_path": "data/knowledge_base/benh.md",
                }

        class StubHybridResponse:
            def __init__(self):
                self.points = [StubHit()]

        class StubQdrantClient:
            def __init__(self):
                self.query_points_calls = []

            def recreate_collection(self, collection_name: str, vectors_config):
                return None

            def upsert(self, collection_name: str, points):
                return None

            def search(self, collection_name: str, query_vector, query_filter, limit: int):
                return []

            def query_points(self, **kwargs):
                self.query_points_calls.append(kwargs)
                return StubHybridResponse()

        qdrant_client = StubQdrantClient()
        client = QdrantVectorStoreClient(
            qdrant_client=qdrant_client,
            qdrant_models=StubModels,
            collection_name="mai-kb",
            hybrid_ready=True,
        )

        hits = client.search_hybrid(
            dense_vector=(0.1, 0.2),
            sparse_query=SparseQuery(indices=(5, 8), values=(0.4, 0.9)),
            filters=(
                RetrievalFilter(field="doc_type", value="disease_guide"),
                RetrievalFilter(field="disease_name", value="ri_sat"),
                RetrievalFilter(field="disease_name", value="thoi_nhun"),
            ),
            dense_vector_name="dense-default",
            sparse_vector_name="sparse-bm25",
            prefetch_limit=12,
            top_k=4,
        )

        self.assertEqual(len(qdrant_client.query_points_calls), 1)
        self.assertEqual(
            qdrant_client.query_points_calls[0],
            {
                "collection_name": "mai-kb",
                "prefetch": [
                    {
                        "using": "dense-default",
                        "query": [0.1, 0.2],
                        "limit": 12,
                    },
                    {
                        "using": "sparse-bm25",
                        "query": {"indices": [5, 8], "values": [0.4, 0.9]},
                        "limit": 12,
                    },
                ],
                "query": {"rrf": {}},
                "query_filter": ANY,
                "limit": 4,
            },
        )
        query_filter = qdrant_client.query_points_calls[0]["query_filter"]
        self.assertEqual(len(query_filter.must), 2)
        self.assertEqual(query_filter.must[0].key, "doc_type")
        self.assertEqual(query_filter.must[0].match.value, "disease_guide")
        self.assertEqual(len(query_filter.must[1].should), 2)
        self.assertEqual(
            [item.match.value for item in query_filter.must[1].should],
            ["ri_sat", "thoi_nhun"],
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk_id, "chunk-1")

    def test_search_hybrid_adapts_payload_to_provider_query_models_when_available(self):
        from src.rag.retrieval.qdrant_store import QdrantVectorStoreClient
        from src.rag.retrieval.provider_types import BM25DocumentQuery

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

            class Prefetch:
                def __init__(self, using, query, limit):
                    self.using = using
                    self.query = query
                    self.limit = limit

            class SparseVector:
                def __init__(self, indices, values):
                    self.indices = indices
                    self.values = values

            class Document:
                def __init__(self, text, model):
                    self.text = text
                    self.model = model

            class Fusion:
                RRF = "rrf"

            class FusionQuery:
                def __init__(self, fusion):
                    self.fusion = fusion

        class StubQdrantClient:
            def __init__(self):
                self.query_points_calls = []

            def recreate_collection(self, collection_name: str, vectors_config):
                return None

            def upsert(self, collection_name: str, points):
                return None

            def search(self, collection_name: str, query_vector, query_filter, limit: int):
                return []

            def query_points(self, **kwargs):
                self.query_points_calls.append(kwargs)
                return []

        qdrant_client = StubQdrantClient()
        client = QdrantVectorStoreClient(
            qdrant_client=qdrant_client,
            qdrant_models=StubModels,
            collection_name="mai-kb",
            hybrid_ready=True,
        )

        client.search_hybrid(
            dense_vector=(0.1, 0.2),
            sparse_query=BM25DocumentQuery(text="la mai bi dom vang", model="qdrant/bm25"),
            filters=(),
            dense_vector_name="dense-default",
            sparse_vector_name="sparse-bm25",
            prefetch_limit=12,
            top_k=4,
        )

        call = qdrant_client.query_points_calls[0]
        self.assertEqual(call["collection_name"], "mai-kb")
        self.assertEqual(call["limit"], 4)
        self.assertIsInstance(call["query"], StubModels.FusionQuery)
        self.assertEqual(call["query"].fusion, StubModels.Fusion.RRF)
        self.assertEqual(len(call["prefetch"]), 2)
        self.assertIsInstance(call["prefetch"][0], StubModels.Prefetch)
        self.assertEqual(call["prefetch"][0].using, "dense-default")
        self.assertEqual(call["prefetch"][0].query, [0.1, 0.2])
        self.assertIsInstance(call["prefetch"][1], StubModels.Prefetch)
        self.assertEqual(call["prefetch"][1].using, "sparse-bm25")
        self.assertIsInstance(call["prefetch"][1].query, StubModels.Document)
        self.assertEqual(call["prefetch"][1].query.text, "la mai bi dom vang")
        self.assertEqual(call["prefetch"][1].query.model, "qdrant/bm25")

    def test_search_hybrid_rejects_client_that_is_not_hybrid_ready(self):
        from src.rag.retrieval.qdrant_store import QdrantVectorStoreClient
        from src.rag.retrieval.sparse_types import SparseQuery

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
                return None

            def upsert(self, collection_name: str, points):
                return None

            def search(self, collection_name: str, query_vector, query_filter, limit: int):
                return []

            def query_points(self, **kwargs):
                raise AssertionError("query_points should not be called when hybrid is not enabled")

        client = QdrantVectorStoreClient(
            qdrant_client=StubQdrantClient(),
            qdrant_models=StubModels,
            collection_name="mai-kb",
        )

        with self.assertRaisesRegex(
            ValueError,
            "QdrantVectorStoreClient.search_hybrid\\(\\) requires a hybrid-ready client configuration.",
        ):
            client.search_hybrid(
                dense_vector=(0.1, 0.2),
                sparse_query=SparseQuery(indices=(5, 8), values=(0.4, 0.9)),
                filters=(),
                dense_vector_name="dense-default",
                sparse_vector_name="sparse-bm25",
                prefetch_limit=12,
                top_k=4,
            )


class Phase5RuntimeConfigTests(unittest.TestCase):
    def test_build_phase5_runtime_config_requires_sparse_vector_name(self):
        from src.rag.retrieval.runtime import build_phase5_runtime_config

        with self.assertRaisesRegex(
            ValueError,
            "Phase 5 sparse vector name must be configured for dense\\+sparse hybrid retrieval\\.",
        ):
            build_phase5_runtime_config(
                collection_name="mai-kb",
                dense_vector_name="dense-default",
                sparse_vector_name="",
            )

    def test_build_phase5_runtime_config_returns_dense_sparse_boundary_values(self):
        from src.rag.retrieval.runtime import build_phase5_runtime_config

        config = build_phase5_runtime_config(
            collection_name="mai-kb",
            dense_vector_name="dense-default",
            sparse_vector_name="sparse-bm25",
        )

        self.assertEqual(config.collection_name, "mai-kb")
        self.assertEqual(config.dense_vector_name, "dense-default")
        self.assertEqual(config.sparse_vector_name, "sparse-bm25")


if __name__ == "__main__":
    unittest.main()
