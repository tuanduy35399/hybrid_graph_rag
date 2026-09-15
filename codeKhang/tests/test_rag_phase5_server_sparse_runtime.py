import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class Phase5ServerSparseRuntimeTests(unittest.TestCase):
    def test_bm25_document_query_contract_keeps_text_and_model(self):
        from src.rag.retrieval.provider_types import BM25DocumentQuery

        query = BM25DocumentQuery(
            text="la mai bi dom vang",
            model="qdrant/bm25",
        )

        self.assertEqual(query.text, "la mai bi dom vang")
        self.assertEqual(query.model, "qdrant/bm25")

    def test_build_runtime_sparse_query_returns_bm25_document_query(self):
        from src.rag.integration.types import BranchType
        from src.rag.normalization.types import QueryType
        from src.rag.planning.types import QueryIntent
        from src.rag.retrieval.runtime import build_runtime_sparse_query_builder
        from src.rag.retrieval.types import RetrievalRequest

        request = RetrievalRequest(
            query_text="la mai bi dom vang",
            branch=BranchType.TEXT_ONLY_BRANCH,
            query_type=QueryType.DISEASE_QUERY,
            intent=QueryIntent.DISEASE_LOOKUP,
            query_terms=("dom vang", "ri sat"),
            entity_hints=(),
            topic_hints=("disease",),
            filters=(),
            subqueries=("la mai bi dom vang",),
        )

        builder = build_runtime_sparse_query_builder()
        sparse_query = builder.build(request)

        self.assertEqual(sparse_query.model, "qdrant/bm25")
        self.assertIn("la mai bi dom vang", sparse_query.text)

    def test_build_runtime_text_retrieval_service_propagates_blank_sparse_vector_name(self):
        import src.rag.retrieval.runtime as runtime_module

        original_build_provider_config = runtime_module.build_runtime_provider_config

        try:
            runtime_module.build_runtime_provider_config = lambda: type(
                "Cfg",
                (),
                {
                    "qdrant_collection_name": "mai-kb",
                    "qdrant_dense_vector_name": "dense-runtime",
                },
            )()

            with self.assertRaisesRegex(
                ValueError,
                "Phase 5 sparse vector name must be configured for dense\\+sparse hybrid retrieval\\.",
            ):
                runtime_module.build_runtime_text_retrieval_service(
                    sparse_vector_name="",
                )
        finally:
            runtime_module.build_runtime_provider_config = original_build_provider_config

    def test_build_runtime_text_retrieval_service_reads_vector_names_from_provider_config(self):
        import src.rag.retrieval.runtime as runtime_module

        original_build_provider_config = runtime_module.build_runtime_provider_config
        original_build_sparse = runtime_module.build_runtime_sparse_query_builder
        original_build_hybrid_retriever = runtime_module.build_runtime_hybrid_retriever
        original_build_hybrid_service = runtime_module.build_runtime_hybrid_retrieval_service
        original_build_vector_store = runtime_module.build_runtime_vector_store_client

        captured = {}

        class StubVectorStoreClient:
            def assert_hybrid_collection_ready(self, *, dense_vector_name, sparse_vector_name):
                captured["ready_check"] = (dense_vector_name, sparse_vector_name)

        try:
            runtime_module.build_runtime_provider_config = lambda: type(
                "Cfg",
                (),
                {
                    "qdrant_collection_name": "mai-kb",
                    "qdrant_dense_vector_name": "dense-runtime",
                    "qdrant_sparse_vector_name": "sparse-runtime",
                },
            )()
            runtime_module.build_runtime_sparse_query_builder = lambda: "builder"
            runtime_module.build_runtime_vector_store_client = lambda **kwargs: StubVectorStoreClient()

            def fake_build_hybrid_retriever(**kwargs):
                captured["runtime_config"] = kwargs["runtime_config"]
                return "retriever"

            runtime_module.build_runtime_hybrid_retriever = fake_build_hybrid_retriever
            runtime_module.build_runtime_hybrid_retrieval_service = lambda **kwargs: kwargs

            service = runtime_module.build_runtime_text_retrieval_service()
        finally:
            runtime_module.build_runtime_provider_config = original_build_provider_config
            runtime_module.build_runtime_sparse_query_builder = original_build_sparse
            runtime_module.build_runtime_hybrid_retriever = original_build_hybrid_retriever
            runtime_module.build_runtime_hybrid_retrieval_service = original_build_hybrid_service
            runtime_module.build_runtime_vector_store_client = original_build_vector_store

        self.assertEqual(captured["runtime_config"].dense_vector_name, "dense-runtime")
        self.assertEqual(captured["runtime_config"].sparse_vector_name, "sparse-runtime")
        self.assertEqual(captured["ready_check"], ("dense-runtime", "sparse-runtime"))
        self.assertEqual(service["hybrid_retriever"], "retriever")

    def test_build_runtime_text_retrieval_service_raises_when_collection_lacks_sparse_vector(self):
        import src.rag.retrieval.runtime as runtime_module

        original_build_provider_config = runtime_module.build_runtime_provider_config
        original_build_vector_store = runtime_module.build_runtime_vector_store_client

        class StubVectorStoreClient:
            def assert_hybrid_collection_ready(self, *, dense_vector_name, sparse_vector_name):
                raise ValueError(
                    "Qdrant collection 'mai-kb' is missing sparse vector 'sparse-runtime'."
                )

        try:
            runtime_module.build_runtime_provider_config = lambda: type(
                "Cfg",
                (),
                {
                    "qdrant_collection_name": "mai-kb",
                    "qdrant_dense_vector_name": "dense-runtime",
                    "qdrant_sparse_vector_name": "sparse-runtime",
                },
            )()
            runtime_module.build_runtime_vector_store_client = lambda **kwargs: StubVectorStoreClient()

            with self.assertRaisesRegex(
                ValueError,
                "Qdrant collection 'mai-kb' is missing sparse vector 'sparse-runtime'\\.",
            ):
                runtime_module.build_runtime_text_retrieval_service()
        finally:
            runtime_module.build_runtime_provider_config = original_build_provider_config
            runtime_module.build_runtime_vector_store_client = original_build_vector_store
