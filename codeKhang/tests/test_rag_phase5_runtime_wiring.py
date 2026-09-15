import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.integration.types import BranchType
from src.rag.normalization.types import QueryType
from src.rag.planning.types import EntityHint, QueryIntent
from src.rag.retrieval.provider_types import RetrievedChunk
from src.rag.retrieval.types import RetrievalRequest


def _make_request():
    return RetrievalRequest(
        query_text="la mai bi dom vang la benh gi",
        branch=BranchType.TEXT_ONLY_BRANCH,
        query_type=QueryType.DISEASE_QUERY,
        intent=QueryIntent.DISEASE_LOOKUP,
        query_terms=("dom vang", "ri sat"),
        entity_hints=(EntityHint(kind="disease_name", value="ri_sat"),),
        topic_hints=("symptoms",),
        filters=(),
        subqueries=("la mai bi dom vang la benh gi",),
    )


class HybridRuntimeRetrieverTests(unittest.TestCase):
    def test_retriever_embeds_dense_text_builds_sparse_query_and_calls_hybrid_search(self):
        from src.rag.retrieval.hybrid_runtime import HybridRuntimeRetriever
        from src.rag.retrieval.provider_types import Phase5RuntimeConfig
        from src.rag.retrieval.sparse_types import SparseQuery

        class StubEmbeddingClient:
            def __init__(self):
                self.calls = []

            def embed_texts(self, texts):
                self.calls.append(texts)
                return ((0.1, 0.2),)

        class StubSparseQueryBuilder:
            def __init__(self):
                self.calls = []
                self.result = SparseQuery(indices=(3, 7), values=(0.9, 0.4))

            def build(self, request):
                self.calls.append(request)
                return self.result

        class StubVectorStoreClient:
            def __init__(self):
                self.calls = []
                self.result = (
                    RetrievedChunk(
                        chunk_id="chunk-1",
                        content="La mai co dom vang.",
                        score=0.88,
                        metadata=None,
                    ),
                )

            def search_hybrid(
                self,
                *,
                dense_vector,
                sparse_query,
                filters,
                dense_vector_name,
                sparse_vector_name,
                prefetch_limit,
                top_k,
            ):
                self.calls.append(
                    {
                        "dense_vector": dense_vector,
                        "sparse_query": sparse_query,
                        "filters": filters,
                        "dense_vector_name": dense_vector_name,
                        "sparse_vector_name": sparse_vector_name,
                        "prefetch_limit": prefetch_limit,
                        "top_k": top_k,
                    }
                )
                return self.result

        request = _make_request()
        embedding_client = StubEmbeddingClient()
        sparse_builder = StubSparseQueryBuilder()
        vector_store = StubVectorStoreClient()

        retriever = HybridRuntimeRetriever(
            embedding_client=embedding_client,
            sparse_query_builder=sparse_builder,
            vector_store_client=vector_store,
            runtime_config=Phase5RuntimeConfig(
                collection_name="mai-kb",
                dense_vector_name="dense-default",
                sparse_vector_name="sparse-bm25",
            ),
            prefetch_limit=12,
            top_k=4,
        )

        hits = retriever.retrieve(request)

        self.assertEqual(
            embedding_client.calls,
            [("la mai bi dom vang la benh gi\ndom vang\nri sat",)],
        )
        self.assertEqual(sparse_builder.calls, [request])
        self.assertEqual(
            vector_store.calls,
            [
                {
                    "dense_vector": (0.1, 0.2),
                    "sparse_query": sparse_builder.result,
                    "filters": (),
                    "dense_vector_name": "dense-default",
                    "sparse_vector_name": "sparse-bm25",
                    "prefetch_limit": 12,
                    "top_k": 4,
                }
            ],
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk_id, "chunk-1")
        self.assertEqual(hits[0].sources, ("dense", "sparse"))
        self.assertEqual(dict(hits[0].raw_scores), {"hybrid": 0.88})

    def test_builder_constructs_hybrid_runtime_retriever_with_hybrid_ready_vector_store(self):
        import src.rag.retrieval.runtime as runtime_module
        from src.rag.retrieval.provider_types import Phase5RuntimeConfig

        original_embedding_builder = runtime_module.build_runtime_embedding_client
        original_vector_builder = runtime_module.build_runtime_vector_store_client

        embedding_client = object()
        vector_store_client = object()
        sparse_query_builder = object()
        captured = {}

        try:
            runtime_module.build_runtime_embedding_client = lambda: embedding_client

            def stub_vector_builder(*, hybrid_ready=False, collection_name=None):
                captured["hybrid_ready"] = hybrid_ready
                captured["collection_name"] = collection_name
                return vector_store_client

            runtime_module.build_runtime_vector_store_client = stub_vector_builder

            retriever = runtime_module.build_runtime_hybrid_retriever(
                sparse_query_builder=sparse_query_builder,
                runtime_config=Phase5RuntimeConfig(
                    collection_name="mai-kb",
                    dense_vector_name="dense-default",
                    sparse_vector_name="sparse-bm25",
                ),
                prefetch_limit=9,
                top_k=3,
            )
        finally:
            runtime_module.build_runtime_embedding_client = original_embedding_builder
            runtime_module.build_runtime_vector_store_client = original_vector_builder

        self.assertIs(retriever.embedding_client, embedding_client)
        self.assertIs(retriever.sparse_query_builder, sparse_query_builder)
        self.assertIs(retriever.vector_store_client, vector_store_client)
        self.assertEqual(retriever.prefetch_limit, 9)
        self.assertEqual(retriever.top_k, 3)
        self.assertIs(captured["hybrid_ready"], True)

    def test_builder_constructs_hybrid_runtime_retriever_with_runtime_collection_name(self):
        import src.rag.retrieval.runtime as runtime_module
        from src.rag.retrieval.provider_types import Phase5RuntimeConfig

        original_embedding_builder = runtime_module.build_runtime_embedding_client
        original_vector_builder = runtime_module.build_runtime_vector_store_client

        try:
            runtime_module.build_runtime_embedding_client = lambda: object()

            captured = {}

            def stub_vector_builder(*, hybrid_ready=False, collection_name=None):
                captured["hybrid_ready"] = hybrid_ready
                captured["collection_name"] = collection_name
                return object()

            runtime_module.build_runtime_vector_store_client = stub_vector_builder

            runtime_module.build_runtime_hybrid_retriever(
                sparse_query_builder=object(),
                runtime_config=Phase5RuntimeConfig(
                    collection_name="mai-phase5-runtime",
                    dense_vector_name="dense-default",
                    sparse_vector_name="sparse-bm25",
                ),
            )
        finally:
            runtime_module.build_runtime_embedding_client = original_embedding_builder
            runtime_module.build_runtime_vector_store_client = original_vector_builder

        self.assertEqual(captured["collection_name"], "mai-phase5-runtime")

    def test_build_runtime_hybrid_retrieval_service_uses_graph_retriever_and_injected_hybrid_retriever(
        self,
    ):
        import src.rag.retrieval.runtime as runtime_module

        original_graph_builder = runtime_module.build_runtime_graph_retriever
        graph_retriever = object()
        hybrid_retriever = object()

        try:
            runtime_module.build_runtime_graph_retriever = lambda: graph_retriever
            service = runtime_module.build_runtime_hybrid_retrieval_service(
                hybrid_retriever=hybrid_retriever,
            )
        finally:
            runtime_module.build_runtime_graph_retriever = original_graph_builder

        self.assertIs(service.graph_retriever, graph_retriever)
        self.assertIs(service.hybrid_retriever, hybrid_retriever)


class RuntimeServicePhase5WiringTests(unittest.TestCase):
    def test_build_runtime_service_preserves_legacy_two_arg_runtime_constructor_when_optional_factories_absent(
        self,
    ):
        from src.rag.runtime.runtime import build_runtime_service

        class StubRuntime:
            def __init__(self, preprocessor, planner):
                self.preprocessor = preprocessor
                self.planner = planner

        runtime = build_runtime_service(
            preprocessor_factory=lambda: "pre",
            planner_factory=lambda: "plan",
            runtime_service_cls=StubRuntime,
        )

        self.assertEqual(runtime.preprocessor, "pre")
        self.assertEqual(runtime.planner, "plan")

    def test_build_runtime_service_phase5_path_does_not_require_phase6_answering_factory(
        self,
    ):
        from src.rag.runtime.runtime import build_runtime_service

        class StubRuntime:
            def __init__(
                self,
                preprocessor,
                planner,
                text_retrieval_service=None,
                graph_chunk_provider=None,
            ):
                self.preprocessor = preprocessor
                self.planner = planner
                self.text_retrieval_service = text_retrieval_service
                self.graph_chunk_provider = graph_chunk_provider

        runtime = build_runtime_service(
            preprocessor_factory=lambda: "pre",
            planner_factory=lambda: "plan",
            runtime_service_cls=StubRuntime,
            text_retrieval_service_factory=lambda: "retriever",
            graph_chunk_provider_factory=lambda: "graph-provider",
        )

        self.assertEqual(runtime.preprocessor, "pre")
        self.assertEqual(runtime.planner, "plan")
        self.assertEqual(runtime.text_retrieval_service, "retriever")
        self.assertEqual(runtime.graph_chunk_provider, "graph-provider")

    def test_prepare_text_runtime_result_raises_for_partial_phase5_wiring_with_only_retrieval_service(
        self,
    ):
        from src.rag.integration.types import BranchType, PreparedBranchInput
        from src.rag.normalization.types import NormalizationResult, QueryType
        from src.rag.planning.types import PlannedQuery, QueryIntent
        from src.rag.runtime.service import RAGRuntimeService

        prepared = PreparedBranchInput(
            branch=BranchType.TEXT_ONLY_BRANCH,
            normalization=NormalizationResult(
                normalized_query="query",
                query_type=QueryType.CARE_QUERY,
                normalization_notes=("no_major_change",),
            ),
        )
        planned = PlannedQuery(
            branch=prepared.branch,
            normalized_query="query",
            query_type=QueryType.CARE_QUERY,
            intent=QueryIntent.CARE_GUIDANCE,
            query_terms=("query",),
            entity_hints=(),
            subqueries=("query",),
        )

        class StubPreprocessor:
            def prepare(self, raw_text, has_image):
                return prepared

        class StubPlanner:
            def plan(self, prepared):
                return planned

        runtime = RAGRuntimeService(
            preprocessor=StubPreprocessor(),
            planner=StubPlanner(),
            text_retrieval_service=object(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "text_retrieval_service and graph_chunk_provider must be provided together",
        ):
            runtime.prepare_text_runtime_result("query")

    def test_prepare_text_runtime_result_attaches_phase5_bundle_when_retrieval_service_is_present(
        self,
    ):
        from src.rag.integration.types import BranchType, PreparedBranchInput
        from src.rag.normalization.types import NormalizationResult, QueryType
        from src.rag.planning.types import PlannedQuery, QueryIntent
        from src.rag.retrieval.hybrid_types import RetrievalBundle
        from src.rag.runtime.service import RAGRuntimeService

        prepared = PreparedBranchInput(
            branch=BranchType.TEXT_ONLY_BRANCH,
            normalization=NormalizationResult(
                normalized_query="query",
                query_type=QueryType.CARE_QUERY,
                normalization_notes=("no_major_change",),
            ),
        )
        planned = PlannedQuery(
            branch=prepared.branch,
            normalized_query="query",
            query_type=QueryType.CARE_QUERY,
            intent=QueryIntent.CARE_GUIDANCE,
            query_terms=("query",),
            entity_hints=(),
            subqueries=("query",),
        )

        class StubPreprocessor:
            def prepare(self, raw_text, has_image):
                return prepared

        class StubPlanner:
            def plan(self, prepared):
                return planned

        class StubBundleRetriever:
            def __init__(self):
                self.calls = []

            def retrieve(self, *, retrieval_request, graph_chunks):
                self.calls.append((retrieval_request, graph_chunks))
                return RetrievalBundle(
                    retrieval_request=retrieval_request,
                    graph_candidates=(),
                    hybrid_candidates=(),
                    merged_candidates=(),
                )

        class StubGraphChunkProvider:
            def __init__(self):
                self.calls = []

            def get_chunks(self, retrieval_request):
                self.calls.append(retrieval_request)
                return ("chunk-a", "chunk-b")

        retriever = StubBundleRetriever()
        graph_chunk_provider = StubGraphChunkProvider()
        runtime = RAGRuntimeService(
            preprocessor=StubPreprocessor(),
            planner=StubPlanner(),
            text_retrieval_service=retriever,
            graph_chunk_provider=graph_chunk_provider,
        )

        result = runtime.prepare_text_runtime_result("query")

        self.assertIsNotNone(result.retrieval_bundle)
        self.assertEqual(result.retrieval_bundle.merged_candidates, ())
        self.assertEqual(len(graph_chunk_provider.calls), 1)
        self.assertIs(graph_chunk_provider.calls[0], result.retrieval_request)
        self.assertEqual(
            retriever.calls,
            [(result.retrieval_request, ("chunk-a", "chunk-b"))],
        )

    def test_build_runtime_text_retrieval_service_composes_server_sparse_builder_hybrid_retriever_and_service(self):
        import src.rag.retrieval.runtime as runtime_module

        original_build_config = runtime_module.build_phase5_runtime_config
        original_build_sparse = runtime_module.build_runtime_sparse_query_builder
        original_build_retriever = runtime_module.build_runtime_hybrid_retriever
        original_build_service = runtime_module.build_runtime_hybrid_retrieval_service
        original_build_vector_store = runtime_module.build_runtime_vector_store_client
        original_build_provider_config = runtime_module.build_runtime_provider_config

        captured = {}

        try:
            runtime_module.build_phase5_runtime_config = lambda **kwargs: type(
                "RuntimeConfig",
                (),
                {
                    "collection_name": "mai-kb",
                    "dense_vector_name": "dense-runtime",
                    "sparse_vector_name": "sparse-runtime",
                    "raw_kwargs": kwargs,
                },
            )()
            runtime_module.build_runtime_sparse_query_builder = lambda: "server-sparse-builder"
            runtime_module.build_runtime_provider_config = lambda: type(
                "Cfg",
                (),
                {
                    "qdrant_collection_name": "mai-kb",
                    "qdrant_dense_vector_name": "dense-runtime",
                    "qdrant_sparse_vector_name": "sparse-runtime",
                },
            )()
            runtime_module.build_runtime_vector_store_client = lambda **kwargs: type(
                "VectorStore",
                (),
                {
                    "assert_hybrid_collection_ready": lambda self, **ready_kwargs: captured.update(
                        {"ready_kwargs": ready_kwargs, "vector_store_kwargs": kwargs}
                    )
                },
            )()

            def fake_build_retriever(**kwargs):
                captured["retriever_kwargs"] = kwargs
                return "hybrid-retriever"

            runtime_module.build_runtime_hybrid_retriever = fake_build_retriever
            runtime_module.build_runtime_hybrid_retrieval_service = lambda **kwargs: (
                "retrieval-service",
                kwargs,
            )

            service = runtime_module.build_runtime_text_retrieval_service()
        finally:
            runtime_module.build_phase5_runtime_config = original_build_config
            runtime_module.build_runtime_sparse_query_builder = original_build_sparse
            runtime_module.build_runtime_hybrid_retriever = original_build_retriever
            runtime_module.build_runtime_hybrid_retrieval_service = original_build_service
            runtime_module.build_runtime_vector_store_client = original_build_vector_store
            runtime_module.build_runtime_provider_config = original_build_provider_config

        self.assertEqual(service[0], "retrieval-service")
        self.assertEqual(
            captured["retriever_kwargs"]["sparse_query_builder"],
            "server-sparse-builder",
        )
        self.assertEqual(
            captured["retriever_kwargs"]["runtime_config"].raw_kwargs["collection_name"],
            "mai-kb",
        )
        self.assertEqual(
            captured["ready_kwargs"],
            {
                "dense_vector_name": "dense-runtime",
                "sparse_vector_name": "sparse-runtime",
            },
        )
