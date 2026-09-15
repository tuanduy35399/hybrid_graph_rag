import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class Phase6RuntimeWiringTests(unittest.TestCase):
    def test_answer_text_query_runs_existing_runtime_path_then_calls_answering_service(
        self,
    ):
        from src.rag.answering.types import TextAnswerResult
        from src.rag.integration.types import BranchType, PreparedBranchInput
        from src.rag.normalization.types import NormalizationResult, QueryType
        from src.rag.planning.types import PlannedQuery, QueryIntent
        from src.rag.retrieval.hybrid_types import RetrievalBundle
        from src.rag.retrieval.types import RetrievalRequest
        from src.rag.runtime.service import RAGRuntimeService

        prepared = PreparedBranchInput(
            branch=BranchType.TEXT_ONLY_BRANCH,
            normalization=NormalizationResult(
                normalized_query="la mai bi dom vang",
                query_type=QueryType.DISEASE_QUERY,
                normalization_notes=("no_major_change",),
            ),
        )
        planned = PlannedQuery(
            branch=prepared.branch,
            normalized_query="la mai bi dom vang",
            query_type=QueryType.DISEASE_QUERY,
            intent=QueryIntent.DISEASE_LOOKUP,
            query_terms=("dom vang",),
            entity_hints=(),
            subqueries=("la mai bi dom vang",),
        )

        class StubPreprocessor:
            def __init__(self):
                self.calls = []

            def prepare(self, raw_text, has_image):
                self.calls.append((raw_text, has_image))
                return prepared

        class StubPlanner:
            def __init__(self):
                self.calls = []

            def plan(self, prepared_input):
                self.calls.append(prepared_input)
                return planned

        class StubGraphChunkProvider:
            def __init__(self):
                self.calls = []

            def get_chunks(self, retrieval_request):
                self.calls.append(retrieval_request)
                return ()

        class StubTextRetrievalService:
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

        class StubAnsweringService:
            def __init__(self):
                self.calls = []

            def answer(self, runtime_result):
                self.calls.append(runtime_result)
                return TextAnswerResult(
                    runtime_result=runtime_result,
                    answer_context=object(),
                    grounded_answer=object(),
                )

        preprocessor = StubPreprocessor()
        planner = StubPlanner()
        graph_provider = StubGraphChunkProvider()
        retrieval_service = StubTextRetrievalService()
        answering_service = StubAnsweringService()

        runtime = RAGRuntimeService(
            preprocessor=preprocessor,
            planner=planner,
            text_retrieval_service=retrieval_service,
            graph_chunk_provider=graph_provider,
            text_answering_service=answering_service,
        )

        result = runtime.answer_text_query("la mai bi dom vang")

        self.assertEqual(preprocessor.calls, [("la mai bi dom vang", False)])
        self.assertEqual(planner.calls, [prepared])
        self.assertEqual(len(graph_provider.calls), 1)
        self.assertEqual(len(retrieval_service.calls), 1)
        self.assertEqual(len(answering_service.calls), 1)
        self.assertIs(answering_service.calls[0], result.runtime_result)

    def test_answer_text_query_requires_text_answering_service(self):
        from src.rag.runtime.service import RAGRuntimeService

        runtime = RAGRuntimeService(preprocessor=object(), planner=object())

        with self.assertRaisesRegex(
            ValueError,
            "text_answering_service must be provided for text answer synthesis",
        ):
            runtime.answer_text_query("query")

    def test_answer_text_query_requires_retrieval_bundle_before_answer_synthesis(
        self,
    ):
        from src.rag.integration.types import BranchType, PreparedBranchInput
        from src.rag.normalization.types import NormalizationResult, QueryType
        from src.rag.planning.types import PlannedQuery, QueryIntent
        from src.rag.runtime.service import RAGRuntimeService

        prepared = PreparedBranchInput(
            branch=BranchType.TEXT_ONLY_BRANCH,
            normalization=NormalizationResult(
                normalized_query="la mai",
                query_type=QueryType.CARE_QUERY,
                normalization_notes=("no_major_change",),
            ),
        )
        planned = PlannedQuery(
            branch=prepared.branch,
            normalized_query="la mai",
            query_type=QueryType.CARE_QUERY,
            intent=QueryIntent.CARE_GUIDANCE,
            query_terms=("la mai",),
            entity_hints=(),
            subqueries=("la mai",),
        )

        class StubPreprocessor:
            def prepare(self, raw_text, has_image):
                return prepared

        class StubPlanner:
            def plan(self, prepared_input):
                return planned

        class StubAnsweringService:
            def answer(self, runtime_result):
                raise AssertionError(
                    "answer() should not run when retrieval_bundle is missing"
                )

        runtime = RAGRuntimeService(
            preprocessor=StubPreprocessor(),
            planner=StubPlanner(),
            text_answering_service=StubAnsweringService(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "text runtime answer path requires retrieval_bundle before answer synthesis",
        ):
            runtime.answer_text_query("la mai")

    def test_build_runtime_service_composes_phase6_answering_dependency_only_when_all_factories_are_present(
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
                text_answering_service=None,
            ):
                self.preprocessor = preprocessor
                self.planner = planner
                self.text_retrieval_service = text_retrieval_service
                self.graph_chunk_provider = graph_chunk_provider
                self.text_answering_service = text_answering_service

        runtime = build_runtime_service(
            preprocessor_factory=lambda: "pre",
            planner_factory=lambda: "plan",
            runtime_service_cls=StubRuntime,
            text_retrieval_service_factory=lambda: "retriever",
            graph_chunk_provider_factory=lambda: "graph-provider",
            text_answering_service_factory=lambda: "answerer",
        )

        self.assertEqual(runtime.preprocessor, "pre")
        self.assertEqual(runtime.planner, "plan")
        self.assertEqual(runtime.text_retrieval_service, "retriever")
        self.assertEqual(runtime.graph_chunk_provider, "graph-provider")
        self.assertEqual(runtime.text_answering_service, "answerer")

    def test_build_runtime_service_rejects_phase6_answering_factory_without_phase5_retrieval_factories(
        self,
    ):
        from src.rag.runtime.runtime import build_runtime_service

        with self.assertRaisesRegex(
            ValueError,
            "text_answering_service_factory requires text_retrieval_service_factory and graph_chunk_provider_factory",
        ):
            build_runtime_service(
                preprocessor_factory=lambda: "pre",
                planner_factory=lambda: "plan",
                text_answering_service_factory=lambda: "answerer",
            )

    def test_build_runtime_service_rejects_partial_phase5_retrieval_factories(self):
        from src.rag.runtime.runtime import build_runtime_service

        with self.subTest("retrieval factory without graph factory"):
            with self.assertRaisesRegex(
                ValueError,
                "text_retrieval_service_factory and graph_chunk_provider_factory must be provided together",
            ):
                build_runtime_service(
                    preprocessor_factory=lambda: "pre",
                    planner_factory=lambda: "plan",
                    text_retrieval_service_factory=lambda: "retriever",
                )

        with self.subTest("graph factory without retrieval factory"):
            with self.assertRaisesRegex(
                ValueError,
                "text_retrieval_service_factory and graph_chunk_provider_factory must be provided together",
            ):
                build_runtime_service(
                    preprocessor_factory=lambda: "pre",
                    planner_factory=lambda: "plan",
                    graph_chunk_provider_factory=lambda: "graph-provider",
                )

    def test_build_runtime_service_can_build_text_answer_path_and_run_answer_text_query(
        self,
    ):
        from src.rag.answering.types import TextAnswerResult
        from src.rag.integration.types import BranchType, PreparedBranchInput
        from src.rag.normalization.types import NormalizationResult, QueryType
        from src.rag.planning.types import PlannedQuery, QueryIntent
        from src.rag.retrieval.hybrid_types import RetrievalBundle
        from src.rag.runtime.runtime import build_runtime_service

        prepared = PreparedBranchInput(
            branch=BranchType.TEXT_ONLY_BRANCH,
            normalization=NormalizationResult(
                normalized_query="la mai vang la",
                query_type=QueryType.DISEASE_QUERY,
                normalization_notes=("no_major_change",),
            ),
        )
        planned = PlannedQuery(
            branch=prepared.branch,
            normalized_query="la mai vang la",
            query_type=QueryType.DISEASE_QUERY,
            intent=QueryIntent.DISEASE_LOOKUP,
            query_terms=("vang la",),
            entity_hints=(),
            subqueries=("la mai vang la",),
        )

        class StubPreprocessor:
            def __init__(self):
                self.calls = []

            def prepare(self, raw_text, has_image):
                self.calls.append((raw_text, has_image))
                return prepared

        class StubPlanner:
            def __init__(self):
                self.calls = []

            def plan(self, prepared_input):
                self.calls.append(prepared_input)
                return planned

        class StubGraphChunkProvider:
            def __init__(self):
                self.calls = []

            def get_chunks(self, retrieval_request):
                self.calls.append(retrieval_request)
                return ()

        class StubTextRetrievalService:
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

        class StubAnsweringService:
            def __init__(self):
                self.calls = []

            def answer(self, runtime_result):
                self.calls.append(runtime_result)
                return TextAnswerResult(
                    runtime_result=runtime_result,
                    answer_context=object(),
                    grounded_answer=object(),
                )

        preprocessor = StubPreprocessor()
        planner = StubPlanner()
        graph_provider = StubGraphChunkProvider()
        retrieval_service = StubTextRetrievalService()
        answering_service = StubAnsweringService()

        runtime = build_runtime_service(
            preprocessor_factory=lambda: preprocessor,
            planner_factory=lambda: planner,
            text_retrieval_service_factory=lambda: retrieval_service,
            graph_chunk_provider_factory=lambda: graph_provider,
            text_answering_service_factory=lambda: answering_service,
        )

        result = runtime.answer_text_query("la mai vang la")

        self.assertEqual(preprocessor.calls, [("la mai vang la", False)])
        self.assertEqual(planner.calls, [prepared])
        self.assertEqual(len(graph_provider.calls), 1)
        self.assertEqual(len(retrieval_service.calls), 1)
        self.assertEqual(len(answering_service.calls), 1)
        self.assertIs(answering_service.calls[0], result.runtime_result)
