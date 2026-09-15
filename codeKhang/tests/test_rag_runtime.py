import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.integration.types import BranchType, PreparedBranchInput
from src.rag.normalization.types import NormalizationResult, QueryType
from src.rag.planning.types import PlannedQuery, QueryIntent
from src.rag.retrieval.types import RetrievalFilter, RetrievalRequest


def test_text_runtime_result_wraps_prepared_planned_and_request():
    from src.rag.runtime.types import TextRuntimeResult

    prepared = PreparedBranchInput(
        branch=BranchType.TEXT_ONLY_BRANCH,
        normalization=NormalizationResult(
            normalized_query="cach tuoi mai",
            query_type=QueryType.CARE_QUERY,
            normalization_notes=("no_major_change",),
        ),
    )
    planned = PlannedQuery(
        branch=prepared.branch,
        normalized_query="cach tuoi mai",
        query_type=QueryType.CARE_QUERY,
        intent=QueryIntent.CARE_GUIDANCE,
        query_terms=("tuoi mai",),
        entity_hints=(),
        subqueries=("cach tuoi mai",),
    )
    request = RetrievalRequest(
        query_text="cach tuoi mai",
        branch=prepared.branch,
        query_type=QueryType.CARE_QUERY,
        intent=QueryIntent.CARE_GUIDANCE,
        query_terms=("tuoi mai",),
        entity_hints=(),
        topic_hints=("tuoi mai", "care"),
        filters=(),
        subqueries=("cach tuoi mai",),
    )

    result = TextRuntimeResult(
        prepared=prepared,
        planned=planned,
        retrieval_request=request,
    )

    assert result.prepared.branch is BranchType.TEXT_ONLY_BRANCH
    assert result.planned.intent is QueryIntent.CARE_GUIDANCE
    assert result.retrieval_request.query_terms == ("tuoi mai",)


class TextRuntimeResultPhase5Tests(unittest.TestCase):
    def test_text_runtime_result_defaults_retrieval_bundle_to_none(self):
        from src.rag.runtime.types import TextRuntimeResult

        prepared = PreparedBranchInput(
            branch=BranchType.TEXT_ONLY_BRANCH,
            normalization=NormalizationResult(
                normalized_query="cach tuoi mai",
                query_type=QueryType.CARE_QUERY,
                normalization_notes=("no_major_change",),
            ),
        )
        planned = PlannedQuery(
            branch=prepared.branch,
            normalized_query="cach tuoi mai",
            query_type=QueryType.CARE_QUERY,
            intent=QueryIntent.CARE_GUIDANCE,
            query_terms=("tuoi mai",),
            entity_hints=(),
            subqueries=("cach tuoi mai",),
        )
        request = RetrievalRequest(
            query_text="cach tuoi mai",
            branch=prepared.branch,
            query_type=QueryType.CARE_QUERY,
            intent=QueryIntent.CARE_GUIDANCE,
            query_terms=("tuoi mai",),
            entity_hints=(),
            topic_hints=("tuoi mai", "care"),
            filters=(),
            subqueries=("cach tuoi mai",),
        )

        result = TextRuntimeResult(
            prepared=prepared,
            planned=planned,
            retrieval_request=request,
        )

        self.assertIsNone(result.retrieval_bundle)

    def test_text_runtime_result_carries_optional_retrieval_bundle(self):
        from src.rag.retrieval import MergedCandidate, RetrievalBundle
        from src.rag.runtime.types import TextRuntimeResult

        prepared = PreparedBranchInput(
            branch=BranchType.TEXT_ONLY_BRANCH,
            normalization=NormalizationResult(
                normalized_query="cach tuoi mai",
                query_type=QueryType.CARE_QUERY,
                normalization_notes=("no_major_change",),
            ),
        )
        planned = PlannedQuery(
            branch=prepared.branch,
            normalized_query="cach tuoi mai",
            query_type=QueryType.CARE_QUERY,
            intent=QueryIntent.CARE_GUIDANCE,
            query_terms=("tuoi mai",),
            entity_hints=(),
            subqueries=("cach tuoi mai",),
        )
        request = RetrievalRequest(
            query_text="cach tuoi mai",
            branch=prepared.branch,
            query_type=QueryType.CARE_QUERY,
            intent=QueryIntent.CARE_GUIDANCE,
            query_terms=("tuoi mai",),
            entity_hints=(),
            topic_hints=("tuoi mai", "care"),
            filters=(),
            subqueries=("cach tuoi mai",),
        )
        bundle = RetrievalBundle(
            retrieval_request=request,
            graph_candidates=(),
            hybrid_candidates=(),
            merged_candidates=(
                MergedCandidate(
                    chunk_id="chunk-1",
                    sources=("dense", "graph"),
                    graph_match_reasons=("care_guide",),
                    raw_scores={"dense": 0.92},
                ),
            ),
        )

        result = TextRuntimeResult(
            prepared=prepared,
            planned=planned,
            retrieval_request=request,
            retrieval_bundle=bundle,
        )

        self.assertIs(result.retrieval_bundle, bundle)
        self.assertEqual(
            result.retrieval_bundle.merged_candidates[0].chunk_id,
            "chunk-1",
        )
        self.assertEqual(
            result.retrieval_bundle.merged_candidates[0].sources,
            ("dense", "graph"),
        )


def test_image_runtime_result_wraps_prepared_planned_request_and_disease_hints():
    from src.rag.runtime.types import ImageRuntimeResult

    prepared = PreparedBranchInput(
        branch=BranchType.IMAGE_BRANCH,
        normalization=NormalizationResult(
            normalized_query="la mai bi dom den",
            query_type=QueryType.DISEASE_QUERY,
            normalization_notes=("image_safe_norm",),
        ),
    )
    planned = PlannedQuery(
        branch=prepared.branch,
        normalized_query="la mai bi dom den",
        query_type=QueryType.DISEASE_QUERY,
        intent=QueryIntent.DISEASE_LOOKUP,
        query_terms=("dom den la mai",),
        entity_hints=(),
        subqueries=("benh dom den tren la mai",),
    )
    request = RetrievalRequest(
        query_text="la mai bi dom den",
        branch=prepared.branch,
        query_type=QueryType.DISEASE_QUERY,
        intent=QueryIntent.DISEASE_LOOKUP,
        query_terms=("dom den la mai",),
        entity_hints=(),
        topic_hints=("dom den", "care"),
        filters=(),
        subqueries=("benh dom den tren la mai",),
    )

    result = ImageRuntimeResult(
        prepared=prepared,
        planned=planned,
        retrieval_request=request,
        disease_hints=("dom den", "ri sat"),
    )

    assert result.prepared.branch is BranchType.IMAGE_BRANCH
    assert result.planned.intent is QueryIntent.DISEASE_LOOKUP
    assert result.retrieval_request.query_terms == ("dom den la mai",)
    assert result.disease_hints == ("dom den", "ri sat")


def test_prepare_text_runtime_result_runs_prepare_plan_and_request_in_order():
    from src.rag.runtime.service import RAGRuntimeService

    class StubPreprocessor:
        def __init__(self, prepared):
            self.prepared = prepared
            self.calls = []

        def prepare(self, raw_text: str, has_image: bool):
            self.calls.append((raw_text, has_image))
            return self.prepared

    class StubPlanner:
        def __init__(self, planned):
            self.planned = planned
            self.calls = []

        def plan(self, prepared):
            self.calls.append(prepared)
            return self.planned

    prepared = PreparedBranchInput(
        branch=BranchType.TEXT_ONLY_BRANCH,
        normalization=NormalizationResult(
            normalized_query="cach tuoi mai",
            query_type=QueryType.CARE_QUERY,
            normalization_notes=("no_major_change",),
        ),
    )
    planned = PlannedQuery(
        branch=prepared.branch,
        normalized_query="cach tuoi mai",
        query_type=QueryType.CARE_QUERY,
        intent=QueryIntent.CARE_GUIDANCE,
        query_terms=("tuoi mai",),
        entity_hints=(),
        subqueries=("cach tuoi mai",),
    )

    preprocessor = StubPreprocessor(prepared)
    planner = StubPlanner(planned)
    runtime = RAGRuntimeService(preprocessor=preprocessor, planner=planner)

    result = runtime.prepare_text_runtime_result("cach tuoi mai")

    assert preprocessor.calls == [("cach tuoi mai", False)]
    assert planner.calls == [prepared]
    assert result.prepared is prepared
    assert result.planned is planned
    assert result.retrieval_request.query_text == "cach tuoi mai"
    assert result.retrieval_request.query_terms == ("tuoi mai",)


def test_prepare_text_runtime_result_leaves_bundle_none_without_phase5_wiring():
    from src.rag.runtime.service import RAGRuntimeService

    class StubPreprocessor:
        def __init__(self, prepared):
            self.prepared = prepared

        def prepare(self, raw_text: str, has_image: bool):
            return self.prepared

    class StubPlanner:
        def __init__(self, planned):
            self.planned = planned

        def plan(self, prepared):
            return self.planned

    prepared = PreparedBranchInput(
        branch=BranchType.TEXT_ONLY_BRANCH,
        normalization=NormalizationResult(
            normalized_query="cach tuoi mai",
            query_type=QueryType.CARE_QUERY,
            normalization_notes=("no_major_change",),
        ),
    )
    planned = PlannedQuery(
        branch=prepared.branch,
        normalized_query="cach tuoi mai",
        query_type=QueryType.CARE_QUERY,
        intent=QueryIntent.CARE_GUIDANCE,
        query_terms=("tuoi mai",),
        entity_hints=(),
        subqueries=("cach tuoi mai",),
    )

    runtime = RAGRuntimeService(
        preprocessor=StubPreprocessor(prepared),
        planner=StubPlanner(planned),
    )

    result = runtime.prepare_text_runtime_result("cach tuoi mai")

    assert result.retrieval_bundle is None


class RuntimeServicePhase5ValidationTests(unittest.TestCase):
    def test_prepare_text_runtime_result_raises_for_partial_phase5_wiring_with_only_graph_provider(
        self,
    ):
        from src.rag.runtime.service import RAGRuntimeService

        class StubPreprocessor:
            def __init__(self, prepared):
                self.prepared = prepared

            def prepare(self, raw_text: str, has_image: bool):
                return self.prepared

        class StubPlanner:
            def __init__(self, planned):
                self.planned = planned

            def plan(self, prepared):
                return self.planned

        prepared = PreparedBranchInput(
            branch=BranchType.TEXT_ONLY_BRANCH,
            normalization=NormalizationResult(
                normalized_query="cach tuoi mai",
                query_type=QueryType.CARE_QUERY,
                normalization_notes=("no_major_change",),
            ),
        )
        planned = PlannedQuery(
            branch=prepared.branch,
            normalized_query="cach tuoi mai",
            query_type=QueryType.CARE_QUERY,
            intent=QueryIntent.CARE_GUIDANCE,
            query_terms=("tuoi mai",),
            entity_hints=(),
            subqueries=("cach tuoi mai",),
        )

        runtime = RAGRuntimeService(
            preprocessor=StubPreprocessor(prepared),
            planner=StubPlanner(planned),
            graph_chunk_provider=object(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "text_retrieval_service and graph_chunk_provider must be provided together",
        ):
            runtime.prepare_text_runtime_result("cach tuoi mai")


def test_prepare_image_runtime_result_uses_has_image_true_and_passes_disease_hints():
    from src.rag.runtime.service import RAGRuntimeService

    class StubPreprocessor:
        def __init__(self, prepared):
            self.prepared = prepared
            self.calls = []

        def prepare(self, raw_text: str, has_image: bool):
            self.calls.append((raw_text, has_image))
            return self.prepared

    class StubPlanner:
        def __init__(self, planned):
            self.planned = planned
            self.calls = []

        def plan(self, prepared):
            self.calls.append(prepared)
            return self.planned

    prepared = PreparedBranchInput(
        branch=BranchType.IMAGE_BRANCH,
        normalization=NormalizationResult(
            normalized_query="la mai bi gi va xu ly sao",
            query_type=QueryType.MIXED_QUERY,
            normalization_notes=("rewritten_for_retrieval",),
        ),
    )
    planned = PlannedQuery(
        branch=prepared.branch,
        normalized_query="la mai bi gi va xu ly sao",
        query_type=QueryType.MIXED_QUERY,
        intent=QueryIntent.MIXED_REASONING,
        query_terms=("mai", "xu ly"),
        entity_hints=(),
        subqueries=("la mai bi gi", "xu ly sao"),
    )

    preprocessor = StubPreprocessor(prepared)
    planner = StubPlanner(planned)
    runtime = RAGRuntimeService(preprocessor=preprocessor, planner=planner)

    result = runtime.prepare_image_runtime_result(
        raw_text="la mai bi gi va xu ly sao",
        disease_hints=("ri sat", "dom la"),
    )

    assert preprocessor.calls == [("la mai bi gi va xu ly sao", True)]
    assert planner.calls == [prepared]
    assert result.prepared is prepared
    assert result.planned is planned
    assert result.disease_hints == ("ri sat", "dom la")
    assert result.retrieval_request.filters[0].value == "ri sat"
    assert result.retrieval_request.filters[1].value == "dom la"


def test_prepare_image_runtime_result_stores_normalized_disease_hints():
    from src.rag.runtime.service import RAGRuntimeService

    class StubPreprocessor:
        def __init__(self, prepared):
            self.prepared = prepared

        def prepare(self, raw_text: str, has_image: bool):
            return self.prepared

    class StubPlanner:
        def __init__(self, planned):
            self.planned = planned

        def plan(self, prepared):
            return self.planned

    prepared = PreparedBranchInput(
        branch=BranchType.IMAGE_BRANCH,
        normalization=NormalizationResult(
            normalized_query="la mai bi gi",
            query_type=QueryType.DISEASE_QUERY,
            normalization_notes=("image_safe_norm",),
        ),
    )
    planned = PlannedQuery(
        branch=prepared.branch,
        normalized_query="la mai bi gi",
        query_type=QueryType.DISEASE_QUERY,
        intent=QueryIntent.DISEASE_LOOKUP,
        query_terms=("la mai",),
        entity_hints=(),
        subqueries=("la mai bi gi",),
    )

    runtime = RAGRuntimeService(
        preprocessor=StubPreprocessor(prepared),
        planner=StubPlanner(planned),
    )

    result = runtime.prepare_image_runtime_result(
        raw_text="la mai bi gi",
        disease_hints=("  ri   sat  ", "dom la", "ri sat", ""),
    )

    assert result.disease_hints == ("ri sat", "dom la")
    assert tuple(item.value for item in result.retrieval_request.filters) == (
        "ri sat",
        "dom la",
    )


def test_prepare_image_runtime_result_ignores_non_disease_filters_in_stored_hints():
    import src.rag.runtime.service as runtime_service_module
    from src.rag.runtime.service import RAGRuntimeService

    class StubPreprocessor:
        def __init__(self, prepared):
            self.prepared = prepared

        def prepare(self, raw_text: str, has_image: bool):
            return self.prepared

    class StubPlanner:
        def __init__(self, planned):
            self.planned = planned

        def plan(self, prepared):
            return self.planned

    prepared = PreparedBranchInput(
        branch=BranchType.IMAGE_BRANCH,
        normalization=NormalizationResult(
            normalized_query="la mai bi gi",
            query_type=QueryType.DISEASE_QUERY,
            normalization_notes=("image_safe_norm",),
        ),
    )
    planned = PlannedQuery(
        branch=prepared.branch,
        normalized_query="la mai bi gi",
        query_type=QueryType.DISEASE_QUERY,
        intent=QueryIntent.DISEASE_LOOKUP,
        query_terms=("la mai",),
        entity_hints=(),
        subqueries=("la mai bi gi",),
    )
    fake_request = RetrievalRequest(
        query_text="la mai bi gi",
        branch=prepared.branch,
        query_type=QueryType.DISEASE_QUERY,
        intent=QueryIntent.DISEASE_LOOKUP,
        query_terms=("la mai",),
        entity_hints=(),
        topic_hints=("la mai",),
        filters=(
            RetrievalFilter(field="disease_name", value="ri sat"),
            RetrievalFilter(field="severity", value="nang"),
            RetrievalFilter(field="disease_name", value="dom la"),
        ),
        subqueries=("la mai bi gi",),
    )

    original_builder = runtime_service_module.build_retrieval_request

    def fake_builder(planned_query, disease_hints=()):
        assert planned_query is planned
        assert disease_hints == ("ri sat", "dom la")
        return fake_request

    runtime_service_module.build_retrieval_request = fake_builder
    try:
        runtime = RAGRuntimeService(
            preprocessor=StubPreprocessor(prepared),
            planner=StubPlanner(planned),
        )
        result = runtime.prepare_image_runtime_result(
            raw_text="la mai bi gi",
            disease_hints=("ri sat", "dom la"),
        )
    finally:
        runtime_service_module.build_retrieval_request = original_builder

    assert result.retrieval_request is fake_request
    assert result.disease_hints == ("ri sat", "dom la")


def test_build_runtime_service_composes_preprocessor_and_planner_factories():
    from src.rag.runtime.runtime import build_runtime_service

    class StubRuntime:
        def __init__(self, preprocessor, planner):
            self.preprocessor = preprocessor
            self.planner = planner

    class StubPreprocessor:
        pass

    class StubPlanner:
        pass

    preprocessor = StubPreprocessor()
    planner = StubPlanner()
    preprocessor_factory_calls = []
    planner_factory_calls = []

    def preprocessor_factory():
        preprocessor_factory_calls.append("called")
        return preprocessor

    def planner_factory():
        planner_factory_calls.append("called")
        return planner

    runtime = build_runtime_service(
        preprocessor_factory=preprocessor_factory,
        planner_factory=planner_factory,
        runtime_service_cls=StubRuntime,
    )

    assert runtime.__class__ is StubRuntime
    assert preprocessor_factory_calls == ["called"]
    assert planner_factory_calls == ["called"]
    assert runtime.preprocessor is preprocessor
    assert runtime.planner is planner


def test_runtime_package_exports_public_api():
    import src.rag.runtime as runtime_pkg
    from src.rag.runtime.predict_bridge import (
        build_image_runtime_payload,
        extract_disease_hints_from_detection,
    )
    from src.rag.runtime.runtime import build_runtime_service
    from src.rag.runtime.service import RAGRuntimeService
    from src.rag.runtime.text_probe import prepare_text_runtime_probe
    from src.rag.runtime.types import ImageRuntimeResult, TextRuntimeResult

    assert runtime_pkg.ImageRuntimeResult is ImageRuntimeResult
    assert runtime_pkg.RAGRuntimeService is RAGRuntimeService
    assert runtime_pkg.TextRuntimeResult is TextRuntimeResult
    assert runtime_pkg.build_image_runtime_payload is build_image_runtime_payload
    assert runtime_pkg.build_runtime_service is build_runtime_service
    assert (
        runtime_pkg.extract_disease_hints_from_detection
        is extract_disease_hints_from_detection
    )
    assert runtime_pkg.prepare_text_runtime_probe is prepare_text_runtime_probe
    assert runtime_pkg.__all__ == [
        "ImageRuntimeResult",
        "RAGRuntimeService",
        "TextRuntimeResult",
        "build_image_runtime_payload",
        "build_runtime_service",
        "extract_disease_hints_from_detection",
        "prepare_text_runtime_probe",
    ]
