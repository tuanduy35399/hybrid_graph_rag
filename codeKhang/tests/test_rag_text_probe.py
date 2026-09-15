import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.integration.types import BranchType, PreparedBranchInput
from src.rag.normalization.types import NormalizationResult, QueryType
from src.rag.planning.types import PlannedQuery, QueryIntent
from src.rag.retrieval.types import RetrievalRequest
from src.rag.runtime.types import TextRuntimeResult


def test_prepare_text_runtime_probe_returns_text_runtime_result():
    from src.rag.runtime.text_probe import prepare_text_runtime_probe

    class StubRuntimeService:
        def __init__(self):
            self.calls = []

        def prepare_text_runtime_result(self, raw_text: str):
            self.calls.append(raw_text)
            prepared = PreparedBranchInput(
                branch=BranchType.TEXT_ONLY_BRANCH,
                normalization=NormalizationResult(
                    normalized_query="cach tuoi mai dung",
                    query_type=QueryType.CARE_QUERY,
                    normalization_notes=("rewritten_for_retrieval",),
                ),
            )
            planned = PlannedQuery(
                branch=BranchType.TEXT_ONLY_BRANCH,
                normalized_query="cach tuoi mai dung",
                query_type=QueryType.CARE_QUERY,
                intent=QueryIntent.CARE_GUIDANCE,
                query_terms=("tuoi mai",),
                entity_hints=(),
                subqueries=("cach tuoi mai dung",),
            )
            request = RetrievalRequest(
                query_text="cach tuoi mai dung",
                branch=BranchType.TEXT_ONLY_BRANCH,
                query_type=QueryType.CARE_QUERY,
                intent=QueryIntent.CARE_GUIDANCE,
                query_terms=("tuoi mai",),
                entity_hints=(),
                topic_hints=("tuoi mai", "care"),
                filters=(),
                subqueries=("cach tuoi mai dung",),
            )
            return TextRuntimeResult(
                prepared=prepared,
                planned=planned,
                retrieval_request=request,
            )

    runtime = StubRuntimeService()
    result = prepare_text_runtime_probe("cach tuoi mai dung", runtime_service=runtime)

    assert runtime.calls == ["cach tuoi mai dung"]
    assert result.prepared.branch is BranchType.TEXT_ONLY_BRANCH
    assert result.retrieval_request.query_terms == ("tuoi mai",)


def test_runtime_package_exports_probe_and_bridge_helpers():
    import src.rag.runtime as runtime_pkg
    from src.rag.runtime.predict_bridge import (
        build_image_runtime_payload,
        extract_disease_hints_from_detection,
    )
    from src.rag.runtime.text_probe import prepare_text_runtime_probe

    assert runtime_pkg.build_image_runtime_payload is build_image_runtime_payload
    assert (
        runtime_pkg.extract_disease_hints_from_detection
        is extract_disease_hints_from_detection
    )
    assert runtime_pkg.prepare_text_runtime_probe is prepare_text_runtime_probe
