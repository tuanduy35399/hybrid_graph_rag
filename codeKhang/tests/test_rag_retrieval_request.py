import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.integration.types import BranchType
from src.rag.normalization.types import QueryType
from src.rag.planning.types import EntityHint, PlannedQuery, QueryIntent


def test_retrieval_filter_preserves_field_and_value():
    from src.rag.retrieval.types import RetrievalFilter

    retrieval_filter = RetrievalFilter(field="topic", value="symptoms")

    assert retrieval_filter.field == "topic"
    assert retrieval_filter.value == "symptoms"


def test_retrieval_request_wraps_branch_query_and_filters():
    from src.rag.retrieval.types import RetrievalFilter, RetrievalRequest

    request = RetrievalRequest(
        query_text="la mai bi dom den la benh gi",
        branch=BranchType.TEXT_ONLY_BRANCH,
        query_type=QueryType.DISEASE_QUERY,
        intent=QueryIntent.DISEASE_LOOKUP,
        query_terms=("mai", "dom den"),
        entity_hints=(EntityHint(kind="symptom", value="dom den"),),
        topic_hints=("symptoms", "treatment"),
        filters=(RetrievalFilter(field="doc_type", value="disease_guide"),),
        subqueries=("la mai bi dom den la benh gi",),
    )

    assert request.branch is BranchType.TEXT_ONLY_BRANCH
    assert request.query_type is QueryType.DISEASE_QUERY
    assert request.intent is QueryIntent.DISEASE_LOOKUP
    assert request.query_terms == ("mai", "dom den")
    assert request.filters[0].field == "doc_type"


def test_build_retrieval_request_maps_text_only_disease_query_to_disease_topics():
    from src.rag.retrieval.request_builder import build_retrieval_request

    planned = PlannedQuery(
        branch=BranchType.TEXT_ONLY_BRANCH,
        normalized_query="la mai bi dom den la benh gi",
        query_type=QueryType.DISEASE_QUERY,
        intent=QueryIntent.DISEASE_LOOKUP,
        query_terms=("mai", "benh", "dom den"),
        entity_hints=(
            EntityHint(kind="plant_name", value="mai"),
            EntityHint(kind="symptom", value="dom den"),
        ),
        subqueries=("la mai bi dom den la benh gi",),
    )

    request = build_retrieval_request(planned)

    assert request.query_text == "la mai bi dom den la benh gi"
    assert request.query_terms == ("mai", "benh", "dom den")
    assert request.topic_hints == ("mai", "benh", "dom den", "symptoms", "treatment")
    assert request.filters == ()


def test_build_retrieval_request_adds_disease_filters_for_image_branch_post_prediction():
    from src.rag.retrieval.request_builder import build_retrieval_request

    planned = PlannedQuery(
        branch=BranchType.IMAGE_BRANCH,
        normalized_query="la mai bi gi va xu ly sao",
        query_type=QueryType.MIXED_QUERY,
        intent=QueryIntent.MIXED_REASONING,
        query_terms=("mai", "xu ly"),
        entity_hints=(EntityHint(kind="plant_name", value="mai"),),
        subqueries=("la mai bi gi", "xu ly sao"),
    )

    request = build_retrieval_request(
        planned,
        disease_hints=("ri sat", "dom la"),
    )

    assert request.branch is BranchType.IMAGE_BRANCH
    assert request.query_terms == ("mai", "xu ly")
    assert request.topic_hints == ("mai", "xu ly", "ri sat", "dom la", "symptoms", "care")
    assert request.filters[0].field == "disease_name"
    assert request.filters[0].value == "ri sat"
    assert request.filters[1].value == "dom la"
    assert any(
        hint.kind == "disease_name" and hint.value == "ri sat"
        for hint in request.entity_hints
    )


def test_build_retrieval_request_deduplicates_repeated_disease_hints():
    from src.rag.retrieval.request_builder import build_retrieval_request

    planned = PlannedQuery(
        branch=BranchType.IMAGE_BRANCH,
        normalized_query="xu ly sao",
        query_type=QueryType.CARE_QUERY,
        intent=QueryIntent.CARE_GUIDANCE,
        query_terms=("xu ly",),
        entity_hints=(),
        subqueries=("xu ly sao",),
    )

    request = build_retrieval_request(
        planned,
        disease_hints=("ri sat", "ri sat"),
    )

    assert request.topic_hints == ("xu ly", "ri sat", "care")
    assert request.filters == (request.filters[0],)
    assert request.filters[0].field == "disease_name"
    assert request.filters[0].value == "ri sat"


def test_retrieval_package_exports_public_api():
    import src.rag.retrieval as retrieval
    from src.rag.retrieval.request_builder import build_retrieval_request
    from src.rag.retrieval.types import RetrievalFilter, RetrievalRequest

    assert retrieval.RetrievalFilter is RetrievalFilter
    assert retrieval.RetrievalRequest is RetrievalRequest
    assert retrieval.build_retrieval_request is build_retrieval_request
    assert retrieval.__all__ == [
        "RetrievalFilter",
        "RetrievalRequest",
        "build_retrieval_request",
    ]
