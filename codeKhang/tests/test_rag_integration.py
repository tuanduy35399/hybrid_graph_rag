import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.normalization.types import (
    NormalizationResult,
    QueryType,
    fallback_normalization,
)


def test_branch_type_supports_image_and_text_only():
    from src.rag.integration.types import BranchType

    assert BranchType.IMAGE_BRANCH.value == "image_branch"
    assert BranchType.TEXT_ONLY_BRANCH.value == "text_only_branch"


def test_prepared_branch_input_wraps_normalization_result():
    from src.rag.integration.types import BranchType, PreparedBranchInput

    result = NormalizationResult(
        normalized_query="la mai bi dom den la gi",
        query_type=QueryType.DISEASE_QUERY,
        normalization_notes=("no_major_change",),
    )

    prepared = PreparedBranchInput(
        branch=BranchType.TEXT_ONLY_BRANCH,
        normalization=result,
    )

    assert prepared.branch is BranchType.TEXT_ONLY_BRANCH
    assert prepared.normalization is result


class StubNormalizer:
    def __init__(self, result: NormalizationResult):
        self.result = result
        self.calls: list[str] = []

    def normalize(self, raw_text: str) -> NormalizationResult:
        self.calls.append(raw_text)
        return self.result


def test_shared_branch_preprocessor_routes_to_image_branch_when_image_exists():
    from src.rag.integration.service import SharedBranchPreprocessor
    from src.rag.integration.types import BranchType

    result = NormalizationResult(
        normalized_query="la mai bi dom den la gi",
        query_type=QueryType.DISEASE_QUERY,
        normalization_notes=("no_major_change",),
    )
    preprocessor = SharedBranchPreprocessor(StubNormalizer(result))

    prepared = preprocessor.prepare("la mai bi dom den la gi", has_image=True)

    assert prepared.branch is BranchType.IMAGE_BRANCH
    assert prepared.normalization == result
    assert preprocessor.normalizer.calls == ["la mai bi dom den la gi"]


def test_shared_branch_preprocessor_routes_to_text_only_branch_without_image():
    from src.rag.integration.service import SharedBranchPreprocessor
    from src.rag.integration.types import BranchType

    result = NormalizationResult(
        normalized_query="cach tuoi mai",
        query_type=QueryType.CARE_QUERY,
        normalization_notes=("no_major_change",),
    )
    preprocessor = SharedBranchPreprocessor(StubNormalizer(result))

    prepared = preprocessor.prepare("cach tuoi mai", has_image=False)

    assert prepared.branch is BranchType.TEXT_ONLY_BRANCH
    assert prepared.normalization == result
    assert preprocessor.normalizer.calls == ["cach tuoi mai"]


def test_shared_branch_preprocessor_preserves_fallback_result_without_mutation():
    from src.rag.integration.service import SharedBranchPreprocessor
    from src.rag.integration.types import BranchType

    result = fallback_normalization("  cach tuoi mai  ")
    preprocessor = SharedBranchPreprocessor(StubNormalizer(result))

    prepared = preprocessor.prepare("  cach tuoi mai  ", has_image=False)

    assert prepared.branch is BranchType.TEXT_ONLY_BRANCH
    assert prepared.normalization is result
    assert prepared.normalization.normalization_notes == ("fallback_raw_query",)


def test_integration_package_exports_public_api():
    import src.rag.integration as integration
    from src.rag.integration.service import SharedBranchPreprocessor
    from src.rag.integration.types import BranchType, PreparedBranchInput

    assert integration.BranchType is BranchType
    assert integration.PreparedBranchInput is PreparedBranchInput
    assert integration.SharedBranchPreprocessor is SharedBranchPreprocessor
    assert integration.__all__ == [
        "BranchType",
        "PreparedBranchInput",
        "SharedBranchPreprocessor",
    ]


def test_shared_branch_preprocessor_uses_image_presence_not_query_type_for_branch():
    from src.rag.integration.service import SharedBranchPreprocessor
    from src.rag.integration.types import BranchType

    result = NormalizationResult(
        normalized_query="la mai bi dom den la benh gi va xu ly sao",
        query_type=QueryType.MIXED_QUERY,
        normalization_notes=("rewritten_for_retrieval",),
    )
    preprocessor = SharedBranchPreprocessor(StubNormalizer(result))

    prepared = preprocessor.prepare(
        "la mai bi dom den la benh gi va xu ly sao",
        has_image=True,
    )

    assert prepared.branch is BranchType.IMAGE_BRANCH
    assert prepared.normalization.query_type is QueryType.MIXED_QUERY
