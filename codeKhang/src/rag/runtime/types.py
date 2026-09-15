from dataclasses import dataclass

from src.rag.integration.types import PreparedBranchInput
from src.rag.planning.types import PlannedQuery
from src.rag.retrieval.hybrid_types import RetrievalBundle
from src.rag.retrieval.types import RetrievalRequest


@dataclass(frozen=True)
class TextRuntimeResult:
    prepared: PreparedBranchInput
    planned: PlannedQuery
    retrieval_request: RetrievalRequest
    retrieval_bundle: RetrievalBundle | None = None


@dataclass(frozen=True)
class ImageRuntimeResult:
    prepared: PreparedBranchInput
    planned: PlannedQuery
    retrieval_request: RetrievalRequest
    disease_hints: tuple[str, ...]
