from dataclasses import dataclass
from enum import Enum


class QueryType(str, Enum):
    DISEASE_QUERY = "disease_query"
    CARE_QUERY = "care_query"
    MIXED_QUERY = "mixed_query"


@dataclass(frozen=True)
class NormalizationResult:
    normalized_query: str
    query_type: QueryType
    normalization_notes: tuple[str, ...]


def fallback_normalization(raw_text: str) -> NormalizationResult:
    normalized = " ".join(raw_text.strip().split())
    return NormalizationResult(
        normalized_query=normalized,
        query_type=QueryType.CARE_QUERY,
        normalization_notes=("fallback_raw_query",),
    )
