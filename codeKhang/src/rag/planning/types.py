from dataclasses import dataclass
from enum import Enum

from src.rag.integration.types import BranchType
from src.rag.normalization.types import QueryType


class QueryIntent(str, Enum):
    DISEASE_LOOKUP = "disease_lookup"
    CARE_GUIDANCE = "care_guidance"
    MIXED_REASONING = "mixed_reasoning"
    GENERAL_LOOKUP = "general_lookup"


@dataclass(frozen=True)
class EntityHint:
    kind: str
    value: str


@dataclass(frozen=True)
class PlannedQuery:
    branch: BranchType
    normalized_query: str
    query_type: QueryType
    intent: QueryIntent
    query_terms: tuple[str, ...]
    entity_hints: tuple[EntityHint, ...]
    subqueries: tuple[str, ...]
