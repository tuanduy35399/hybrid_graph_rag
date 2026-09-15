from dataclasses import dataclass

from src.rag.integration.types import BranchType
from src.rag.normalization.types import QueryType
from src.rag.planning.types import EntityHint, QueryIntent


@dataclass(frozen=True)
class RetrievalFilter:
    field: str
    value: str


@dataclass(frozen=True)
class RetrievalRequest:
    query_text: str
    branch: BranchType
    query_type: QueryType
    intent: QueryIntent
    query_terms: tuple[str, ...]
    entity_hints: tuple[EntityHint, ...]
    topic_hints: tuple[str, ...]
    filters: tuple[RetrievalFilter, ...]
    subqueries: tuple[str, ...]
