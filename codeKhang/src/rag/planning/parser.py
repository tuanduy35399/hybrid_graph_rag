import json

from src.rag.integration.types import BranchType
from src.rag.normalization.types import QueryType
from src.rag.planning.types import EntityHint, PlannedQuery, QueryIntent


def parse_planner_response(
    branch: BranchType,
    normalized_query: str,
    query_type: QueryType,
    llm_text: str,
) -> PlannedQuery:
    try:
        payload = json.loads(llm_text)
    except Exception:
        return fallback_planned_query(branch, normalized_query, query_type)

    if not isinstance(payload, dict):
        return fallback_planned_query(branch, normalized_query, query_type)

    intent_value = payload.get("intent")
    query_terms = payload.get("query_terms")
    entity_hints = payload.get("entity_hints")
    subqueries = payload.get("subqueries")

    if not isinstance(intent_value, str):
        return fallback_planned_query(branch, normalized_query, query_type)
    if not isinstance(query_terms, list) or not all(
        isinstance(item, str) for item in query_terms
    ):
        return fallback_planned_query(branch, normalized_query, query_type)
    if not isinstance(entity_hints, list):
        return fallback_planned_query(branch, normalized_query, query_type)
    if not isinstance(subqueries, list) or not all(
        isinstance(item, str) for item in subqueries
    ):
        return fallback_planned_query(branch, normalized_query, query_type)

    try:
        intent = QueryIntent(intent_value)
    except ValueError:
        return fallback_planned_query(branch, normalized_query, query_type)
    if not _is_intent_compatible(query_type, intent):
        return fallback_planned_query(branch, normalized_query, query_type)

    parsed_hints = []
    seen_hints = set()
    for item in entity_hints:
        if not isinstance(item, dict):
            return fallback_planned_query(branch, normalized_query, query_type)
        kind = item.get("kind")
        value = item.get("value")
        if not isinstance(kind, str) or not isinstance(value, str):
            return fallback_planned_query(branch, normalized_query, query_type)
        cleaned_kind = kind.strip()
        cleaned_value = " ".join(value.strip().split())
        if not cleaned_kind or not cleaned_value:
            continue
        key = (cleaned_kind, cleaned_value)
        if key in seen_hints:
            continue
        seen_hints.add(key)
        parsed_hints.append(EntityHint(kind=cleaned_kind, value=cleaned_value))

    cleaned_terms = tuple(
        term
        for term in (" ".join(item.strip().split()) for item in query_terms)
        if term
    )
    cleaned_subqueries = tuple(
        subquery
        for subquery in (" ".join(item.strip().split()) for item in subqueries)
        if subquery
    )

    if not cleaned_subqueries:
        return fallback_planned_query(branch, normalized_query, query_type)

    return PlannedQuery(
        branch=branch,
        normalized_query=normalized_query,
        query_type=query_type,
        intent=intent,
        query_terms=cleaned_terms,
        entity_hints=tuple(parsed_hints),
        subqueries=cleaned_subqueries,
    )


def fallback_planned_query(
    branch: BranchType,
    normalized_query: str,
    query_type: QueryType,
) -> PlannedQuery:
    intent = _expected_intent_for_query_type(query_type)

    return PlannedQuery(
        branch=branch,
        normalized_query=normalized_query,
        query_type=query_type,
        intent=intent,
        query_terms=(),
        entity_hints=(),
        subqueries=(normalized_query,),
    )


def _is_intent_compatible(
    query_type: QueryType,
    intent: QueryIntent,
) -> bool:
    return intent is _expected_intent_for_query_type(query_type)


def _expected_intent_for_query_type(query_type: QueryType) -> QueryIntent:
    if query_type is QueryType.DISEASE_QUERY:
        return QueryIntent.DISEASE_LOOKUP
    if query_type is QueryType.CARE_QUERY:
        return QueryIntent.CARE_GUIDANCE
    if query_type is QueryType.MIXED_QUERY:
        return QueryIntent.MIXED_REASONING
    return QueryIntent.GENERAL_LOOKUP
