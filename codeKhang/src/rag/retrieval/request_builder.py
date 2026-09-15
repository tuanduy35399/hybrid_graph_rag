from src.rag.integration.types import BranchType
from src.rag.planning.types import EntityHint, PlannedQuery, QueryIntent
from src.rag.retrieval.types import RetrievalFilter, RetrievalRequest


def build_retrieval_request(
    planned: PlannedQuery,
    disease_hints: tuple[str, ...] = (),
) -> RetrievalRequest:
    normalized_disease_hints = _normalize_values(disease_hints)
    entity_hints = _merge_entity_hints(
        planned.entity_hints,
        normalized_disease_hints,
    )
    topic_hints = _derive_topic_hints(
        planned,
        planned.intent,
        normalized_disease_hints,
    )
    filters = _build_filters(planned.branch, normalized_disease_hints)

    return RetrievalRequest(
        query_text=planned.normalized_query,
        branch=planned.branch,
        query_type=planned.query_type,
        intent=planned.intent,
        query_terms=planned.query_terms,
        entity_hints=entity_hints,
        topic_hints=topic_hints,
        filters=filters,
        subqueries=planned.subqueries,
    )


def _normalize_values(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = []
    seen = set()

    for value in values:
        cleaned = " ".join(value.strip().split())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)

    return tuple(normalized)


def _merge_entity_hints(
    entity_hints: tuple[EntityHint, ...],
    disease_hints: tuple[str, ...],
) -> tuple[EntityHint, ...]:
    merged = list(entity_hints)
    seen = {(hint.kind, hint.value) for hint in merged}

    for disease_name in disease_hints:
        key = ("disease_name", disease_name)
        if key in seen:
            continue
        seen.add(key)
        merged.append(EntityHint(kind="disease_name", value=disease_name))

    return tuple(merged)


def _derive_topic_hints(
    planned: PlannedQuery,
    intent: QueryIntent,
    disease_hints: tuple[str, ...],
) -> tuple[str, ...]:
    topic_hints: list[str] = []
    seen = set()

    def add_topic(value: str) -> None:
        cleaned = " ".join(value.strip().split())
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        topic_hints.append(cleaned)

    for term in planned.query_terms:
        add_topic(term)

    for hint in planned.entity_hints:
        if hint.kind in {"symptom", "plant_part", "care_action"}:
            add_topic(hint.value)

    if planned.branch is BranchType.IMAGE_BRANCH:
        for disease_name in disease_hints:
            add_topic(disease_name)

    if intent is QueryIntent.DISEASE_LOOKUP:
        add_topic("symptoms")
        add_topic("treatment")
    elif intent is QueryIntent.CARE_GUIDANCE:
        add_topic("care")
    elif intent is QueryIntent.MIXED_REASONING:
        add_topic("symptoms")
        add_topic("care")

    return tuple(topic_hints)


def _build_filters(
    branch: BranchType,
    disease_hints: tuple[str, ...],
) -> tuple[RetrievalFilter, ...]:
    if branch is not BranchType.IMAGE_BRANCH or not disease_hints:
        return ()
    return tuple(
        RetrievalFilter(field="disease_name", value=disease_name)
        for disease_name in disease_hints
    )
