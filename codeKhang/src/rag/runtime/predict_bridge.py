from src.rag.runtime.types import ImageRuntimeResult


def extract_disease_hints_from_detection(
    detections: list[dict],
    class_names: list[str],
    max_hints: int = 3,
) -> tuple[str, ...]:
    ranked: list[str] = []
    seen = set()

    for item in detections:
        try:
            class_id = int(item["class_id"])
        except (KeyError, TypeError, ValueError):
            continue

        if not 0 <= class_id < len(class_names):
            continue

        label = class_names[class_id]

        cleaned = " ".join(label.strip().split())
        if not cleaned or cleaned in seen:
            continue

        seen.add(cleaned)
        ranked.append(cleaned)

        if len(ranked) >= max_hints:
            break

    return tuple(ranked)


def build_image_runtime_payload(result: ImageRuntimeResult) -> dict:
    return {
        "branch": result.prepared.branch.value,
        "normalized_query": result.prepared.normalization.normalized_query,
        "query_type": result.prepared.normalization.query_type.value,
        "intent": result.planned.intent.value,
        "query_terms": result.retrieval_request.query_terms,
        "topic_hints": result.retrieval_request.topic_hints,
        "subqueries": result.retrieval_request.subqueries,
        "disease_hints": result.disease_hints,
        "filters": tuple(
            (item.field, item.value) for item in result.retrieval_request.filters
        ),
    }
