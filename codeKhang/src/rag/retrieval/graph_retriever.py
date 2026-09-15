from src.rag.retrieval.hybrid_types import GraphCandidate
from src.rag.retrieval.types import RetrievalRequest


class GraphRetriever:
    def retrieve(self, request: RetrievalRequest, chunks) -> tuple[GraphCandidate, ...]:
        query_terms = _build_match_values(request.query_terms)
        entity_values = _build_entity_match_values(request)
        candidates = []

        for chunk in chunks:
            reasons = []
            disease_name = (chunk.metadata.disease_name or "").strip()
            if disease_name and _normalize_for_match(disease_name) in entity_values:
                reasons.append(f"metadata:disease_name={disease_name}")

            for relation in chunk.metadata.semantic_graph:
                if _normalize_for_match(relation.source) in query_terms:
                    reasons.append(f"relation:source={relation.source.strip()}")
                if _normalize_for_match(relation.target) in query_terms.union(entity_values):
                    reasons.append(f"relation:target={relation.target.strip()}")

            if reasons:
                candidates.append(
                    GraphCandidate(
                        chunk_id=chunk.chunk_id,
                        match_reasons=_dedupe(reasons),
                        metadata={
                            "source_file": chunk.metadata.source_file,
                            "source_path": chunk.metadata.source_path,
                            "doc_type": chunk.metadata.doc_type,
                            "topic": chunk.metadata.topic,
                        },
                    )
                )

        return tuple(candidates)


def _build_match_values(values) -> set[str]:
    return {
        normalized
        for value in values
        for normalized in (_normalize_for_match(value),)
        if normalized
    }


def _build_entity_match_values(request: RetrievalRequest) -> set[str]:
    return _build_match_values(
        hint.value
        for hint in request.entity_hints
        if _normalize_for_match(hint.kind) in {"disease name"} and hint.value.strip()
    )


def _normalize_for_match(value: str) -> str:
    return " ".join(value.strip().replace("_", " ").split()).lower()


def _dedupe(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
