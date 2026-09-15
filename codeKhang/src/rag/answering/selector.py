from src.rag.retrieval.hybrid_types import RetrievalBundle


def _has_graph_source(candidate) -> bool:
    return "graph" in candidate.sources


def _hybrid_score(candidate) -> float:
    value = candidate.raw_scores.get("hybrid", 0.0)
    return float(value)


class ScorePreservingReranker:
    def rank(self, retrieval_bundle: RetrievalBundle) -> tuple[str, ...]:
        ordered = []
        for index, candidate in enumerate(retrieval_bundle.merged_candidates):
            ordered.append(
                (
                    0 if _has_graph_source(candidate) else 1,
                    -_hybrid_score(candidate),
                    index,
                    candidate.chunk_id,
                )
            )
        ordered.sort()
        return tuple(item[3] for item in ordered)
