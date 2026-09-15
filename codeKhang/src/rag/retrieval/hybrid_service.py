from src.rag.retrieval.hybrid_types import MergedCandidate, RetrievalBundle


def _dedupe_preserve_order(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


class HybridRetrievalService:
    def __init__(self, *, graph_retriever, hybrid_retriever):
        self.graph_retriever = graph_retriever
        self.hybrid_retriever = hybrid_retriever

    def retrieve(self, *, retrieval_request, graph_chunks):
        graph_candidates = self.graph_retriever.retrieve(
            retrieval_request,
            graph_chunks,
        )
        hybrid_candidates = self.hybrid_retriever.retrieve(retrieval_request)
        merged_candidates_by_chunk_id = {}

        for graph_candidate in graph_candidates:
            merged_candidates_by_chunk_id[graph_candidate.chunk_id] = MergedCandidate(
                chunk_id=graph_candidate.chunk_id,
                sources=("graph",),
                graph_match_reasons=graph_candidate.match_reasons,
                raw_scores={},
            )

        for hybrid_candidate in hybrid_candidates:
            existing = merged_candidates_by_chunk_id.get(hybrid_candidate.chunk_id)
            if existing is None:
                merged_candidates_by_chunk_id[hybrid_candidate.chunk_id] = MergedCandidate(
                    chunk_id=hybrid_candidate.chunk_id,
                    sources=hybrid_candidate.sources,
                    graph_match_reasons=(),
                    raw_scores=hybrid_candidate.raw_scores,
                )
                continue

            merged_candidates_by_chunk_id[hybrid_candidate.chunk_id] = MergedCandidate(
                chunk_id=hybrid_candidate.chunk_id,
                sources=_dedupe_preserve_order(existing.sources + hybrid_candidate.sources),
                graph_match_reasons=existing.graph_match_reasons,
                raw_scores={**existing.raw_scores, **hybrid_candidate.raw_scores},
            )

        return RetrievalBundle(
            retrieval_request=retrieval_request,
            graph_candidates=graph_candidates,
            hybrid_candidates=hybrid_candidates,
            merged_candidates=tuple(merged_candidates_by_chunk_id.values()),
        )
