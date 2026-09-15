from src.rag.answering.prompt import build_citation_label, render_context_block
from src.rag.answering.types import AnswerContext, GroundedAnswer, MaterializedChunk, TextAnswerResult


class TextAnsweringService:
    def __init__(self, *, reranker, chunk_resolver, answer_generator, top_k: int = 4):
        self.reranker = reranker
        self.chunk_resolver = chunk_resolver
        self.answer_generator = answer_generator
        self.top_k = top_k

    def answer(self, runtime_result):
        retrieval_bundle = runtime_result.retrieval_bundle
        if retrieval_bundle is None:
            raise ValueError(
                "Text runtime result requires retrieval_bundle before answer synthesis"
            )

        ranked_ids = self.reranker.rank(retrieval_bundle)
        selected_ids = tuple(ranked_ids[: self.top_k])
        records = self.chunk_resolver.resolve(selected_ids)
        candidate_map = {
            candidate.chunk_id: candidate
            for candidate in retrieval_bundle.merged_candidates
        }
        record_map = {
            record.chunk_id: record
            for record in records
            if record.chunk_id in selected_ids
        }

        selected_chunks = []
        for chunk_id in selected_ids:
            record = record_map.get(chunk_id)
            if record is None:
                continue
            candidate = candidate_map.get(record.chunk_id)
            if candidate is None:
                continue
            selected_chunks.append(
                MaterializedChunk(
                    chunk_id=record.chunk_id,
                    content=record.content,
                    metadata=record.metadata,
                    sources=candidate.sources,
                    raw_scores=candidate.raw_scores,
                    graph_match_reasons=candidate.graph_match_reasons,
                )
            )

        if not selected_chunks:
            raise ValueError("Answer synthesis requires at least one materialized chunk")

        answer_context = AnswerContext(
            retrieval_request=retrieval_bundle.retrieval_request,
            selected_chunks=tuple(selected_chunks),
            rendered_context=render_context_block(tuple(selected_chunks)),
        )
        answer_text = self.answer_generator.generate(answer_context)
        if not isinstance(answer_text, str):
            raise ValueError("Answer generator must return answer_text as str.")
        citations = tuple(build_citation_label(chunk) for chunk in selected_chunks)
        grounded_answer = GroundedAnswer(
            answer_text=answer_text,
            citations=citations,
            used_chunk_ids=tuple(chunk.chunk_id for chunk in selected_chunks),
        )
        return TextAnswerResult(
            runtime_result=runtime_result,
            answer_context=answer_context,
            grounded_answer=grounded_answer,
        )
