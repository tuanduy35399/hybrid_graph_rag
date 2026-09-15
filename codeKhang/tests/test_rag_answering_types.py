import os
import sys
from typing import get_type_hints
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class AnsweringTypeTests(unittest.TestCase):
    def _build_retrieval_request(self):
        from src.rag.integration.types import BranchType
        from src.rag.normalization.types import QueryType
        from src.rag.planning.types import EntityHint, QueryIntent
        from src.rag.retrieval.types import RetrievalFilter, RetrievalRequest

        return RetrievalRequest(
            query_text="La mai bi dom vang",
            branch=BranchType.TEXT_ONLY_BRANCH,
            query_type=QueryType.DISEASE_QUERY,
            intent=QueryIntent.DISEASE_LOOKUP,
            query_terms=("la mai", "dom vang"),
            entity_hints=(EntityHint(kind="disease", value="dom vang"),),
            topic_hints=("disease",),
            filters=(RetrievalFilter(field="doc_type", value="faq"),),
            subqueries=("Trieu chung dom vang o la mai",),
        )

    def _build_text_runtime_result(self):
        from src.rag.integration.types import BranchType, PreparedBranchInput
        from src.rag.normalization.types import NormalizationResult, QueryType
        from src.rag.planning.types import EntityHint, PlannedQuery, QueryIntent
        from src.rag.runtime.types import TextRuntimeResult

        retrieval_request = self._build_retrieval_request()
        normalization = NormalizationResult(
            normalized_query="La mai bi dom vang",
            query_type=QueryType.DISEASE_QUERY,
            normalization_notes=("normalized",),
        )
        prepared = PreparedBranchInput(
            branch=BranchType.TEXT_ONLY_BRANCH,
            normalization=normalization,
        )
        planned = PlannedQuery(
            branch=BranchType.TEXT_ONLY_BRANCH,
            normalized_query=normalization.normalized_query,
            query_type=QueryType.DISEASE_QUERY,
            intent=QueryIntent.DISEASE_LOOKUP,
            query_terms=("la mai", "dom vang"),
            entity_hints=(EntityHint(kind="disease", value="dom vang"),),
            subqueries=("Trieu chung dom vang o la mai",),
        )
        return TextRuntimeResult(
            prepared=prepared,
            planned=planned,
            retrieval_request=retrieval_request,
        )

    def test_grounded_answer_contract_keeps_citations_and_chunk_ids(self):
        from src.rag.answering.types import GroundedAnswer

        grounded = GroundedAnswer(
            answer_text="La mai dang bi dom vang.",
            citations=("kb.md#chunk-1", "kb.md#chunk-2"),
            used_chunk_ids=("chunk-1", "chunk-2"),
        )

        self.assertEqual(grounded.answer_text, "La mai dang bi dom vang.")
        self.assertEqual(
            grounded.citations,
            ("kb.md#chunk-1", "kb.md#chunk-2"),
        )
        self.assertEqual(grounded.used_chunk_ids, ("chunk-1", "chunk-2"))

    def test_text_answer_result_keeps_runtime_result_context_and_answer(self):
        from src.rag.answering.types import AnswerContext, GroundedAnswer, MaterializedChunk, TextAnswerResult
        from src.rag.indexing.models import ChunkMetadata, ChunkRecord

        retrieval_request = self._build_retrieval_request()
        record = ChunkRecord(
            chunk_id="chunk-1",
            content="Noi dung chunk 1",
            metadata=ChunkMetadata(
                source_file="kb.md",
                source_path="data/knowledge_base/kb.md",
                heading_path=("Benh",),
                doc_type="faq",
                topic="disease",
            ),
        )
        materialized = MaterializedChunk(
            chunk_id=record.chunk_id,
            content=record.content,
            metadata=record.metadata,
            sources=("graph",),
            raw_scores={"hybrid": 0.91},
            graph_match_reasons=("relation:target=dom vang",),
        )
        context = AnswerContext(
            retrieval_request=retrieval_request,
            selected_chunks=(materialized,),
            rendered_context="[kb.md#chunk-1]\nNoi dung chunk 1",
        )
        grounded = GroundedAnswer(
            answer_text="Tra loi grounded",
            citations=("kb.md#chunk-1",),
            used_chunk_ids=("chunk-1",),
        )
        runtime_result = self._build_text_runtime_result()

        result = TextAnswerResult(
            runtime_result=runtime_result,
            answer_context=context,
            grounded_answer=grounded,
        )

        self.assertIs(result.runtime_result, runtime_result)
        self.assertIs(result.answer_context, context)
        self.assertIs(result.grounded_answer, grounded)
        self.assertEqual(result.answer_context.selected_chunks[0].chunk_id, "chunk-1")
        self.assertEqual(result.grounded_answer.answer_text, "Tra loi grounded")
        self.assertEqual(result.grounded_answer.citations, ("kb.md#chunk-1",))

    def test_types_freeze_mutable_inputs_and_expose_concrete_repo_seams(self):
        from src.rag.answering.types import AnswerContext, GroundedAnswer, MaterializedChunk, TextAnswerResult
        from src.rag.indexing.models import ChunkMetadata
        from src.rag.retrieval.types import RetrievalRequest
        from src.rag.runtime.types import TextRuntimeResult

        sources = ["graph", "dense"]
        graph_match_reasons = ["relation:target=dom vang"]
        raw_scores = {"hybrid": 0.91}
        citations = ["kb.md#chunk-1"]
        used_chunk_ids = ["chunk-1"]
        retrieval_request = self._build_retrieval_request()
        runtime_result = self._build_text_runtime_result()

        materialized = MaterializedChunk(
            chunk_id="chunk-1",
            content="Noi dung chunk 1",
            metadata=ChunkMetadata(
                source_file="kb.md",
                source_path="data/knowledge_base/kb.md",
                heading_path=("Benh",),
                doc_type="faq",
                topic="disease",
            ),
            sources=sources,
            raw_scores=raw_scores,
            graph_match_reasons=graph_match_reasons,
        )
        grounded = GroundedAnswer(
            answer_text="Tra loi grounded",
            citations=citations,
            used_chunk_ids=used_chunk_ids,
        )
        context = AnswerContext(
            retrieval_request=retrieval_request,
            selected_chunks=[materialized],
            rendered_context="[kb.md#chunk-1]\nNoi dung chunk 1",
        )
        result = TextAnswerResult(
            runtime_result=runtime_result,
            answer_context=context,
            grounded_answer=grounded,
        )

        sources.append("sparse")
        graph_match_reasons.append("relation:source=la mai")
        raw_scores["hybrid"] = 0.1
        raw_scores["dense"] = 0.8
        citations.append("kb.md#chunk-2")
        used_chunk_ids.append("chunk-2")

        self.assertEqual(materialized.sources, ("graph", "dense"))
        self.assertEqual(materialized.graph_match_reasons, ("relation:target=dom vang",))
        self.assertEqual(dict(materialized.raw_scores), {"hybrid": 0.91})
        self.assertEqual(grounded.citations, ("kb.md#chunk-1",))
        self.assertEqual(grounded.used_chunk_ids, ("chunk-1",))
        self.assertEqual(tuple(context.selected_chunks), (materialized,))
        self.assertIsInstance(context.retrieval_request, RetrievalRequest)
        self.assertIsInstance(result.runtime_result, TextRuntimeResult)

        type_hints = get_type_hints(AnswerContext)
        self.assertIs(type_hints["retrieval_request"], RetrievalRequest)

        result_hints = get_type_hints(TextAnswerResult)
        self.assertIs(result_hints["runtime_result"], TextRuntimeResult)
