import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TextAnsweringServiceTests(unittest.TestCase):
    def test_service_turns_runtime_bundle_into_grounded_answer(self):
        from src.rag.answering.service import TextAnsweringService
        from src.rag.indexing.models import ChunkMetadata, ChunkRecord
        from src.rag.retrieval.hybrid_types import MergedCandidate, RetrievalBundle
        from src.rag.runtime.types import TextRuntimeResult

        retrieval_bundle = RetrievalBundle(
            retrieval_request=object(),
            graph_candidates=(),
            hybrid_candidates=(),
            merged_candidates=(
                MergedCandidate(chunk_id="chunk-2", sources=("graph",), raw_scores={"hybrid": 0.55}),
                MergedCandidate(chunk_id="chunk-1", sources=("dense", "sparse"), raw_scores={"hybrid": 0.91}),
            ),
        )

        runtime_result = TextRuntimeResult(
            prepared=object(),
            planned=object(),
            retrieval_request=object(),
            retrieval_bundle=retrieval_bundle,
        )

        class StubReranker:
            def rank(self, bundle):
                return ("chunk-1", "chunk-2")

        class StubResolver:
            def resolve(self, chunk_ids):
                records = {
                    "chunk-1": ChunkRecord(
                        chunk_id="chunk-1",
                        content="Noi dung 1",
                        metadata=ChunkMetadata(
                            source_file="kb.md",
                            source_path="data/knowledge_base/kb.md",
                            topic="care",
                        ),
                    ),
                    "chunk-2": ChunkRecord(
                        chunk_id="chunk-2",
                        content="Noi dung 2",
                        metadata=ChunkMetadata(
                            source_file="kb.md",
                            source_path="data/knowledge_base/kb.md",
                            topic="disease",
                        ),
                    ),
                }
                return tuple(records[item] for item in chunk_ids if item in records)

        class StubGenerator:
            def __init__(self):
                self.calls = []

            def generate(self, answer_context):
                self.calls.append(answer_context)
                return "Tra loi co dan chung [kb.md#chunk-1]"

        generator = StubGenerator()
        service = TextAnsweringService(
            reranker=StubReranker(),
            chunk_resolver=StubResolver(),
            answer_generator=generator,
            top_k=1,
        )

        result = service.answer(runtime_result)

        self.assertEqual(result.grounded_answer.used_chunk_ids, ("chunk-1",))
        self.assertEqual(result.grounded_answer.citations, ("kb.md#chunk-1",))
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(
            tuple(chunk.chunk_id for chunk in generator.calls[0].selected_chunks),
            ("chunk-1",),
        )

    def test_service_continues_when_at_least_one_usable_chunk_remains(self):
        from src.rag.answering.service import TextAnsweringService
        from src.rag.indexing.models import ChunkMetadata, ChunkRecord
        from src.rag.retrieval.hybrid_types import MergedCandidate, RetrievalBundle
        from src.rag.runtime.types import TextRuntimeResult

        retrieval_bundle = RetrievalBundle(
            retrieval_request=object(),
            graph_candidates=(),
            hybrid_candidates=(),
            merged_candidates=(
                MergedCandidate(chunk_id="chunk-1", sources=("dense",), raw_scores={"hybrid": 0.91}),
                MergedCandidate(chunk_id="chunk-2", sources=("graph",), raw_scores={"hybrid": 0.55}),
            ),
        )
        runtime_result = TextRuntimeResult(
            prepared=object(),
            planned=object(),
            retrieval_request=object(),
            retrieval_bundle=retrieval_bundle,
        )

        class StubReranker:
            def rank(self, bundle):
                return ("chunk-1", "chunk-2")

        class StubResolver:
            def resolve(self, chunk_ids):
                self.chunk_ids = chunk_ids
                return (
                    ChunkRecord(
                        chunk_id="chunk-2",
                        content="Noi dung 2",
                        metadata=ChunkMetadata(
                            source_file="kb.md",
                            source_path="data/knowledge_base/kb.md",
                            topic="disease",
                        ),
                    ),
                )

        class StubGenerator:
            def __init__(self):
                self.calls = []

            def generate(self, answer_context):
                self.calls.append(answer_context)
                return "Tra loi co dan chung [kb.md#chunk-2]"

        resolver = StubResolver()
        generator = StubGenerator()
        service = TextAnsweringService(
            reranker=StubReranker(),
            chunk_resolver=resolver,
            answer_generator=generator,
            top_k=2,
        )

        result = service.answer(runtime_result)

        self.assertEqual(resolver.chunk_ids, ("chunk-1", "chunk-2"))
        self.assertEqual(result.grounded_answer.used_chunk_ids, ("chunk-2",))
        self.assertEqual(result.grounded_answer.citations, ("kb.md#chunk-2",))
        self.assertEqual(
            tuple(chunk.chunk_id for chunk in result.answer_context.selected_chunks),
            ("chunk-2",),
        )
        self.assertEqual(len(generator.calls), 1)

    def test_service_keeps_only_top_k_selected_ids_in_reranker_order_even_if_resolver_reorders_and_over_returns(self):
        from src.rag.answering.service import TextAnsweringService
        from src.rag.indexing.models import ChunkMetadata, ChunkRecord
        from src.rag.retrieval.hybrid_types import MergedCandidate, RetrievalBundle
        from src.rag.runtime.types import TextRuntimeResult

        retrieval_bundle = RetrievalBundle(
            retrieval_request=object(),
            graph_candidates=(),
            hybrid_candidates=(),
            merged_candidates=(
                MergedCandidate(chunk_id="chunk-1", sources=("dense",), raw_scores={"hybrid": 0.81}),
                MergedCandidate(chunk_id="chunk-2", sources=("graph",), raw_scores={"hybrid": 0.91}),
                MergedCandidate(chunk_id="chunk-3", sources=("dense", "sparse"), raw_scores={"hybrid": 0.55}),
            ),
        )
        runtime_result = TextRuntimeResult(
            prepared=object(),
            planned=object(),
            retrieval_request=object(),
            retrieval_bundle=retrieval_bundle,
        )

        class StubReranker:
            def rank(self, bundle):
                return ("chunk-2", "chunk-1", "chunk-3")

        class StubResolver:
            def resolve(self, chunk_ids):
                self.chunk_ids = chunk_ids
                return (
                    ChunkRecord(
                        chunk_id="chunk-3",
                        content="Noi dung 3",
                        metadata=ChunkMetadata(
                            source_file="kb.md",
                            source_path="data/knowledge_base/kb.md",
                            topic="disease",
                        ),
                    ),
                    ChunkRecord(
                        chunk_id="chunk-1",
                        content="Noi dung 1",
                        metadata=ChunkMetadata(
                            source_file="kb.md",
                            source_path="data/knowledge_base/kb.md",
                            topic="care",
                        ),
                    ),
                    ChunkRecord(
                        chunk_id="chunk-2",
                        content="Noi dung 2",
                        metadata=ChunkMetadata(
                            source_file="kb.md",
                            source_path="data/knowledge_base/kb.md",
                            topic="disease",
                        ),
                    ),
                )

        class StubGenerator:
            def __init__(self):
                self.calls = []

            def generate(self, answer_context):
                self.calls.append(answer_context)
                return "Tra loi co dan chung [kb.md#chunk-2] [kb.md#chunk-1]"

        resolver = StubResolver()
        generator = StubGenerator()
        service = TextAnsweringService(
            reranker=StubReranker(),
            chunk_resolver=resolver,
            answer_generator=generator,
            top_k=2,
        )

        result = service.answer(runtime_result)

        self.assertEqual(resolver.chunk_ids, ("chunk-2", "chunk-1"))
        self.assertEqual(
            tuple(chunk.chunk_id for chunk in result.answer_context.selected_chunks),
            ("chunk-2", "chunk-1"),
        )
        self.assertEqual(result.grounded_answer.used_chunk_ids, ("chunk-2", "chunk-1"))
        self.assertEqual(
            result.grounded_answer.citations,
            ("kb.md#chunk-2", "kb.md#chunk-1"),
        )
        self.assertEqual(
            tuple(chunk.chunk_id for chunk in generator.calls[0].selected_chunks),
            ("chunk-2", "chunk-1"),
        )

    def test_service_raises_when_no_usable_chunks_remain_after_materialization_filtering(self):
        from src.rag.answering.service import TextAnsweringService
        from src.rag.indexing.models import ChunkMetadata, ChunkRecord
        from src.rag.retrieval.hybrid_types import MergedCandidate, RetrievalBundle
        from src.rag.runtime.types import TextRuntimeResult

        retrieval_bundle = RetrievalBundle(
            retrieval_request=object(),
            graph_candidates=(),
            hybrid_candidates=(),
            merged_candidates=(
                MergedCandidate(chunk_id="chunk-1", sources=("dense",), raw_scores={"hybrid": 0.91}),
            ),
        )
        runtime_result = TextRuntimeResult(
            prepared=object(),
            planned=object(),
            retrieval_request=object(),
            retrieval_bundle=retrieval_bundle,
        )

        class StubReranker:
            def rank(self, bundle):
                return ("chunk-1",)

        class StubResolver:
            def resolve(self, chunk_ids):
                return (
                    ChunkRecord(
                        chunk_id="chunk-missing-from-candidates",
                        content="Noi dung khong hop le",
                        metadata=ChunkMetadata(
                            source_file="kb.md",
                            source_path="data/knowledge_base/kb.md",
                            topic="disease",
                        ),
                    ),
                )

        service = TextAnsweringService(
            reranker=StubReranker(),
            chunk_resolver=StubResolver(),
            answer_generator=object(),
        )

        with self.assertRaisesRegex(ValueError, "Answer synthesis requires at least one materialized chunk"):
            service.answer(runtime_result)

    def test_service_raises_when_answer_generator_returns_non_string(self):
        from src.rag.answering.service import TextAnsweringService
        from src.rag.indexing.models import ChunkMetadata, ChunkRecord
        from src.rag.retrieval.hybrid_types import MergedCandidate, RetrievalBundle
        from src.rag.runtime.types import TextRuntimeResult

        retrieval_bundle = RetrievalBundle(
            retrieval_request=object(),
            graph_candidates=(),
            hybrid_candidates=(),
            merged_candidates=(
                MergedCandidate(chunk_id="chunk-1", sources=("dense",), raw_scores={"hybrid": 0.91}),
            ),
        )
        runtime_result = TextRuntimeResult(
            prepared=object(),
            planned=object(),
            retrieval_request=object(),
            retrieval_bundle=retrieval_bundle,
        )

        class StubReranker:
            def rank(self, bundle):
                return ("chunk-1",)

        class StubResolver:
            def resolve(self, chunk_ids):
                return (
                    ChunkRecord(
                        chunk_id="chunk-1",
                        content="Noi dung 1",
                        metadata=ChunkMetadata(
                            source_file="kb.md",
                            source_path="data/knowledge_base/kb.md",
                            topic="care",
                        ),
                    ),
                )

        class StubGenerator:
            def generate(self, answer_context):
                return None

        service = TextAnsweringService(
            reranker=StubReranker(),
            chunk_resolver=StubResolver(),
            answer_generator=StubGenerator(),
        )

        with self.assertRaisesRegex(ValueError, "Answer generator must return answer_text as str."):
            service.answer(runtime_result)

    def test_service_raises_when_runtime_result_has_no_retrieval_bundle(self):
        from src.rag.answering.service import TextAnsweringService
        from src.rag.runtime.types import TextRuntimeResult

        runtime_result = TextRuntimeResult(
            prepared=object(),
            planned=object(),
            retrieval_request=object(),
            retrieval_bundle=None,
        )

        service = TextAnsweringService(
            reranker=object(),
            chunk_resolver=object(),
            answer_generator=object(),
        )

        with self.assertRaisesRegex(ValueError, "Text runtime result requires retrieval_bundle before answer synthesis"):
            service.answer(runtime_result)
