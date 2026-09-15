import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class AnsweringSelectionTests(unittest.TestCase):
    def test_score_preserving_reranker_prefers_graph_then_score_then_order(self):
        from src.rag.answering.selector import ScorePreservingReranker
        from src.rag.retrieval.hybrid_types import GraphCandidate, HybridCandidate, MergedCandidate, RetrievalBundle

        bundle = RetrievalBundle(
            retrieval_request=object(),
            graph_candidates=(
                GraphCandidate(
                    chunk_id="chunk-2",
                    match_reasons=("relation:target=ri sat",),
                    metadata={},
                ),
                GraphCandidate(
                    chunk_id="chunk-5",
                    match_reasons=("relation:target=thieu mau",),
                    metadata={},
                ),
            ),
            hybrid_candidates=(
                HybridCandidate(chunk_id="chunk-1", sources=("dense", "sparse"), raw_scores={"hybrid": 0.81}),
                HybridCandidate(chunk_id="chunk-2", sources=("dense", "sparse"), raw_scores={"hybrid": 0.55}),
                HybridCandidate(chunk_id="chunk-3", sources=("dense", "sparse"), raw_scores={"hybrid": 0.72}),
                HybridCandidate(chunk_id="chunk-4", sources=("dense", "sparse"), raw_scores={"hybrid": 0.72}),
                HybridCandidate(chunk_id="chunk-5", sources=("dense", "sparse"), raw_scores={"hybrid": 0.55}),
            ),
            merged_candidates=(
                MergedCandidate(chunk_id="chunk-1", sources=("dense", "sparse"), raw_scores={"hybrid": 0.81}),
                MergedCandidate(chunk_id="chunk-2", sources=("graph", "dense", "sparse"), graph_match_reasons=("relation:target=ri sat",), raw_scores={"hybrid": 0.55}),
                MergedCandidate(chunk_id="chunk-3", sources=("dense", "sparse"), raw_scores={"hybrid": 0.72}),
                MergedCandidate(chunk_id="chunk-4", sources=("dense", "sparse"), raw_scores={"hybrid": 0.72}),
                MergedCandidate(chunk_id="chunk-5", sources=("graph", "dense", "sparse"), graph_match_reasons=("relation:target=thieu mau",), raw_scores={"hybrid": 0.55}),
            ),
        )

        ranked_ids = ScorePreservingReranker().rank(bundle)

        self.assertEqual(ranked_ids, ("chunk-2", "chunk-5", "chunk-1", "chunk-3", "chunk-4"))

    def test_manifest_chunk_resolver_returns_only_existing_chunk_records_in_requested_order(self):
        from src.rag.answering.resolver import build_manifest_chunk_resolver

        manifest_path = Path(tempfile.mkdtemp()) / "manifest.jsonl"
        manifest_path.write_text(
            "\n".join(
                (
                    '{"chunk_id": "chunk-1", "content": "Noi dung 1", "metadata": {"source_file": "kb.md", "source_path": "data/knowledge_base/kb.md"}}',
                    '{"chunk_id": "chunk-2", "content": "Noi dung 2", "metadata": {"source_file": "kb.md", "source_path": "data/knowledge_base/kb.md"}}',
                )
            ),
            encoding="utf-8",
        )

        resolver = build_manifest_chunk_resolver(manifest_path)
        records = resolver.resolve(("chunk-2", "missing", "chunk-1"))

        self.assertEqual(tuple(record.chunk_id for record in records), ("chunk-2", "chunk-1"))
        self.assertEqual(records[0].content, "Noi dung 2")
        self.assertEqual(records[0].metadata.source_path, "data/knowledge_base/kb.md")

    def test_build_manifest_chunk_resolver_rejects_duplicate_chunk_ids(self):
        from src.rag.answering.resolver import build_manifest_chunk_resolver

        manifest_path = Path(tempfile.mkdtemp()) / "manifest.jsonl"
        manifest_path.write_text(
            "\n".join(
                (
                    '{"chunk_id": "chunk-1", "content": "Noi dung 1", "metadata": {"source_file": "kb.md", "source_path": "data/knowledge_base/kb.md"}}',
                    '{"chunk_id": "chunk-1", "content": "Noi dung 1 ban trung", "metadata": {"source_file": "kb.md", "source_path": "data/knowledge_base/kb.md"}}',
                )
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "Duplicate chunk_id 'chunk-1' found in manifest records."):
            build_manifest_chunk_resolver(manifest_path)
