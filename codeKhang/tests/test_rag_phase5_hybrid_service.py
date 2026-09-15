import os
import sys
import unittest
from types import MappingProxyType

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.integration.types import BranchType
from src.rag.normalization.types import QueryType
from src.rag.planning.types import QueryIntent
from src.rag.retrieval.types import RetrievalRequest


def _make_request():
    return RetrievalRequest(
        query_text="la mai bi dom den",
        branch=BranchType.TEXT_ONLY_BRANCH,
        query_type=QueryType.DISEASE_QUERY,
        intent=QueryIntent.DISEASE_LOOKUP,
        query_terms=("mai", "dom den"),
        entity_hints=(),
        topic_hints=("symptoms",),
        filters=(),
        subqueries=("la mai bi dom den",),
    )


class HybridRetrievalBundleContractTests(unittest.TestCase):
    def test_hybrid_retrieval_bundle_contract_smoke(self):
        import src.rag.retrieval as retrieval
        from src.rag.retrieval.hybrid_types import (
            GraphCandidate,
            HybridCandidate,
            MergedCandidate,
            RetrievalBundle,
        )

        request = RetrievalRequest(
            query_text="la mai bi dom den",
            branch=BranchType.TEXT_ONLY_BRANCH,
            query_type=QueryType.DISEASE_QUERY,
            intent=QueryIntent.DISEASE_LOOKUP,
            query_terms=("mai", "dom den"),
            entity_hints=(),
            topic_hints=("symptoms",),
            filters=(),
            subqueries=("la mai bi dom den",),
        )
        graph_candidate = GraphCandidate(
            chunk_id="chunk-1",
            match_reasons=["symptom_match", "topic_boost"],
            metadata={"doc_id": "doc-1", "section": "symptoms"},
        )
        hybrid_candidate = HybridCandidate(
            chunk_id="chunk-1",
            sources=["dense", "sparse"],
            raw_scores={"dense": 0.82, "sparse": 4.0},
        )
        merged_candidate = MergedCandidate(
            chunk_id="chunk-1",
            sources=["graph", "dense", "sparse"],
            graph_match_reasons=["symptom_match", "graph_neighbor"],
            raw_scores={"dense": 0.82, "sparse": 4.0, "graph": 1.0},
        )

        bundle = RetrievalBundle(
            retrieval_request=request,
            graph_candidates=[graph_candidate],
            hybrid_candidates=[hybrid_candidate],
            merged_candidates=[merged_candidate],
        )

        self.assertIs(bundle.retrieval_request, request)
        self.assertEqual(
            bundle.graph_candidates[0].match_reasons,
            ("symptom_match", "topic_boost"),
        )
        self.assertIsInstance(bundle.graph_candidates, tuple)
        self.assertIsInstance(bundle.graph_candidates[0].match_reasons, tuple)
        self.assertEqual(bundle.hybrid_candidates[0].raw_scores["dense"], 0.82)
        self.assertEqual(bundle.hybrid_candidates[0].sources, ("dense", "sparse"))
        self.assertIsInstance(bundle.hybrid_candidates, tuple)
        self.assertIsInstance(bundle.hybrid_candidates[0].sources, tuple)
        self.assertEqual(
            bundle.merged_candidates[0].sources,
            ("graph", "dense", "sparse"),
        )
        self.assertIsInstance(bundle.merged_candidates, tuple)
        self.assertIsInstance(bundle.merged_candidates[0].sources, tuple)
        self.assertEqual(
            bundle.merged_candidates[0].graph_match_reasons,
            ("symptom_match", "graph_neighbor"),
        )
        self.assertIsInstance(
            bundle.merged_candidates[0].graph_match_reasons,
            tuple,
        )
        self.assertEqual(dict(bundle.graph_candidates[0].metadata)["doc_id"], "doc-1")
        self.assertIsInstance(bundle.graph_candidates[0].metadata, MappingProxyType)
        self.assertIsInstance(bundle.hybrid_candidates[0].raw_scores, MappingProxyType)
        self.assertIsInstance(bundle.merged_candidates[0].raw_scores, MappingProxyType)
        with self.assertRaises(TypeError):
            bundle.graph_candidates[0].metadata["new_key"] = "new_value"
        with self.assertRaises(TypeError):
            bundle.hybrid_candidates[0].raw_scores["dense"] = 0.5
        with self.assertRaises(TypeError):
            bundle.merged_candidates[0].raw_scores["graph"] = 2.0
        self.assertIs(retrieval.GraphCandidate, GraphCandidate)
        self.assertIs(retrieval.HybridCandidate, HybridCandidate)
        self.assertIs(retrieval.MergedCandidate, MergedCandidate)
        self.assertIs(retrieval.RetrievalBundle, RetrievalBundle)


class HybridRetrievalServiceTests(unittest.TestCase):
    class _StubGraphRetriever:
        def __init__(self, hits):
            self.hits = hits
            self.calls = []

        def retrieve(self, retrieval_request, graph_chunks):
            self.calls.append((retrieval_request, graph_chunks))
            return self.hits

    class _StubHybridRetriever:
        def __init__(self, hits):
            self.hits = hits
            self.calls = []

        def retrieve(self, retrieval_request):
            self.calls.append(retrieval_request)
            return self.hits

    def test_service_merges_overlapping_graph_and_hybrid_hits_with_provenance(self):
        from src.rag.retrieval.hybrid_service import HybridRetrievalService
        from src.rag.retrieval.hybrid_types import GraphCandidate, HybridCandidate

        request = _make_request()
        graph_chunks = ("chunk-record-1",)
        graph_hits = (
            GraphCandidate(
                chunk_id="chunk-1",
                match_reasons=("relation:source=dom den",),
                metadata={"doc_id": "doc-1"},
            ),
        )
        hybrid_hits = (
            HybridCandidate(
                chunk_id="chunk-1",
                sources=("dense", "sparse"),
                raw_scores={"dense": 0.91, "sparse": 0.73},
            ),
        )
        graph_retriever = self._StubGraphRetriever(graph_hits)
        hybrid_retriever = self._StubHybridRetriever(hybrid_hits)

        bundle = HybridRetrievalService(
            graph_retriever=graph_retriever,
            hybrid_retriever=hybrid_retriever,
        ).retrieve(retrieval_request=request, graph_chunks=graph_chunks)

        self.assertEqual(graph_retriever.calls, [(request, graph_chunks)])
        self.assertEqual(hybrid_retriever.calls, [request])
        self.assertEqual([item.chunk_id for item in bundle.merged_candidates], ["chunk-1"])
        self.assertEqual(
            bundle.merged_candidates[0].sources,
            ("graph", "dense", "sparse"),
        )
        self.assertEqual(
            bundle.merged_candidates[0].graph_match_reasons,
            ("relation:source=dom den",),
        )
        self.assertEqual(
            dict(bundle.merged_candidates[0].raw_scores),
            {"dense": 0.91, "sparse": 0.73},
        )

    def test_service_includes_hybrid_only_hits_after_graph_order(self):
        from src.rag.retrieval.hybrid_service import HybridRetrievalService
        from src.rag.retrieval.hybrid_types import GraphCandidate, HybridCandidate

        graph_hits = (
            GraphCandidate(
                chunk_id="chunk-1",
                match_reasons=("relation:target=ri sat",),
                metadata={},
            ),
        )
        hybrid_hits = (
            HybridCandidate(
                chunk_id="chunk-1",
                sources=("dense",),
                raw_scores={"dense": 0.88},
            ),
            HybridCandidate(
                chunk_id="chunk-2",
                sources=("sparse",),
                raw_scores={"sparse": 5.0},
            ),
        )

        bundle = HybridRetrievalService(
            graph_retriever=self._StubGraphRetriever(graph_hits),
            hybrid_retriever=self._StubHybridRetriever(hybrid_hits),
        ).retrieve(retrieval_request=_make_request(), graph_chunks=("chunk-record-1",))

        self.assertEqual(
            [item.chunk_id for item in bundle.merged_candidates],
            ["chunk-1", "chunk-2"],
        )
        self.assertEqual(bundle.merged_candidates[1].sources, ("sparse",))
        self.assertEqual(bundle.merged_candidates[1].graph_match_reasons, ())
        self.assertEqual(
            dict(bundle.merged_candidates[1].raw_scores),
            {"sparse": 5.0},
        )

    def test_service_bundle_carries_original_retrieval_request(self):
        from src.rag.retrieval.hybrid_service import HybridRetrievalService
        from src.rag.retrieval.hybrid_types import GraphCandidate, HybridCandidate

        request = _make_request()
        graph_hits = (
            GraphCandidate(
                chunk_id="chunk-1",
                match_reasons=("metadata:disease_name=ri sat",),
                metadata={},
            ),
        )
        hybrid_hits = (
            HybridCandidate(
                chunk_id="chunk-2",
                sources=("dense",),
                raw_scores={"dense": 0.67},
            ),
        )

        bundle = HybridRetrievalService(
            graph_retriever=self._StubGraphRetriever(graph_hits),
            hybrid_retriever=self._StubHybridRetriever(hybrid_hits),
        ).retrieve(retrieval_request=request, graph_chunks=("chunk-record-1",))

        self.assertIs(bundle.retrieval_request, request)
        self.assertEqual(bundle.graph_candidates, graph_hits)
        self.assertEqual(bundle.hybrid_candidates, hybrid_hits)

    def test_service_preserves_graph_only_hit_in_merged_candidates(self):
        from src.rag.retrieval.hybrid_service import HybridRetrievalService
        from src.rag.retrieval.hybrid_types import GraphCandidate

        graph_hits = (
            GraphCandidate(
                chunk_id="chunk-graph-only",
                match_reasons=("metadata:disease_name=dom den", "relation:target=dom den"),
                metadata={"doc_id": "doc-graph"},
            ),
        )

        bundle = HybridRetrievalService(
            graph_retriever=self._StubGraphRetriever(graph_hits),
            hybrid_retriever=self._StubHybridRetriever(()),
        ).retrieve(retrieval_request=_make_request(), graph_chunks=("chunk-record-1",))

        self.assertEqual(
            [item.chunk_id for item in bundle.merged_candidates],
            ["chunk-graph-only"],
        )
        self.assertEqual(bundle.merged_candidates[0].sources, ("graph",))
        self.assertEqual(
            bundle.merged_candidates[0].graph_match_reasons,
            ("metadata:disease_name=dom den", "relation:target=dom den"),
        )
        self.assertEqual(dict(bundle.merged_candidates[0].raw_scores), {})

    def test_service_merges_repeated_hybrid_hits_for_same_chunk(self):
        from src.rag.retrieval.hybrid_service import HybridRetrievalService
        from src.rag.retrieval.hybrid_types import HybridCandidate

        hybrid_hits = (
            HybridCandidate(
                chunk_id="chunk-repeat",
                sources=("dense", "sparse"),
                raw_scores={"dense": 0.64, "sparse": 4.2},
            ),
            HybridCandidate(
                chunk_id="chunk-repeat",
                sources=("sparse", "dense", "rerank"),
                raw_scores={"dense": 0.71, "rerank": 0.93},
            ),
        )

        bundle = HybridRetrievalService(
            graph_retriever=self._StubGraphRetriever(()),
            hybrid_retriever=self._StubHybridRetriever(hybrid_hits),
        ).retrieve(retrieval_request=_make_request(), graph_chunks=("chunk-record-1",))

        self.assertEqual(
            [item.chunk_id for item in bundle.merged_candidates],
            ["chunk-repeat"],
        )
        self.assertEqual(
            bundle.merged_candidates[0].sources,
            ("dense", "sparse", "rerank"),
        )
        self.assertEqual(bundle.merged_candidates[0].graph_match_reasons, ())
        self.assertEqual(
            dict(bundle.merged_candidates[0].raw_scores),
            {"dense": 0.71, "sparse": 4.2, "rerank": 0.93},
        )

    def test_retrieval_package_exports_hybrid_service(self):
        import src.rag.retrieval as retrieval
        from src.rag.retrieval.hybrid_service import HybridRetrievalService

        self.assertIs(retrieval.HybridRetrievalService, HybridRetrievalService)


if __name__ == "__main__":
    unittest.main()
