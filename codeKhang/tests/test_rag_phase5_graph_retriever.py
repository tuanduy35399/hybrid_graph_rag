import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.indexing.models import ChunkMetadata, ChunkRecord, SemanticRelation
from src.rag.integration.types import BranchType
from src.rag.normalization.types import QueryType
from src.rag.planning.types import EntityHint, QueryIntent
from src.rag.retrieval.types import RetrievalRequest


def _make_request(
    *,
    query_text="la mai bi dom vang la benh gi",
    query_terms=("dom vang", "ri sat"),
    entity_hints=(EntityHint(kind="disease_name", value="ri_sat"),),
    subqueries=("la mai bi dom vang la benh gi",),
):
    return RetrievalRequest(
        query_text=query_text,
        branch=BranchType.TEXT_ONLY_BRANCH,
        query_type=QueryType.DISEASE_QUERY,
        intent=QueryIntent.DISEASE_LOOKUP,
        query_terms=query_terms,
        entity_hints=entity_hints,
        topic_hints=("symptoms",),
        filters=(),
        subqueries=subqueries,
    )


def _make_chunk(
    *,
    chunk_id,
    disease_name=None,
    semantic_graph=(),
    topic="symptoms",
):
    return ChunkRecord(
        chunk_id=chunk_id,
        content="stub content",
        metadata=ChunkMetadata(
            source_file=f"{chunk_id}.md",
            source_path=f"data/knowledge_base/{chunk_id}.md",
            heading_path=("Benh",),
            doc_type="disease_guide",
            topic=topic,
            disease_name=disease_name,
            semantic_graph=semantic_graph,
        ),
    )


class GraphRetrieverTests(unittest.TestCase):
    def test_graph_retriever_matches_disease_name_and_relation_endpoints(self):
        from src.rag.retrieval.graph_retriever import GraphRetriever

        request = _make_request(
            query_terms=("dom vang", "ri sat"),
            entity_hints=(EntityHint(kind="disease_name", value="ri_sat"),),
        )
        chunks = (
            _make_chunk(
                chunk_id="chunk-1",
                disease_name="ri_sat",
                semantic_graph=(
                    SemanticRelation(
                        source="dom vang",
                        type="symptom",
                        relation="indicates",
                        target="ri_sat",
                    ),
                ),
            ),
            _make_chunk(
                chunk_id="chunk-2",
                disease_name="phan trang",
                semantic_graph=(
                    SemanticRelation(
                        source="cat tia",
                        type="care_action",
                        relation="treats",
                        target="phan trang",
                    ),
                ),
            ),
        )

        hits = GraphRetriever().retrieve(request, chunks)

        self.assertEqual([hit.chunk_id for hit in hits], ["chunk-1"])
        self.assertEqual(
            hits[0].match_reasons,
            (
                "metadata:disease_name=ri_sat",
                "relation:source=dom vang",
                "relation:target=ri_sat",
            ),
        )
        self.assertEqual(
            dict(hits[0].metadata),
            {
                "source_file": "chunk-1.md",
                "source_path": "data/knowledge_base/chunk-1.md",
                "doc_type": "disease_guide",
                "topic": "symptoms",
            },
        )

    def test_graph_retriever_returns_empty_without_structured_match(self):
        from src.rag.retrieval.graph_retriever import GraphRetriever

        request = _make_request(
            query_text="cat tia sao cho dung",
            query_terms=("cat tia",),
            entity_hints=(),
            subqueries=("cat tia sao cho dung",),
        )
        chunks = (
            _make_chunk(
                chunk_id="chunk-1",
                disease_name="ri_sat",
                semantic_graph=(),
            ),
        )

        self.assertEqual(GraphRetriever().retrieve(request, chunks), ())

    def test_graph_retriever_ignores_non_disease_hints_for_entity_target_matching(self):
        from src.rag.retrieval.graph_retriever import GraphRetriever

        request = _make_request(
            query_text="cay mai co van de gi",
            query_terms=("mai",),
            entity_hints=(
                EntityHint(kind="plant_name", value="ri_sat"),
                EntityHint(kind="symptom", value="ri_sat"),
            ),
            subqueries=("cay mai co van de gi",),
        )
        chunks = (
            _make_chunk(
                chunk_id="chunk-1",
                disease_name="ri_sat",
                semantic_graph=(
                    SemanticRelation(
                        source="dom vang",
                        type="symptom",
                        relation="indicates",
                        target="ri_sat",
                    ),
                ),
            ),
        )

        self.assertEqual(GraphRetriever().retrieve(request, chunks), ())


if __name__ == "__main__":
    unittest.main()
