import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class AnsweringPromptTests(unittest.TestCase):
    def test_rendered_context_includes_citation_labels_and_chunk_content(self):
        from src.rag.answering.prompt import render_context_block
        from src.rag.answering.types import MaterializedChunk
        from src.rag.indexing.models import ChunkMetadata

        chunk = MaterializedChunk(
            chunk_id="chunk-1",
            content="La co dom vang.",
            metadata=ChunkMetadata(
                source_file="kb.md",
                source_path="data/knowledge_base/kb.md",
                heading_path=("Benh", "Ri sat"),
                topic="disease",
            ),
            sources=("graph",),
            raw_scores={"hybrid": 0.91},
            graph_match_reasons=("relation:target=ri sat",),
        )
        second_chunk = MaterializedChunk(
            chunk_id="chunk-2",
            content="Bon phan can doi.",
            metadata=ChunkMetadata(
                source_file="care.md",
                source_path="data/knowledge_base/care.md",
                heading_path=(),
                topic="care",
            ),
            sources=("dense",),
        )

        rendered = render_context_block((chunk, second_chunk))

        self.assertEqual(
            rendered,
            "\n\n".join(
                (
                    "\n".join(
                        (
                            "[kb.md#chunk-1]",
                            "topic: disease",
                            "heading: Benh > Ri sat",
                            "La co dom vang.",
                        )
                    ),
                    "\n".join(
                        (
                            "[care.md#chunk-2]",
                            "topic: care",
                            "heading:",
                            "Bon phan can doi.",
                        )
                    ),
                )
            ),
        )

    def test_gemini_answer_generator_client_returns_response_text(self):
        from src.rag.answering.gemini_client import GeminiAnswerGeneratorClient
        from src.rag.answering.types import AnswerContext
        from src.rag.retrieval.types import RetrievalRequest
        from src.rag.integration.types import BranchType
        from src.rag.normalization.types import QueryType
        from src.rag.planning.types import QueryIntent

        class StubModel:
            def __init__(self):
                self.calls = []

            def generate_content(self, prompt):
                self.calls.append(prompt)
                return type("Response", (), {"text": "Tra loi grounded"})()

        client = GeminiAnswerGeneratorClient(StubModel())
        answer_context = AnswerContext(
            retrieval_request=RetrievalRequest(
                query_text="La mai bi dom vang la benh gi?",
                branch=BranchType.TEXT_ONLY_BRANCH,
                query_type=QueryType.DISEASE_QUERY,
                intent=QueryIntent.DISEASE_LOOKUP,
                query_terms=("la mai", "dom vang"),
                entity_hints=(),
                topic_hints=("disease",),
                filters=(),
                subqueries=(),
            ),
            selected_chunks=(),
            rendered_context="[kb.md#chunk-1]\ntopic: disease\nheading: Benh > Ri sat\nLa co dom vang.",
        )
        answer = client.generate(answer_context)

        self.assertEqual(answer, "Tra loi grounded")
        self.assertEqual(len(client.model.calls), 1)
        self.assertIn("La mai bi dom vang la benh gi?", client.model.calls[0])
        self.assertIn("La co dom vang.", client.model.calls[0])
        self.assertIn("CONTEXT:", client.model.calls[0])
