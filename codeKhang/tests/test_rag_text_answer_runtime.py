import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TextAnswerRuntimeTests(unittest.TestCase):
    def test_answer_text_query_uses_injected_runtime_service_when_provided(self):
        from src.rag.answering.types import TextAnswerResult
        from src.rag.runtime.text_answer import answer_text_query

        class StubRuntimeService:
            def __init__(self):
                self.calls = []

            def answer_text_query(self, raw_text):
                self.calls.append(raw_text)
                return TextAnswerResult(
                    runtime_result=object(),
                    answer_context=object(),
                    grounded_answer=object(),
                )

        runtime_service = StubRuntimeService()

        result = answer_text_query(
            "la mai bi dom vang",
            runtime_service=runtime_service,
        )

        self.assertEqual(runtime_service.calls, ["la mai bi dom vang"])
        self.assertIsNotNone(result)

    def test_build_text_only_rag_runtime_service_composes_repo_default_factories(self):
        import src.rag.runtime.text_answer as runtime_module

        original_build_runtime_service = runtime_module.build_runtime_service
        original_build_retrieval_service = runtime_module.build_runtime_text_retrieval_service
        original_build_graph_provider = runtime_module.build_runtime_graph_chunk_provider
        original_build_answering_service = runtime_module.build_runtime_text_answering_service

        captured = {}

        class StubRuntimeService:
            pass

        try:
            runtime_module.build_runtime_text_retrieval_service = lambda **kwargs: ("retrieval", kwargs)
            runtime_module.build_runtime_graph_chunk_provider = lambda manifest_path: (
                "graph",
                manifest_path,
            )
            runtime_module.build_runtime_text_answering_service = lambda **kwargs: ("answer", kwargs)

            def fake_build_runtime_service(**kwargs):
                captured.update(kwargs)
                return StubRuntimeService()

            runtime_module.build_runtime_service = fake_build_runtime_service

            runtime = runtime_module.build_text_only_rag_runtime_service()
            retrieval_service = captured["text_retrieval_service_factory"]()
            graph_provider = captured["graph_chunk_provider_factory"]()
            answering_service = captured["text_answering_service_factory"]()
        finally:
            runtime_module.build_runtime_service = original_build_runtime_service
            runtime_module.build_runtime_text_retrieval_service = original_build_retrieval_service
            runtime_module.build_runtime_graph_chunk_provider = original_build_graph_provider
            runtime_module.build_runtime_text_answering_service = original_build_answering_service

        self.assertIsInstance(runtime, StubRuntimeService)
        self.assertIn("text_retrieval_service_factory", captured)
        self.assertIn("graph_chunk_provider_factory", captured)
        self.assertIn("text_answering_service_factory", captured)

        self.assertEqual(retrieval_service[0], "retrieval")
        self.assertEqual(graph_provider[0], "graph")
        self.assertEqual(answering_service[0], "answer")
