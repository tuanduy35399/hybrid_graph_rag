import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class AnsweringRuntimeTests(unittest.TestCase):
    def test_build_runtime_text_answering_service_composes_manifest_resolver_and_gemini_generator(self):
        import src.rag.answering.runtime as runtime_module
        from src.rag.answering.gemini_client import GeminiAnswerGeneratorClient
        from src.rag.answering.selector import ScorePreservingReranker
        from src.rag.answering.service import TextAnsweringService

        original_load_env = runtime_module.load_env_file
        original_read_key = runtime_module.read_gemini_api_key_from_env_map
        original_build_resolver = runtime_module.build_manifest_chunk_resolver
        original_genai = runtime_module.google_generativeai

        class StubGenAI:
            def __init__(self):
                self.configured = None
                self.models = []

            def configure(self, **kwargs):
                self.configured = kwargs

            class GenerativeModel:
                def __init__(self, model_name):
                    self.model_name = model_name

        manifest_path = Path(tempfile.mkdtemp()) / "manifest.jsonl"
        manifest_path.write_text("", encoding="utf-8")
        call_log = []

        try:
            def fake_load_env_file(path):
                call_log.append(("load_env_file", path))
                return {"GEMINI_API_KEY": "secret"}

            def fake_read_key(env_map):
                call_log.append(("read_gemini_api_key_from_env_map", env_map))
                return env_map["GEMINI_API_KEY"]

            def fake_build_resolver(path):
                call_log.append(("build_manifest_chunk_resolver", path))
                return ("resolver", str(path))

            stub_genai = StubGenAI()
            runtime_module.load_env_file = fake_load_env_file
            runtime_module.read_gemini_api_key_from_env_map = fake_read_key
            runtime_module.build_manifest_chunk_resolver = fake_build_resolver
            runtime_module.google_generativeai = stub_genai

            service = runtime_module.build_runtime_text_answering_service(
                manifest_path=manifest_path,
            )
        finally:
            runtime_module.load_env_file = original_load_env
            runtime_module.read_gemini_api_key_from_env_map = original_read_key
            runtime_module.build_manifest_chunk_resolver = original_build_resolver
            runtime_module.google_generativeai = original_genai

        self.assertIsInstance(service, TextAnsweringService)
        self.assertIsInstance(service.reranker, ScorePreservingReranker)
        self.assertEqual(service.chunk_resolver, ("resolver", str(manifest_path)))
        self.assertIsInstance(service.answer_generator, GeminiAnswerGeneratorClient)
        self.assertEqual(service.answer_generator.model.model_name, "gemini-2.5-flash")
        self.assertEqual(
            call_log,
            [
                ("load_env_file", Path("src/.env")),
                ("read_gemini_api_key_from_env_map", {"GEMINI_API_KEY": "secret"}),
                ("build_manifest_chunk_resolver", manifest_path),
            ],
        )
        self.assertEqual(stub_genai.configured, {"api_key": "secret"})

    def test_answering_package_exports_task4_public_surface(self):
        import src.rag.answering as answering_pkg
        from src.rag.answering.runtime import build_runtime_text_answering_service
        from src.rag.answering.service import TextAnsweringService
        from src.rag.answering.types import AnswerContext, GroundedAnswer, MaterializedChunk, TextAnswerResult

        self.assertIs(answering_pkg.AnswerContext, AnswerContext)
        self.assertIs(answering_pkg.GroundedAnswer, GroundedAnswer)
        self.assertIs(answering_pkg.MaterializedChunk, MaterializedChunk)
        self.assertIs(answering_pkg.TextAnswerResult, TextAnswerResult)
        self.assertIs(answering_pkg.TextAnsweringService, TextAnsweringService)
        self.assertIs(
            answering_pkg.build_runtime_text_answering_service,
            build_runtime_text_answering_service,
        )

    def test_build_runtime_text_answering_service_raises_when_google_generativeai_is_missing(self):
        import src.rag.answering.runtime as runtime_module

        original_genai = runtime_module.google_generativeai
        try:
            runtime_module.google_generativeai = None

            with self.assertRaisesRegex(
                RuntimeError,
                "google-generativeai is required for answer synthesis.",
            ):
                runtime_module.build_runtime_text_answering_service()
        finally:
            runtime_module.google_generativeai = original_genai

    def test_build_runtime_text_answering_service_propagates_missing_gemini_api_key(self):
        import src.rag.answering.runtime as runtime_module

        original_load_env = runtime_module.load_env_file
        original_build_resolver = runtime_module.build_manifest_chunk_resolver
        original_genai = runtime_module.google_generativeai

        class StubGenAI:
            def configure(self, **kwargs):
                raise AssertionError("configure should not be called when GEMINI_API_KEY is missing")

            class GenerativeModel:
                def __init__(self, model_name):
                    self.model_name = model_name

        manifest_path = Path(tempfile.mkdtemp()) / "manifest.jsonl"
        manifest_path.write_text("", encoding="utf-8")

        try:
            runtime_module.load_env_file = lambda path: {}
            runtime_module.build_manifest_chunk_resolver = lambda path: ("resolver", str(path))
            runtime_module.google_generativeai = StubGenAI()

            with self.assertRaisesRegex(KeyError, "GEMINI_API_KEY"):
                runtime_module.build_runtime_text_answering_service(
                    manifest_path=manifest_path,
                )
        finally:
            runtime_module.load_env_file = original_load_env
            runtime_module.build_manifest_chunk_resolver = original_build_resolver
            runtime_module.google_generativeai = original_genai
