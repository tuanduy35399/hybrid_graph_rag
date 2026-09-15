import json
import os
import sys
import tempfile
import warnings
from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class StubResponse:
    def __init__(self, text: str):
        self.text = text


class StubModel:
    def __init__(self, model_name: str, response_text: str):
        self.model_name = model_name
        self.response_text = response_text
        self.prompts: list[str] = []

    def generate_content(self, prompt: str):
        self.prompts.append(prompt)
        return StubResponse(self.response_text)


class StubGenAI:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.configured: list[str] = []
        self.models: list[StubModel] = []

    def configure(self, api_key: str):
        self.configured.append(api_key)

    def GenerativeModel(self, model_name: str):
        model = StubModel(model_name, self.response_text)
        self.models.append(model)
        return model


class RaisingClient:
    def generate(self, prompt: str):
        raise RuntimeError("gemini unavailable")


class StaticClient:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.prompts: list[str] = []

    def generate(self, prompt: str):
        self.prompts.append(prompt)
        return self.response_text


def test_parse_tagger_response_preserves_provenance_and_normalizes_graph():
    from src.rag.indexing.tagger import (
        ChunkMetadata,
        SemanticRelation,
        parse_tagger_response,
    )

    result = parse_tagger_response(
        chunk_content="Rui la va dom vang thuong do nam.",
        heading_path=["Benh tren cay mai", "Trieu chung"],
        source_file="benh_tren_cay_mai.md",
        source_path="/data/knowledge_base/benh_tren_cay_mai.md",
        llm_text="""
        ```json
        {
          "doc_type": "disease_guide",
          "topic": "trieu chung",
          "disease_name": "ri_sat",
          "semantic_graph": [
            {"source": "dom vang", "type": "symptom", "relation": "indicates", "target": "ri_sat"}
          ]
        }
        ```
        """,
    )

    assert result == ChunkMetadata(
        source_file="benh_tren_cay_mai.md",
        source_path="/data/knowledge_base/benh_tren_cay_mai.md",
        heading_path=("Benh tren cay mai", "Trieu chung"),
        doc_type="disease_guide",
        topic="trieu chung",
        disease_name="ri_sat",
        semantic_graph=(
            SemanticRelation(
                source="dom vang",
                type="symptom",
                relation="indicates",
                target="ri_sat",
            ),
        ),
    )


def test_tagger_service_falls_back_when_provider_raises_and_keeps_provenance():
    from src.rag.indexing.tagger import TaggerService, fallback_chunk_metadata

    service = TaggerService(RaisingClient())

    result = service.tag(
        chunk_content="Cach xu ly la mai bi dom vang.",
        heading_path=["Benh tren cay mai"],
        source_file="benh_tren_cay_mai.md",
        source_path="/data/knowledge_base/benh_tren_cay_mai.md",
    )

    assert result == fallback_chunk_metadata(
        source_file="benh_tren_cay_mai.md",
        source_path="/data/knowledge_base/benh_tren_cay_mai.md",
        heading_path=("Benh tren cay mai",),
    )


def test_tagger_service_debug_prints_raw_response_and_parse_status_for_first_chunks():
    from src.rag.indexing.tagger import TaggerService

    service = TaggerService(
        StaticClient('{"doc_type":"care_guide","topic":"tuoi nuoc","disease_name":null,"semantic_graph":[]}'),
        debug_chunk_limit=1,
    )

    stream = StringIO()
    with redirect_stdout(stream):
        result = service.tag(
            chunk_content="Cach tuoi mai.",
            heading_path=["Cham soc"],
            source_file="cham_soc.md",
            source_path="/data/knowledge_base/cham_soc.md",
        )

    output = stream.getvalue()

    assert result.doc_type == "care_guide"
    assert "raw response:" in output
    assert "parse_ok=True" in output
    assert "source=cham_soc.md" in output


def test_tagger_service_debug_prints_error_when_provider_raises():
    from src.rag.indexing.tagger import TaggerService

    service = TaggerService(RaisingClient(), debug_chunk_limit=1)

    stream = StringIO()
    with redirect_stdout(stream):
        service.tag(
            chunk_content="Cach xu ly la mai bi dom vang.",
            heading_path=["Benh tren cay mai"],
            source_file="benh_tren_cay_mai.md",
            source_path="/data/knowledge_base/benh_tren_cay_mai.md",
        )

    output = stream.getvalue()

    assert "error=RuntimeError: gemini unavailable" in output


def test_tagger_service_debug_prints_parse_failure_when_response_schema_is_invalid():
    from src.rag.indexing.tagger import TaggerService

    service = TaggerService(StaticClient("not json"), debug_chunk_limit=1)

    stream = StringIO()
    with redirect_stdout(stream):
        result = service.tag(
            chunk_content="Cach tuoi mai.",
            heading_path=["Cham soc"],
            source_file="cham_soc.md",
            source_path="/data/knowledge_base/cham_soc.md",
        )

    output = stream.getvalue()

    assert result.doc_type == "unknown"
    assert "raw response:" in output
    assert "parse_ok=False" in output


def test_build_runtime_tagger_service_reads_env_and_invokes_genai_factory():
    from src.rag.indexing.tagger import build_gemini_tagger_service

    genai = StubGenAI(
        """
        {
          "doc_type": "care_guide",
          "topic": "tuoi nuoc",
          "disease_name": null,
          "semantic_graph": []
        }
        """
    )

    service = build_gemini_tagger_service(
        env_map={"GEMINI_API_KEY": "secret-key"},
        genai_module=genai,
        model_name="gemini-2.5-flash",
    )

    assert genai.configured == ["secret-key"]
    assert len(genai.models) == 1
    assert genai.models[0].model_name == "gemini-2.5-flash"

    result = service.tag(
        chunk_content="Cach tuoi mai.",
        heading_path=["Cham soc"],
        source_file="cham_soc.md",
        source_path="/data/knowledge_base/cham_soc.md",
    )

    assert result.source_file == "cham_soc.md"
    assert result.source_path == "/data/knowledge_base/cham_soc.md"
    assert result.heading_path == ("Cham soc",)
    assert result.doc_type == "care_guide"
    assert result.topic == "tuoi nuoc"


def test_build_runtime_tagger_service_reads_gemini_api_key_from_os_environ():
    from src.rag.indexing.tagger import build_gemini_tagger_service

    original = os.environ.get("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = "runtime-secret"
    try:
        genai = StubGenAI(
            """
            {
              "doc_type": "care_guide",
              "topic": "tuoi nuoc",
              "disease_name": null,
              "semantic_graph": []
            }
            """
        )

        service = build_gemini_tagger_service(genai_module=genai)

        assert genai.configured == ["runtime-secret"]
        assert len(genai.models) == 1
        assert service.__class__.__name__ == "TaggerService"
    finally:
        if original is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = original


def test_build_runtime_tagger_service_falls_back_to_src_env_file_when_os_environ_is_missing():
    from src.rag.indexing.tagger import build_gemini_tagger_service

    original = os.environ.pop("GEMINI_API_KEY", None)
    original_cwd = Path.cwd()
    temp_dir = Path(tempfile.mkdtemp())
    src_dir = temp_dir / "src"
    src_dir.mkdir()
    (src_dir / ".env").write_text('GEMINI_API_KEY="file-secret"\n', encoding="utf-8")
    try:
        os.chdir(temp_dir)
        genai = StubGenAI(
            """
            {
              "doc_type": "care_guide",
              "topic": "tuoi nuoc",
              "disease_name": null,
              "semantic_graph": []
            }
            """
        )

        service = build_gemini_tagger_service(genai_module=genai)

        assert genai.configured == ["file-secret"]
        assert len(genai.models) == 1
        assert service.__class__.__name__ == "TaggerService"
    finally:
        os.chdir(original_cwd)
        if original is not None:
            os.environ["GEMINI_API_KEY"] = original


def test_build_runtime_tagger_service_reads_debug_chunk_limit_from_env():
    from src.rag.indexing.tagger import build_gemini_tagger_service

    genai = StubGenAI(
        """
        {
          "doc_type": "care_guide",
          "topic": "tuoi nuoc",
          "disease_name": null,
          "semantic_graph": []
        }
        """
    )

    service = build_gemini_tagger_service(
        env_map={"GEMINI_API_KEY": "secret-key", "RAG_TAGGER_DEBUG_CHUNKS": "2"},
        genai_module=genai,
    )

    assert service.debug_chunk_limit == 2


def test_run_ingestion_pipeline_walks_markdown_files_and_writes_stable_manifest(tmp_path: Path):
    from src.rag.indexing.pipeline import run_ingestion_pipeline
    from src.rag.indexing.tagger import ChunkMetadata, SemanticRelation

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "b.md").write_text("# Beta\nSecond file.\n", encoding="utf-8")
    (docs_dir / "notes.txt").write_text("ignore me", encoding="utf-8")
    nested_dir = docs_dir / "nested"
    nested_dir.mkdir()
    (nested_dir / "a.md").write_text(
        "# Alpha\nFirst paragraph.\n\n## Details\nMore text.\n",
        encoding="utf-8",
    )

    class EchoTagger:
        def __init__(self):
            self.calls: list[tuple[str, tuple[str, ...], str, str]] = []

        def tag(self, chunk_content: str, heading_path, source_file: str, source_path: str):
            heading_path = tuple(heading_path)
            self.calls.append((chunk_content, heading_path, source_file, source_path))
            return ChunkMetadata(
                source_file=source_file,
                source_path=source_path,
                heading_path=heading_path,
                doc_type="doc",
                topic=heading_path[-1] if heading_path else "General",
                disease_name=None,
                semantic_graph=(
                    SemanticRelation(
                        source=source_file,
                        type="document",
                        relation="describes",
                        target=heading_path[-1] if heading_path else "General",
                    ),
                ),
            )

    output_file = tmp_path / "manifest.jsonl"
    tagger = EchoTagger()

    records = run_ingestion_pipeline(
        docs_dir=str(docs_dir),
        output_file=str(output_file),
        tagger=tagger,
    )

    second_run = run_ingestion_pipeline(
        docs_dir=str(docs_dir),
        output_file=str(tmp_path / "manifest-2.jsonl"),
        tagger=tagger,
    )

    assert [record.chunk_id for record in records] == [record.chunk_id for record in second_run]
    assert [record.metadata.source_file for record in records] == ["a.md", "a.md", "b.md"]
    assert output_file.exists()

    lines = output_file.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines]

    assert [item["metadata"]["source_file"] for item in payloads] == ["a.md", "a.md", "b.md"]
    assert payloads[0]["metadata"]["heading_path"] == ["Alpha"]
    assert payloads[1]["metadata"]["heading_path"] == ["Alpha", "Details"]
    assert payloads[2]["metadata"]["heading_path"] == ["Beta"]
    assert all(item["chunk_id"] for item in payloads)


def test_run_ingestion_pipeline_warns_when_gemini_tagger_is_unavailable():
    from src.rag.indexing.models import ChunkMetadata
    from src.rag.indexing.pipeline import run_ingestion_pipeline

    class FallbackTagger:
        fallback_reason = "Gemini tagger unavailable; using fallback metadata only."

        def tag(self, chunk_content: str, heading_path, source_file: str, source_path: str):
            return ChunkMetadata(
                source_file=source_file,
                source_path=source_path,
                heading_path=tuple(heading_path),
                doc_type="unknown",
                topic=heading_path[-1] if heading_path else "General",
                disease_name=None,
                semantic_graph=(),
            )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        docs_dir = temp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "care.md").write_text("# Care\nWater daily.\n", encoding="utf-8")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch("src.rag.indexing.pipeline.build_gemini_tagger_service", return_value=FallbackTagger()):
                records = run_ingestion_pipeline(
                    docs_dir=str(docs_dir),
                    output_file=str(temp_path / "manifest.jsonl"),
                )

        assert len(records) == 1
        assert any("Gemini tagger unavailable" in str(item.message) for item in caught)
