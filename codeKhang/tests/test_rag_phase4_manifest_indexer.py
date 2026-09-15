import importlib.util
import os
import sys
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.indexing.models import ChunkMetadata, ChunkRecord, SemanticRelation
from src.rag.retrieval.provider_types import ProviderConfig, VectorPoint


def test_load_manifest_records_reads_jsonl_into_chunk_records():
    from src.rag.retrieval.manifest_indexer import load_manifest_records

    temp_dir = Path(tempfile.mkdtemp())
    manifest_path = temp_dir / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                '{"chunk_id":"chunk-1","content":"La mai co dom vang.","metadata":{"source_file":"benh.md","source_path":"data/knowledge_base/benh.md","heading_path":["Benh","Trieu chung"],"doc_type":"disease_guide","topic":"symptoms","disease_name":"ri_sat","semantic_graph":[{"source":"dom vang","type":"symptom","relation":"indicates","target":"ri_sat"}]}}',
                '{"chunk_id":"chunk-2","content":"Xu ly bang cat tia.","metadata":{"source_file":"tri.md","source_path":"data/knowledge_base/tri.md","heading_path":["Dieu tri"],"doc_type":"treatment_guide","topic":"treatment","semantic_graph":[]}}',
            ]
        ),
        encoding="utf-8",
    )

    records = load_manifest_records(manifest_path)

    assert len(records) == 2
    assert isinstance(records[0], ChunkRecord)
    assert records[0].chunk_id == "chunk-1"
    assert records[0].metadata == ChunkMetadata(
        source_file="benh.md",
        source_path="data/knowledge_base/benh.md",
        heading_path=("Benh", "Trieu chung"),
        doc_type="disease_guide",
        topic="symptoms",
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
    assert records[1].metadata.semantic_graph == ()


def test_prepare_vector_points_embeds_texts_and_builds_payload():
    from src.rag.retrieval.manifest_indexer import prepare_vector_points

    class StubEmbeddingClient:
        def __init__(self):
            self.calls = []

        def embed_texts(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
            self.calls.append(texts)
            return ((0.1, 0.2), (0.3, 0.4))

    records = (
        ChunkRecord(
            chunk_id="chunk-1",
            content="La mai co dom vang.",
            metadata=ChunkMetadata(
                source_file="benh.md",
                source_path="data/knowledge_base/benh.md",
                heading_path=("Benh", "Trieu chung"),
                doc_type="disease_guide",
                topic="symptoms",
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
        ),
        ChunkRecord(
            chunk_id="chunk-2",
            content="Xu ly bang cat tia.",
            metadata=ChunkMetadata(
                source_file="tri.md",
                source_path="data/knowledge_base/tri.md",
                heading_path=("Dieu tri",),
                doc_type="treatment_guide",
                topic="treatment",
                disease_name=None,
                semantic_graph=(),
            ),
        ),
    )

    embedding_client = StubEmbeddingClient()
    points = prepare_vector_points(records, embedding_client)

    assert embedding_client.calls == [("La mai co dom vang.", "Xu ly bang cat tia.")]
    assert points == (
        VectorPoint(
            point_id=str(uuid.uuid5(uuid.NAMESPACE_URL, "chunk:chunk-1")),
            vector=(0.1, 0.2),
            dense_vector_name=None,
            sparse_vector_name=None,
            sparse_document=None,
            payload={
                "chunk_id": "chunk-1",
                "content": "La mai co dom vang.",
                "source_file": "benh.md",
                "source_path": "data/knowledge_base/benh.md",
                "heading_path": ["Benh", "Trieu chung"],
                "doc_type": "disease_guide",
                "topic": "symptoms",
                "disease_name": "ri_sat",
                "semantic_graph": [
                    {
                        "source": "dom vang",
                        "type": "symptom",
                        "relation": "indicates",
                        "target": "ri_sat",
                    }
                ],
            },
        ),
        VectorPoint(
            point_id=str(uuid.uuid5(uuid.NAMESPACE_URL, "chunk:chunk-2")),
            vector=(0.3, 0.4),
            dense_vector_name=None,
            sparse_vector_name=None,
            sparse_document=None,
            payload={
                "chunk_id": "chunk-2",
                "content": "Xu ly bang cat tia.",
                "source_file": "tri.md",
                "source_path": "data/knowledge_base/tri.md",
                "heading_path": ["Dieu tri"],
                "doc_type": "treatment_guide",
                "topic": "treatment",
                "disease_name": None,
                "semantic_graph": [],
            },
        ),
    )


def test_prepare_vector_points_can_attach_named_dense_and_server_sparse_contract():
    from src.rag.retrieval.manifest_indexer import prepare_vector_points
    from src.rag.retrieval.provider_types import BM25DocumentQuery

    class StubEmbeddingClient:
        def embed_texts(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
            return ((0.1, 0.2),)

    records = (
        ChunkRecord(
            chunk_id="chunk-1",
            content="La mai co dom vang.",
            metadata=ChunkMetadata(
                source_file="benh.md",
                source_path="data/knowledge_base/benh.md",
                heading_path=("Benh",),
                doc_type="disease_guide",
                topic="symptoms",
                disease_name="ri_sat",
                semantic_graph=(),
            ),
        ),
    )

    points = prepare_vector_points(
        records,
        StubEmbeddingClient(),
        dense_vector_name="dense-default",
        sparse_vector_name="sparse-bm25",
    )

    assert points == (
        VectorPoint(
            point_id=str(uuid.uuid5(uuid.NAMESPACE_URL, "chunk:chunk-1")),
            vector=(0.1, 0.2),
            dense_vector_name="dense-default",
            sparse_vector_name="sparse-bm25",
            sparse_document=BM25DocumentQuery(
                text="La mai co dom vang.",
                model="qdrant/bm25",
            ),
            payload={
                "chunk_id": "chunk-1",
                "content": "La mai co dom vang.",
                "source_file": "benh.md",
                "source_path": "data/knowledge_base/benh.md",
                "heading_path": ["Benh"],
                "doc_type": "disease_guide",
                "topic": "symptoms",
                "disease_name": "ri_sat",
                "semantic_graph": [],
            },
        ),
    )


def test_prepare_vector_points_converts_non_qdrant_chunk_ids_to_stable_uuid_point_ids():
    from src.rag.retrieval.manifest_indexer import prepare_vector_points

    class StubEmbeddingClient:
        def embed_texts(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
            return ((0.1, 0.2),)

    records = (
        ChunkRecord(
            chunk_id="459851ebd030",
            content="La mai co dom vang.",
            metadata=ChunkMetadata(
                source_file="benh.md",
                source_path="data/knowledge_base/benh.md",
                heading_path=("Benh",),
                doc_type="disease_guide",
                topic="symptoms",
                disease_name="ri_sat",
                semantic_graph=(),
            ),
        ),
    )

    points = prepare_vector_points(records, StubEmbeddingClient())

    assert points[0].point_id == str(uuid.uuid5(uuid.NAMESPACE_URL, "chunk:459851ebd030"))
    assert points[0].payload["chunk_id"] == "459851ebd030"


def test_prepare_vector_points_converts_numeric_chunk_ids_to_unsigned_integer_point_ids():
    from src.rag.retrieval.manifest_indexer import prepare_vector_points

    class StubEmbeddingClient:
        def embed_texts(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
            return ((0.1, 0.2),)

    records = (
        ChunkRecord(
            chunk_id="556322497504",
            content="La mai co dom vang.",
            metadata=ChunkMetadata(
                source_file="benh.md",
                source_path="data/knowledge_base/benh.md",
                heading_path=("Benh",),
                doc_type="disease_guide",
                topic="symptoms",
                disease_name="ri_sat",
                semantic_graph=(),
            ),
        ),
    )

    points = prepare_vector_points(records, StubEmbeddingClient())

    assert points[0].point_id == 556322497504
    assert points[0].payload["chunk_id"] == "556322497504"


def test_prepare_vector_points_rejects_vector_count_mismatch():
    from src.rag.retrieval.manifest_indexer import prepare_vector_points

    class StubEmbeddingClient:
        def embed_texts(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
            return ((0.1, 0.2),)

    records = (
        ChunkRecord(
            chunk_id="chunk-1",
            content="La mai co dom vang.",
            metadata=ChunkMetadata(
                source_file="benh.md",
                source_path="data/knowledge_base/benh.md",
                heading_path=("Benh",),
                doc_type="disease_guide",
                topic="symptoms",
                disease_name="ri_sat",
                semantic_graph=(),
            ),
        ),
        ChunkRecord(
            chunk_id="chunk-2",
            content="Xu ly bang cat tia.",
            metadata=ChunkMetadata(
                source_file="tri.md",
                source_path="data/knowledge_base/tri.md",
                heading_path=("Dieu tri",),
                doc_type="treatment_guide",
                topic="treatment",
                disease_name=None,
                semantic_graph=(),
            ),
        ),
    )

    try:
        prepare_vector_points(records, StubEmbeddingClient())
    except ValueError as exc:
        assert str(exc) == "Embedding client returned 1 vectors for 2 manifest records."
    else:
        raise AssertionError("Expected prepare_vector_points() to reject vector count mismatch")


def test_index_manifest_records_ensures_collection_upserts_points_and_returns_count():
    from src.rag.retrieval.manifest_indexer import index_manifest_records

    class StubEmbeddingClient:
        def __init__(self):
            self.calls = []

        def embed_texts(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
            self.calls.append(texts)
            return ((0.1, 0.2, 0.3),)

    class StubVectorStoreClient:
        def __init__(self):
            self.ensure_calls = []
            self.upsert_calls = []

        def ensure_collection(
            self,
            vector_size: int,
            dense_vector_name: str | None = None,
            sparse_vector_name: str | None = None,
        ) -> None:
            self.ensure_calls.append((vector_size, dense_vector_name, sparse_vector_name))

        def upsert_points(self, points: tuple[VectorPoint, ...]) -> None:
            self.upsert_calls.append(points)

    records = (
        ChunkRecord(
            chunk_id="chunk-1",
            content="La mai co dom vang.",
            metadata=ChunkMetadata(
                source_file="benh.md",
                source_path="data/knowledge_base/benh.md",
                heading_path=("Benh",),
                doc_type="disease_guide",
                topic="symptoms",
                disease_name="ri_sat",
                semantic_graph=(),
            ),
        ),
    )

    embedding_client = StubEmbeddingClient()
    vector_store_client = StubVectorStoreClient()

    count = index_manifest_records(records, embedding_client, vector_store_client)

    assert count == 1
    assert embedding_client.calls == [("La mai co dom vang.",)]
    assert vector_store_client.ensure_calls == [(3, None, None)]
    assert len(vector_store_client.upsert_calls) == 1
    assert vector_store_client.upsert_calls[0][0].point_id == str(
        uuid.uuid5(uuid.NAMESPACE_URL, "chunk:chunk-1")
    )


def test_index_manifest_records_can_prepare_hybrid_ready_collection_and_points():
    from src.rag.retrieval.manifest_indexer import index_manifest_records

    class StubEmbeddingClient:
        def embed_texts(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
            return ((0.1, 0.2, 0.3),)

    class StubVectorStoreClient:
        def __init__(self):
            self.ensure_calls = []
            self.upsert_calls = []

        def ensure_collection(
            self,
            vector_size: int,
            dense_vector_name: str | None = None,
            sparse_vector_name: str | None = None,
        ) -> None:
            self.ensure_calls.append((vector_size, dense_vector_name, sparse_vector_name))

        def upsert_points(self, points: tuple[VectorPoint, ...]) -> None:
            self.upsert_calls.append(points)

    records = (
        ChunkRecord(
            chunk_id="chunk-1",
            content="La mai co dom vang.",
            metadata=ChunkMetadata(
                source_file="benh.md",
                source_path="data/knowledge_base/benh.md",
                heading_path=("Benh",),
                doc_type="disease_guide",
                topic="symptoms",
                disease_name="ri_sat",
                semantic_graph=(),
            ),
        ),
    )

    vector_store_client = StubVectorStoreClient()
    count = index_manifest_records(
        records,
        StubEmbeddingClient(),
        vector_store_client,
        dense_vector_name="dense-default",
        sparse_vector_name="sparse-bm25",
    )

    assert count == 1
    assert vector_store_client.ensure_calls == [(3, "dense-default", "sparse-bm25")]
    assert vector_store_client.upsert_calls[0][0].dense_vector_name == "dense-default"
    assert vector_store_client.upsert_calls[0][0].sparse_vector_name == "sparse-bm25"
    assert vector_store_client.upsert_calls[0][0].sparse_document.text == "La mai co dom vang."


def test_load_manifest_records_rejects_string_heading_path():
    from src.rag.retrieval.manifest_indexer import load_manifest_records

    temp_dir = Path(tempfile.mkdtemp())
    manifest_path = temp_dir / "manifest.jsonl"
    manifest_path.write_text(
        '{"chunk_id":"chunk-1","content":"La mai co dom vang.","metadata":{"source_file":"benh.md","source_path":"data/knowledge_base/benh.md","heading_path":"Benh/Trieu chung"}}',
        encoding="utf-8",
    )

    try:
        load_manifest_records(manifest_path)
    except ValueError as exc:
        assert str(exc) == "Manifest metadata field 'heading_path' must be a sequence of strings, not a string."
    else:
        raise AssertionError("Expected malformed heading_path to raise ValueError")


def test_load_manifest_records_rejects_non_sequence_semantic_graph():
    from src.rag.retrieval.manifest_indexer import load_manifest_records

    temp_dir = Path(tempfile.mkdtemp())
    manifest_path = temp_dir / "manifest.jsonl"
    manifest_path.write_text(
        '{"chunk_id":"chunk-1","content":"La mai co dom vang.","metadata":{"source_file":"benh.md","source_path":"data/knowledge_base/benh.md","semantic_graph":{"source":"dom vang"}}}',
        encoding="utf-8",
    )

    try:
        load_manifest_records(manifest_path)
    except TypeError as exc:
        assert (
            str(exc)
            == "Manifest metadata field 'semantic_graph' must be a sequence of relation mappings."
        )
    else:
        raise AssertionError("Expected malformed semantic_graph to raise TypeError")


def test_build_runtime_embedding_client_uses_runtime_provider_config_and_local_builder():
    import src.rag.retrieval.runtime as runtime_module

    config = ProviderConfig(
        qdrant_api_key="secret",
        qdrant_url="https://example.qdrant.io",
        qdrant_collection_name="mai-kb",
    )
    captured = {}

    class StubSentenceTransformer:
        pass

    with patch.object(
        runtime_module,
        "build_runtime_provider_config",
        return_value=config,
    ), patch.object(
        runtime_module,
        "build_local_embedding_client",
        side_effect=lambda *, config, sentence_transformer_cls: captured.update(
            {
                "config": config,
                "sentence_transformer_cls": sentence_transformer_cls,
            }
        )
        or "embedding-client",
    ), patch.dict(
        sys.modules,
        {"sentence_transformers": type("StubModule", (), {"SentenceTransformer": StubSentenceTransformer})},
    ):
        client = runtime_module.build_runtime_embedding_client()

    assert client == "embedding-client"
    assert captured["config"] is config
    assert captured["sentence_transformer_cls"] is StubSentenceTransformer


def test_build_runtime_embedding_client_raises_clear_error_when_dependency_missing():
    import src.rag.retrieval.runtime as runtime_module

    original_module = sys.modules.pop("sentence_transformers", None)

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sentence_transformers":
            raise ImportError("missing sentence-transformers")
        return real_import(name, globals, locals, fromlist, level)

    try:
        with patch("builtins.__import__", side_effect=fake_import):
            try:
                runtime_module.build_runtime_embedding_client()
            except RuntimeError as exc:
                assert (
                    str(exc)
                    == "sentence-transformers is required to build the runtime embedding client."
                )
            else:
                raise AssertionError("Expected missing sentence-transformers to raise RuntimeError")
    finally:
        if original_module is not None:
            sys.modules["sentence_transformers"] = original_module


def test_build_runtime_vector_store_client_uses_runtime_provider_config_and_qdrant_builder():
    import src.rag.retrieval.runtime as runtime_module

    config = ProviderConfig(
        qdrant_api_key="secret",
        qdrant_url="https://example.qdrant.io",
        qdrant_collection_name="mai-kb",
    )

    class StubQdrantClient:
        def __init__(self, *, url: str, api_key: str):
            self.url = url
            self.api_key = api_key

    class StubModels:
        pass

    stub_module = type(
        "StubQdrantModule",
        (),
        {"QdrantClient": StubQdrantClient, "models": StubModels},
    )

    with patch.object(
        runtime_module,
        "build_runtime_provider_config",
        return_value=config,
    ), patch.dict(sys.modules, {"qdrant_client": stub_module}):
        client = runtime_module.build_runtime_vector_store_client()

    assert client.__class__.__name__ == "QdrantVectorStoreClient"
    assert client.collection_name == "mai-kb"
    assert client.qdrant_models is StubModels
    assert client.qdrant_client.url == "https://example.qdrant.io"
    assert client.qdrant_client.api_key == "secret"


def test_build_runtime_vector_store_client_raises_clear_error_when_dependency_missing():
    import src.rag.retrieval.runtime as runtime_module

    original_module = sys.modules.pop("qdrant_client", None)
    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "qdrant_client":
            raise ImportError("missing qdrant-client")
        return real_import(name, globals, locals, fromlist, level)

    try:
        with patch("builtins.__import__", side_effect=fake_import):
            try:
                runtime_module.build_runtime_vector_store_client()
            except RuntimeError as exc:
                assert (
                    str(exc)
                    == "qdrant-client is required to build the runtime vector store client."
                )
            else:
                raise AssertionError("Expected missing qdrant-client to raise RuntimeError")
    finally:
        if original_module is not None:
            sys.modules["qdrant_client"] = original_module


def test_manifest_indexing_script_parses_manifest_file_and_indexes_records():
    script_module = _load_script_module()
    manifest_path = Path(tempfile.mkdtemp()) / "manifest.jsonl"
    captured = {}

    with patch.object(
        script_module,
        "load_manifest_records",
        side_effect=lambda path: captured.update({"manifest_path": path}) or ("record-1",),
    ), patch.object(
        script_module,
        "build_runtime_embedding_client",
        return_value="embedding-client",
    ), patch.object(
        script_module,
        "build_runtime_vector_store_client",
        return_value="vector-store-client",
    ), patch.object(
        script_module,
        "build_runtime_provider_config",
        return_value=type(
            "Cfg",
            (),
            {
                "qdrant_dense_vector_name": "dense",
                "qdrant_sparse_vector_name": "sparse",
            },
        )(),
    ), patch.object(
        script_module,
        "index_manifest_records",
        side_effect=lambda *,
        records,
        embedding_client,
        vector_store_client,
        dense_vector_name,
        sparse_vector_name: captured.update(
            {
                "records": records,
                "embedding_client": embedding_client,
                "vector_store_client": vector_store_client,
                "dense_vector_name": dense_vector_name,
                "sparse_vector_name": sparse_vector_name,
            }
        )
        or 7,
    ), patch("builtins.print") as mock_print:
        exit_code = script_module.main(["--manifest-file", str(manifest_path)])

    assert exit_code == 7
    assert captured == {
        "manifest_path": manifest_path,
        "records": ("record-1",),
        "embedding_client": "embedding-client",
        "vector_store_client": "vector-store-client",
        "dense_vector_name": "dense",
        "sparse_vector_name": "sparse",
    }
    mock_print.assert_called_once_with("Indexed 7 chunks into Qdrant")


def test_manifest_indexing_script_main_resolves_default_manifest_relative_to_repo_root():
    script_module = _load_script_module()
    temp_repo_root = Path(tempfile.mkdtemp())
    captured = {}

    with patch.object(script_module, "REPO_ROOT", temp_repo_root), patch.object(
        script_module,
        "load_manifest_records",
        side_effect=lambda path: captured.update({"manifest_path": path}) or (),
    ), patch.object(
        script_module,
        "build_runtime_embedding_client",
        return_value="embedding-client",
    ), patch.object(
        script_module,
        "build_runtime_vector_store_client",
        return_value="vector-store-client",
    ), patch.object(
        script_module,
        "build_runtime_provider_config",
        return_value=type(
            "Cfg",
            (),
            {
                "qdrant_dense_vector_name": "dense",
                "qdrant_sparse_vector_name": "sparse",
            },
        )(),
    ), patch.object(
        script_module,
        "index_manifest_records",
        return_value=0,
    ), patch("builtins.print"):
        result = script_module.main([])

    assert result == 0
    assert captured["manifest_path"] == temp_repo_root / "manifest.jsonl"


def test_manifest_indexing_script_run_cli_returns_zero_on_success():
    script_module = _load_script_module()

    with patch.object(script_module, "main", return_value=7) as mock_main:
        exit_code = script_module.run_cli(["--manifest-file", "custom.jsonl"])

    assert exit_code == 0
    mock_main.assert_called_once_with(["--manifest-file", "custom.jsonl"])


def test_manifest_indexing_script_parser_accepts_manifest_file_flag():
    script_module = _load_script_module()

    args = script_module.build_parser().parse_args(["--manifest-file", "custom.jsonl"])

    assert args.manifest_file == "custom.jsonl"


def test_manifest_indexing_script_parser_defaults_manifest_file():
    script_module = _load_script_module()

    args = script_module.build_parser().parse_args([])

    assert args.manifest_file == "manifest.jsonl"


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_rag_manifest_indexing.py"
    spec = importlib.util.spec_from_file_location("run_rag_manifest_indexing", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
