import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_cli_main_uses_repo_friendly_defaults_and_calls_pipeline():
    from src.rag.indexing import cli

    calls = []

    def fake_run_ingestion_pipeline(**kwargs):
        calls.append(kwargs)
        return ["ok"]

    result = cli.main([], runner=fake_run_ingestion_pipeline)

    assert result == ["ok"]
    assert calls == [
        {
            "docs_dir": "data/knowledge_base",
            "output_file": "manifest.jsonl",
            "tagger": None,
            "max_length_tokens": 1000,
        }
    ]


def test_repo_root_wrapper_delegates_to_cli_main():
    import scripts.run_rag_ingestion as run_rag_ingestion

    calls = []
    original = run_rag_ingestion.run_ingestion_main

    def fake_run_ingestion_main(argv=None):
        calls.append(argv)
        return ["record"]

    run_rag_ingestion.run_ingestion_main = fake_run_ingestion_main
    try:
        result = run_rag_ingestion.main(["--output-file", "custom.jsonl"])
    finally:
        run_rag_ingestion.run_ingestion_main = original

    assert result == ["record"]
    assert calls == [["--output-file", "custom.jsonl"]]


def test_repo_root_wrapper_is_runnable_as_a_script():
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_rag_ingestion.py"),
        "--help",
    ]

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Run the Phase 3 RAG ingestion pipeline" in result.stdout


def test_cli_module_export_shape_matches_package_exports():
    import src.rag.indexing as indexing
    from src.rag.indexing.cli import build_parser, main
    from src.rag.indexing.pipeline import run_ingestion_pipeline

    assert indexing.build_parser is build_parser
    assert indexing.main is main
    assert indexing.run_ingestion_pipeline is run_ingestion_pipeline


def test_retrieval_cli_main_indexes_manifest_into_qdrant_with_named_vectors():
    import src.rag.retrieval.cli as retrieval_cli

    calls = {}

    class StubConfig:
        qdrant_dense_vector_name = "dense"
        qdrant_sparse_vector_name = "sparse"

    result = retrieval_cli.main(
        ["--manifest-file", "custom.jsonl"],
        load_manifest_records=lambda path: calls.update({"manifest_path": path}) or ("record-1",),
        build_runtime_embedding_client=lambda: "embedding-client",
        build_runtime_vector_store_client=lambda: "vector-store-client",
        build_runtime_provider_config=lambda: StubConfig(),
        index_manifest_records=lambda **kwargs: calls.update(kwargs) or 5,
        printer=lambda message: calls.update({"printed": message}),
        repo_root=REPO_ROOT,
    )

    assert result == 5
    assert calls["manifest_path"] == REPO_ROOT / "custom.jsonl"
    assert calls["records"] == ("record-1",)
    assert calls["embedding_client"] == "embedding-client"
    assert calls["vector_store_client"] == "vector-store-client"
    assert calls["dense_vector_name"] == "dense"
    assert calls["sparse_vector_name"] == "sparse"
    assert calls["printed"] == "Indexed 5 chunks into Qdrant"


def test_retrieval_cli_module_export_shape_matches_package_exports():
    import src.rag.retrieval as retrieval
    from src.rag.retrieval.cli import build_parser, main

    assert retrieval.build_manifest_index_parser is build_parser
    assert retrieval.run_manifest_indexing_main is main


def test_main_can_emit_a_manifest_for_temp_docs():
    from src.rag.indexing.cli import main

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        docs_dir = temp_path / "kb"
        docs_dir.mkdir()
        (docs_dir / "a.md").write_text("# Alpha\nBody text.\n", encoding="utf-8")

        output_file = temp_path / "manifest.jsonl"

        records = main(
            [
                "--docs-dir",
                str(docs_dir),
                "--output-file",
                str(output_file),
                "--max-length-tokens",
                "100",
            ]
        )

        assert len(records) == 1
        assert output_file.exists()
        assert output_file.read_text(encoding="utf-8").strip()


def test_repo_root_manifest_indexing_wrapper_delegates_to_retrieval_cli_main():
    import scripts.run_rag_manifest_indexing as run_rag_manifest_indexing

    calls = {}
    original = run_rag_manifest_indexing.run_manifest_indexing_main

    def fake_run_manifest_indexing_main(argv=None, **kwargs):
        calls["argv"] = argv
        calls["kwargs"] = kwargs
        return 9

    run_rag_manifest_indexing.run_manifest_indexing_main = fake_run_manifest_indexing_main
    try:
        result = run_rag_manifest_indexing.main(["--manifest-file", "custom.jsonl"])
    finally:
        run_rag_manifest_indexing.run_manifest_indexing_main = original

    assert result == 9
    assert calls["argv"] == ["--manifest-file", "custom.jsonl"]
    assert calls["kwargs"]["repo_root"] == run_rag_manifest_indexing.REPO_ROOT
