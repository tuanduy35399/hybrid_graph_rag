import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_indexing_package_exports_public_models_and_chunker():
    import src.rag.indexing as indexing
    from src.rag.indexing.chunker import chunk_markdown
    from src.rag.indexing.models import ChunkMetadata, ChunkRecord, SemanticRelation

    assert indexing.ChunkMetadata is ChunkMetadata
    assert indexing.ChunkRecord is ChunkRecord
    assert indexing.SemanticRelation is SemanticRelation
    assert indexing.chunk_markdown is chunk_markdown
    assert indexing.__all__ == [
        "ChunkMetadata",
        "ChunkRecord",
        "DEFAULT_DOCS_DIR",
        "DEFAULT_MAX_LENGTH_TOKENS",
        "DEFAULT_OUTPUT_FILE",
        "SemanticRelation",
        "build_parser",
        "chunk_markdown",
        "main",
        "run_ingestion_pipeline",
    ]


def test_chunk_metadata_preserves_source_provenance_and_heading_path_defaults():
    from src.rag.indexing.models import ChunkMetadata

    metadata = ChunkMetadata(
        source_file="benh_tren_cay_mai.md",
        source_path="data/knowledge_base/benh_tren_cay_mai.md",
        heading_path=["Bệnh trên cây mai", "Triệu chứng"],
        doc_type="disease_guide",
        topic="triệu chứng",
        disease_name="ri_sat",
    )

    assert metadata.source_file == "benh_tren_cay_mai.md"
    assert metadata.source_path == "data/knowledge_base/benh_tren_cay_mai.md"
    assert metadata.heading_path == ["Bệnh trên cây mai", "Triệu chứng"]
    assert metadata.semantic_graph == []


def test_chunk_markdown_preserves_heading_ancestry_for_recursive_splits():
    from src.rag.indexing.chunker import chunk_markdown

    markdown = """# Plant Care

Intro text.

## Symptoms

alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu

### Treatment

short advice
"""

    chunks = chunk_markdown(markdown, max_length_tokens=3)

    assert [chunk["heading_path"] for chunk in chunks] == [
        ["Plant Care"],
        ["Plant Care", "Symptoms"],
        ["Plant Care", "Symptoms"],
        ["Plant Care", "Symptoms"],
        ["Plant Care", "Symptoms"],
        ["Plant Care", "Symptoms", "Treatment"],
    ]
    assert all(chunk["content"].strip() for chunk in chunks)
    assert chunks[1]["content"].split() == ["alpha", "beta", "gamma"]
    assert chunks[2]["content"].split() == ["delta", "epsilon", "zeta"]


def test_chunk_markdown_uses_general_heading_before_first_header():
    from src.rag.indexing.chunker import chunk_markdown

    chunks = chunk_markdown("lead in text\n\n# Main Topic\nbody", max_length_tokens=100)

    assert chunks[0]["heading_path"] == ["General"]
    assert chunks[0]["content"] == "lead in text"
    assert chunks[1]["heading_path"] == ["Main Topic"]
    assert chunks[1]["content"] == "body"
