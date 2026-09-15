from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence

from src.rag.retrieval.config import build_runtime_provider_config
from src.rag.retrieval.manifest_indexer import index_manifest_records, load_manifest_records
from src.rag.retrieval.runtime import (
    build_runtime_embedding_client,
    build_runtime_vector_store_client,
)

DEFAULT_MANIFEST_FILE = "manifest.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index a Phase 3 manifest into Qdrant")
    parser.add_argument(
        "--manifest-file",
        default=DEFAULT_MANIFEST_FILE,
        help="Path to the Phase 3 manifest file",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    load_manifest_records: Callable[[Path], tuple[object, ...]] = load_manifest_records,
    build_runtime_embedding_client: Callable[[], object] = build_runtime_embedding_client,
    build_runtime_vector_store_client: Callable[[], object] = build_runtime_vector_store_client,
    build_runtime_provider_config: Callable[[], object] = build_runtime_provider_config,
    index_manifest_records: Callable[..., int] = index_manifest_records,
    printer: Callable[[str], None] = print,
    repo_root: Path | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    resolved_repo_root = repo_root or Path(__file__).resolve().parents[3]
    manifest_path = _resolve_manifest_path(args.manifest_file, resolved_repo_root)
    provider_config = build_runtime_provider_config()
    records = load_manifest_records(manifest_path)
    count = index_manifest_records(
        records=records,
        embedding_client=build_runtime_embedding_client(),
        vector_store_client=build_runtime_vector_store_client(),
        dense_vector_name=getattr(provider_config, "qdrant_dense_vector_name", None),
        sparse_vector_name=getattr(provider_config, "qdrant_sparse_vector_name", None),
    )
    printer(f"Indexed {count} chunks into Qdrant")
    return count


def run_cli(argv: Sequence[str] | None = None) -> int:
    main(argv)
    return 0


def _resolve_manifest_path(manifest_file: str, repo_root: Path) -> Path:
    manifest_path = Path(manifest_file)
    if manifest_path.is_absolute():
        return manifest_path
    return repo_root / manifest_path
