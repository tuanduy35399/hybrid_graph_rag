from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag.retrieval.cli import (
    build_parser,
    main as run_manifest_indexing_main,
)
from src.rag.retrieval.config import build_runtime_provider_config
from src.rag.retrieval.manifest_indexer import index_manifest_records, load_manifest_records
from src.rag.retrieval.runtime import (
    build_runtime_embedding_client,
    build_runtime_vector_store_client,
)

def main(argv: list[str] | None = None) -> int:
    return run_manifest_indexing_main(
        argv,
        load_manifest_records=load_manifest_records,
        build_runtime_embedding_client=build_runtime_embedding_client,
        build_runtime_vector_store_client=build_runtime_vector_store_client,
        build_runtime_provider_config=build_runtime_provider_config,
        index_manifest_records=index_manifest_records,
        printer=print,
        repo_root=REPO_ROOT,
    )


def run_cli(argv: list[str] | None = None) -> int:
    main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
