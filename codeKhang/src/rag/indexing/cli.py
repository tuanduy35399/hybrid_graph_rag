from __future__ import annotations

import argparse
from typing import Callable, Sequence

from .pipeline import run_ingestion_pipeline

DEFAULT_DOCS_DIR = "data/knowledge_base"
DEFAULT_OUTPUT_FILE = "manifest.jsonl"
DEFAULT_MAX_LENGTH_TOKENS = 1000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 3 RAG ingestion pipeline")
    parser.add_argument(
        "--docs-dir",
        default=DEFAULT_DOCS_DIR,
        help="Directory containing markdown knowledge documents",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="Path to the JSONL manifest output",
    )
    parser.add_argument(
        "--max-length-tokens",
        type=int,
        default=DEFAULT_MAX_LENGTH_TOKENS,
        help="Maximum token budget for one chunk before recursive splitting",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    runner: Callable[..., object] = run_ingestion_pipeline,
) -> object:
    args = build_parser().parse_args(argv)
    records = runner(
        docs_dir=args.docs_dir,
        output_file=args.output_file,
        tagger=None,
        max_length_tokens=args.max_length_tokens,
    )
    print(f"Processed {len(records)} chunks into {args.output_file}")
    return records


if __name__ == "__main__":
    raise SystemExit(main())
