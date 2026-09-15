from __future__ import annotations

import json
import uuid
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .chunker import chunk_markdown
from .models import ChunkMetadata, ChunkRecord
from .tagger import TaggerService, build_gemini_tagger_service, fallback_chunk_metadata


def run_ingestion_pipeline(
    docs_dir: str,
    output_file: str,
    tagger: Any | None = None,
    max_length_tokens: int = 1000,
) -> list[ChunkRecord]:
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {docs_dir}")

    chunk_tagger = _resolve_tagger(tagger)
    records: list[ChunkRecord] = []

    markdown_files = sorted(
        (path for path in docs_path.rglob("*.md") if path.is_file()),
        key=lambda path: (path.name.lower(), path.relative_to(docs_path).as_posix().lower()),
    )

    for file_path in markdown_files:
        content = file_path.read_text(encoding="utf-8")
        chunks = chunk_markdown(content, max_length_tokens=max_length_tokens)

        for index, chunk in enumerate(chunks):
            heading_path = tuple(chunk["heading_path"])
            metadata = _tag_chunk(
                chunk_tagger,
                chunk_content=chunk["content"],
                heading_path=heading_path,
                source_file=file_path.name,
                source_path=str(file_path),
            )

            record = ChunkRecord(
                chunk_id=_build_chunk_id(
                    source_path=str(file_path),
                    chunk_index=index,
                    heading_path=heading_path,
                    content=chunk["content"],
                ),
                content=chunk["content"],
                metadata=metadata,
            )
            records.append(record)

    _write_manifest(output_file, records)
    return records


def _resolve_tagger(tagger: Any | None) -> Any:
    if tagger is not None:
        return tagger
    resolved = build_gemini_tagger_service()
    fallback_reason = getattr(resolved, "fallback_reason", None)
    if isinstance(fallback_reason, str) and fallback_reason:
        warnings.warn(fallback_reason, RuntimeWarning, stacklevel=2)
    return resolved


def _tag_chunk(
    tagger: Any,
    chunk_content: str,
    heading_path: tuple[str, ...],
    source_file: str,
    source_path: str,
) -> ChunkMetadata:
    if hasattr(tagger, "tag"):
        return tagger.tag(chunk_content, heading_path, source_file, source_path)
    if callable(tagger):
        return tagger(chunk_content, heading_path, source_file, source_path)
    return fallback_chunk_metadata(source_file, source_path, heading_path)


def _build_chunk_id(
    source_path: str,
    chunk_index: int,
    heading_path: tuple[str, ...],
    content: str,
) -> str:
    fingerprint = f"{source_path}::{chunk_index}::{' > '.join(heading_path)}::{content}"
    return uuid.uuid5(uuid.NAMESPACE_URL, fingerprint).hex[:12]


def _write_manifest(output_file: str, records: list[ChunkRecord]) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_record_to_json(record), ensure_ascii=False) + "\n")


def _record_to_json(record: ChunkRecord) -> dict[str, object]:
    payload = asdict(record)
    metadata = payload["metadata"]
    metadata["heading_path"] = list(metadata["heading_path"])
    return payload
