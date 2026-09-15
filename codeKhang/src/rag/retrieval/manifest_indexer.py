import json
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.rag.indexing.models import ChunkMetadata, ChunkRecord, SemanticRelation
from src.rag.retrieval.provider_types import (
    BM25DocumentQuery,
    EmbeddingClient,
    VectorPoint,
    VectorStoreClient,
)


def load_manifest_records(manifest_path: Path) -> tuple[ChunkRecord, ...]:
    records = []
    for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(_manifest_line_to_record(line))
    return tuple(records)


def prepare_vector_points(
    records: tuple[ChunkRecord, ...],
    embedding_client: EmbeddingClient,
    *,
    dense_vector_name: str | None = None,
    sparse_vector_name: str | None = None,
) -> tuple[VectorPoint, ...]:
    if not records:
        return ()

    vectors = embedding_client.embed_texts(tuple(record.content for record in records))
    if len(vectors) != len(records):
        raise ValueError(
            f"Embedding client returned {len(vectors)} vectors for {len(records)} manifest records."
        )
    return tuple(
        VectorPoint(
            point_id=_to_qdrant_point_id(record.chunk_id),
            vector=vector,
            dense_vector_name=dense_vector_name,
            sparse_vector_name=sparse_vector_name,
            sparse_document=_build_sparse_document(record, sparse_vector_name=sparse_vector_name),
            payload=_record_to_payload(record),
        )
        for record, vector in zip(records, vectors)
    )


def index_manifest_records(
    records: tuple[ChunkRecord, ...],
    embedding_client: EmbeddingClient,
    vector_store_client: VectorStoreClient,
    *,
    dense_vector_name: str | None = None,
    sparse_vector_name: str | None = None,
) -> int:
    points = prepare_vector_points(
        records=records,
        embedding_client=embedding_client,
        dense_vector_name=dense_vector_name,
        sparse_vector_name=sparse_vector_name,
    )
    if not points:
        return 0

    vector_store_client.ensure_collection(
        vector_size=len(points[0].vector),
        dense_vector_name=dense_vector_name,
        sparse_vector_name=sparse_vector_name,
    )
    vector_store_client.upsert_points(points)
    return len(points)


def _build_sparse_document(
    record: ChunkRecord,
    *,
    sparse_vector_name: str | None,
) -> BM25DocumentQuery | None:
    if sparse_vector_name is None:
        return None
    return BM25DocumentQuery(text=record.content, model="qdrant/bm25")


def _to_qdrant_point_id(chunk_id: str) -> str | int:
    if chunk_id.isdigit():
        return int(chunk_id)

    try:
        return str(uuid.UUID(chunk_id))
    except (ValueError, AttributeError, TypeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"chunk:{chunk_id}"))


def _manifest_line_to_record(line: str) -> ChunkRecord:
    payload = json.loads(line)
    chunk_id = _require_mapping_value(payload, "chunk_id", "Manifest record")
    content = _require_mapping_value(payload, "content", "Manifest record")
    metadata_payload = _require_mapping_value(payload, "metadata", "Manifest record")
    if not isinstance(metadata_payload, Mapping):
        raise TypeError("Manifest field 'metadata' must be a mapping.")

    return ChunkRecord(
        chunk_id=str(chunk_id),
        content=str(content),
        metadata=ChunkMetadata(
            source_file=str(
                _require_mapping_value(metadata_payload, "source_file", "Manifest metadata")
            ),
            source_path=str(
                _require_mapping_value(metadata_payload, "source_path", "Manifest metadata")
            ),
            heading_path=_coerce_heading_path(metadata_payload.get("heading_path", ())),
            doc_type=metadata_payload.get("doc_type", "unknown"),
            topic=metadata_payload.get("topic", "General"),
            disease_name=metadata_payload.get("disease_name"),
            semantic_graph=tuple(
                _coerce_semantic_relation(item)
                for item in _coerce_semantic_graph(metadata_payload.get("semantic_graph", ()))
            ),
        ),
    )


def _record_to_payload(record: ChunkRecord) -> dict[str, object]:
    return {
        "chunk_id": record.chunk_id,
        "content": record.content,
        "source_file": record.metadata.source_file,
        "source_path": record.metadata.source_path,
        "heading_path": list(record.metadata.heading_path),
        "doc_type": record.metadata.doc_type,
        "topic": record.metadata.topic,
        "disease_name": record.metadata.disease_name,
        "semantic_graph": [
            {
                "source": item.source,
                "type": item.type,
                "relation": item.relation,
                "target": item.target,
            }
            for item in record.metadata.semantic_graph
        ],
    }


def _require_mapping_value(payload: Mapping[str, object], key: str, label: str) -> object:
    if key not in payload:
        raise ValueError(f"{label} is missing required field '{key}'.")
    return payload[key]


def _coerce_heading_path(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise ValueError(
            "Manifest metadata field 'heading_path' must be a sequence of strings, not a string."
        )
    if not isinstance(value, Sequence):
        raise TypeError("Manifest metadata field 'heading_path' must be a sequence of strings.")
    return tuple(str(item) for item in value)


def _coerce_semantic_graph(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            "Manifest metadata field 'semantic_graph' must be a sequence of relation mappings."
        )
    return tuple(value)


def _coerce_semantic_relation(value: object) -> SemanticRelation:
    if not isinstance(value, Mapping):
        raise TypeError("Manifest semantic_graph items must be relation mappings.")
    return SemanticRelation(
        source=str(_require_mapping_value(value, "source", "Manifest semantic_graph item")),
        type=str(_require_mapping_value(value, "type", "Manifest semantic_graph item")),
        relation=str(_require_mapping_value(value, "relation", "Manifest semantic_graph item")),
        target=str(_require_mapping_value(value, "target", "Manifest semantic_graph item")),
    )
