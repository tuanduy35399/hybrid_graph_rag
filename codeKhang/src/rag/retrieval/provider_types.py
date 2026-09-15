from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from src.rag.indexing.models import ChunkMetadata
from src.rag.retrieval.types import RetrievalFilter


@dataclass(frozen=True)
class ProviderConfig:
    qdrant_api_key: str
    qdrant_url: str
    qdrant_collection_name: str
    qdrant_dense_vector_name: str | None = None
    qdrant_sparse_vector_name: str | None = None
    embedding_model_name: str = "bkai-foundation-models/vietnamese-bi-encoder"


@dataclass(frozen=True)
class Phase5RuntimeConfig:
    collection_name: str
    dense_vector_name: str
    sparse_vector_name: str


@dataclass(frozen=True)
class BM25DocumentQuery:
    text: str
    model: str = "qdrant/bm25"


@dataclass(frozen=True)
class VectorPoint:
    point_id: str | int
    vector: tuple[float, ...]
    payload: Mapping[str, object]
    dense_vector_name: str | None = None
    sparse_vector_name: str | None = None
    sparse_document: BM25DocumentQuery | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", tuple(float(value) for value in self.vector))
        object.__setattr__(self, "payload", _freeze_json_like(self.payload))


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    content: str
    score: float
    metadata: ChunkMetadata


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        ...


class VectorStoreClient(Protocol):
    def ensure_collection(
        self,
        vector_size: int,
        dense_vector_name: str | None = None,
        sparse_vector_name: str | None = None,
    ) -> None:
        ...

    def upsert_points(self, points: tuple[VectorPoint, ...]) -> None:
        ...

    def assert_hybrid_collection_ready(
        self,
        *,
        dense_vector_name: str,
        sparse_vector_name: str,
    ) -> None:
        ...

    def search(
        self,
        query_vector: tuple[float, ...],
        filters: tuple[RetrievalFilter, ...],
        top_k: int,
    ) -> tuple[RetrievedChunk, ...]:
        ...


def _freeze_json_like(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json_like(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_like(item) for item in value)
    return value
