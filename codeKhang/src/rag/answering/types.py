from __future__ import annotations
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from src.rag.indexing.models import ChunkMetadata
from src.rag.retrieval.types import RetrievalRequest

if TYPE_CHECKING:
    from src.rag.runtime.types import TextRuntimeResult


def _freeze_mapping(mapping: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(mapping))


def _freeze_tuple(values) -> tuple:
    return tuple(values)


@dataclass(frozen=True)
class MaterializedChunk:
    chunk_id: str
    content: str
    metadata: ChunkMetadata
    sources: tuple[str, ...]
    raw_scores: Mapping[str, float] = field(default_factory=dict)
    graph_match_reasons: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "sources", _freeze_tuple(self.sources))
        object.__setattr__(self, "raw_scores", _freeze_mapping(self.raw_scores))
        object.__setattr__(
            self,
            "graph_match_reasons",
            _freeze_tuple(self.graph_match_reasons),
        )


@dataclass(frozen=True)
class AnswerContext:
    retrieval_request: RetrievalRequest
    selected_chunks: tuple[MaterializedChunk, ...]
    rendered_context: str

    def __post_init__(self):
        object.__setattr__(self, "selected_chunks", _freeze_tuple(self.selected_chunks))


@dataclass(frozen=True)
class GroundedAnswer:
    answer_text: str
    citations: tuple[str, ...]
    used_chunk_ids: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "citations", _freeze_tuple(self.citations))
        object.__setattr__(self, "used_chunk_ids", _freeze_tuple(self.used_chunk_ids))


@dataclass(frozen=True)
class TextAnswerResult:
    runtime_result: TextRuntimeResult
    answer_context: AnswerContext
    grounded_answer: GroundedAnswer
