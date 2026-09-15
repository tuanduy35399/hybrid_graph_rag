from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


def _freeze_mapping(mapping: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(mapping))


def _freeze_tuple(values) -> tuple:
    return tuple(values)


@dataclass(frozen=True)
class GraphCandidate:
    chunk_id: str
    match_reasons: tuple[str, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "match_reasons", _freeze_tuple(self.match_reasons))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class HybridCandidate:
    chunk_id: str
    sources: tuple[str, ...]
    raw_scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "sources", _freeze_tuple(self.sources))
        object.__setattr__(self, "raw_scores", _freeze_mapping(self.raw_scores))


@dataclass(frozen=True)
class MergedCandidate:
    chunk_id: str
    sources: tuple[str, ...]
    graph_match_reasons: tuple[str, ...] = ()
    raw_scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "sources", _freeze_tuple(self.sources))
        object.__setattr__(
            self,
            "graph_match_reasons",
            _freeze_tuple(self.graph_match_reasons),
        )
        object.__setattr__(self, "raw_scores", _freeze_mapping(self.raw_scores))


@dataclass(frozen=True)
class RetrievalBundle:
    retrieval_request: object
    graph_candidates: tuple[GraphCandidate, ...]
    hybrid_candidates: tuple[HybridCandidate, ...]
    merged_candidates: tuple[MergedCandidate, ...]

    def __post_init__(self):
        object.__setattr__(self, "graph_candidates", _freeze_tuple(self.graph_candidates))
        object.__setattr__(
            self,
            "hybrid_candidates",
            _freeze_tuple(self.hybrid_candidates),
        )
        object.__setattr__(
            self,
            "merged_candidates",
            _freeze_tuple(self.merged_candidates),
        )
