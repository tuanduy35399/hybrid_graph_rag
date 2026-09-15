from dataclasses import dataclass, field


class _FlexibleTuple(tuple):
    def __eq__(self, other):
        try:
            return tuple(self) == tuple(other)
        except TypeError:
            return False


class HeadingPath(_FlexibleTuple):
    pass


class SemanticGraph(_FlexibleTuple):
    pass


@dataclass(frozen=True)
class SemanticRelation:
    source: str
    type: str
    relation: str
    target: str


@dataclass(frozen=True)
class ChunkMetadata:
    source_file: str
    source_path: str
    heading_path: HeadingPath = field(default_factory=HeadingPath)
    doc_type: str = "unknown"
    topic: str = "General"
    disease_name: str | None = None
    semantic_graph: SemanticGraph = field(default_factory=SemanticGraph)

    def __post_init__(self) -> None:
        object.__setattr__(self, "heading_path", HeadingPath(self.heading_path))
        object.__setattr__(self, "semantic_graph", SemanticGraph(self.semantic_graph))


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    content: str
    metadata: ChunkMetadata
