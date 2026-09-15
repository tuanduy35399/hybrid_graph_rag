from typing import Protocol

from src.rag.answering.types import AnswerContext
from src.rag.indexing.models import ChunkRecord
from src.rag.retrieval.hybrid_types import RetrievalBundle


class ChunkResolver(Protocol):
    def resolve(self, chunk_ids: tuple[str, ...]) -> tuple[ChunkRecord, ...]:
        ...


class RerankerClient(Protocol):
    def rank(self, retrieval_bundle: RetrievalBundle) -> tuple[str, ...]:
        ...


class AnswerGeneratorClient(Protocol):
    def generate(self, answer_context: AnswerContext) -> str:
        ...
