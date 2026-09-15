from collections.abc import Iterable

from src.rag.retrieval.provider_types import EmbeddingClient, ProviderConfig


class LocalEmbeddingClient(EmbeddingClient):
    def __init__(self, model):
        self.model = model

    def embed_texts(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()

        vector_rows = tuple(self.model.encode(texts, normalize_embeddings=True))
        if len(texts) == 1 and _looks_like_single_vector(vector_rows):
            raise ValueError(
                "LocalEmbeddingClient.encode() must return one embedding row per input text; "
                "got a 1-D vector for a singleton batch."
            )

        if len(vector_rows) != len(texts):
            raise ValueError(
                f"LocalEmbeddingClient.encode() returned {len(vector_rows)} embedding rows for "
                f"{len(texts)} input texts."
            )

        return tuple(tuple(float(value) for value in vector) for vector in vector_rows)


def build_local_embedding_client(
    config: ProviderConfig,
    sentence_transformer_cls,
):
    model = sentence_transformer_cls(config.embedding_model_name)
    return LocalEmbeddingClient(model=model)


def _looks_like_single_vector(vectors) -> bool:
    if not vectors:
        return False
    return _is_scalar_like(vectors[0])


def _is_scalar_like(value) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return True
    if isinstance(value, Iterable):
        return False
    return True
