from collections.abc import Mapping, Sequence

from src.rag.indexing.models import ChunkMetadata, SemanticRelation
from src.rag.retrieval.hybrid_qdrant import build_hybrid_qdrant_query_payload
from src.rag.retrieval.provider_types import RetrievedChunk, VectorPoint
from src.rag.retrieval.sparse_types import SparseQuery
from src.rag.retrieval.types import RetrievalFilter


def build_qdrant_filter_payload(
    filters: tuple[RetrievalFilter, ...],
) -> tuple[tuple[str, str], ...]:
    return tuple((item.field, item.value) for item in filters)


class QdrantVectorStoreClient:
    def __init__(
        self,
        qdrant_client,
        qdrant_models,
        collection_name: str,
        hybrid_ready: bool = False,
    ):
        self.qdrant_client = qdrant_client
        self.qdrant_models = qdrant_models
        self.collection_name = collection_name
        self.hybrid_ready = hybrid_ready

    def ensure_collection(
        self,
        vector_size: int,
        dense_vector_name: str | None = None,
        sparse_vector_name: str | None = None,
    ) -> None:
        dense_params = self.qdrant_models.VectorParams(
            size=vector_size,
            distance=self.qdrant_models.Distance.COSINE,
        )
        if dense_vector_name is None and sparse_vector_name is None:
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=dense_params,
            )
            return

        vectors_config = (
            {dense_vector_name: dense_params}
            if dense_vector_name is not None
            else {}
        )
        sparse_vectors_config = (
            {sparse_vector_name: self.qdrant_models.SparseVectorParams()}
            if sparse_vector_name is not None
            else None
        )
        self.qdrant_client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )

    def upsert_points(self, points: tuple[VectorPoint, ...]) -> None:
        provider_points = tuple(
            self.qdrant_models.PointStruct(
                id=point.point_id,
                vector=_build_point_vector_payload(point),
                payload=_to_plain_payload_mapping(point.payload),
            )
            for point in points
        )
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=provider_points,
        )

    def assert_hybrid_collection_ready(
        self,
        *,
        dense_vector_name: str,
        sparse_vector_name: str,
    ) -> None:
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        dense_vectors = _read_collection_config_mapping(
            collection_info,
            ("config", "params", "vectors"),
            ("config", "params", "vectors_config"),
        )
        sparse_vectors = _read_collection_config_mapping(
            collection_info,
            ("config", "params", "sparse_vectors"),
            ("config", "params", "sparse_vectors_config"),
        )

        if dense_vector_name not in dense_vectors:
            raise ValueError(
                f"Qdrant collection '{self.collection_name}' is missing dense vector '{dense_vector_name}'."
            )
        if sparse_vector_name not in sparse_vectors:
            raise ValueError(
                f"Qdrant collection '{self.collection_name}' is missing sparse vector '{sparse_vector_name}'."
            )

    def search(
        self,
        query_vector: tuple[float, ...],
        filters: tuple[RetrievalFilter, ...],
        top_k: int,
    ) -> tuple[RetrievedChunk, ...]:
        query_filter = _build_qdrant_query_filter(
            qdrant_models=self.qdrant_models,
            filters=filters,
        )
        hits = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
        )
        return _hits_to_retrieved_chunks(hits)

    def search_hybrid(
        self,
        dense_vector: tuple[float, ...],
        sparse_query,
        filters: tuple[RetrievalFilter, ...],
        dense_vector_name: str,
        sparse_vector_name: str,
        prefetch_limit: int,
        top_k: int,
    ) -> tuple[RetrievedChunk, ...]:
        if not self.hybrid_ready:
            raise ValueError(
                "QdrantVectorStoreClient.search_hybrid() requires a hybrid-ready client configuration."
            )

        query_filter = _build_qdrant_query_filter(
            qdrant_models=self.qdrant_models,
            filters=filters,
        )
        payload = build_hybrid_qdrant_query_payload(
            dense_vector=dense_vector,
            sparse_query=sparse_query,
            dense_vector_name=dense_vector_name,
            sparse_vector_name=sparse_vector_name,
            prefetch_limit=prefetch_limit,
            result_limit=top_k,
        )
        provider_payload = _build_provider_hybrid_query_payload(payload, self.qdrant_models)
        response = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            prefetch=provider_payload["prefetch"],
            query=provider_payload["query"],
            query_filter=query_filter,
            limit=provider_payload["limit"],
        )
        return _hits_to_retrieved_chunks(_extract_query_points_hits(response))


def _hit_to_retrieved_chunk(hit) -> RetrievedChunk:
    payload = _require_mapping(hit.payload, "Qdrant hit payload")
    metadata = _payload_to_chunk_metadata(payload)
    return RetrievedChunk(
        chunk_id=str(payload.get("chunk_id", hit.id)),
        content=str(_require_payload_value(payload, "content")),
        score=float(hit.score),
        metadata=metadata,
    )


def _hits_to_retrieved_chunks(hits) -> tuple[RetrievedChunk, ...]:
    return tuple(_hit_to_retrieved_chunk(hit) for hit in hits)


def _extract_query_points_hits(response):
    if hasattr(response, "points"):
        return response.points
    return response


def _build_provider_hybrid_query_payload(
    payload: Mapping[str, object],
    qdrant_models,
) -> dict[str, object]:
    if not all(hasattr(qdrant_models, name) for name in ("Prefetch", "FusionQuery", "Fusion")):
        return dict(payload)

    return {
        "prefetch": tuple(
            _build_provider_prefetch_item(item, qdrant_models)
            for item in payload.get("prefetch", ())
        ),
        "query": _build_provider_fusion_query(payload.get("query"), qdrant_models),
        "limit": payload.get("limit"),
    }


def _build_provider_prefetch_item(item: object, qdrant_models):
    prefetch_payload = _require_mapping(item, "Hybrid prefetch item")
    return qdrant_models.Prefetch(
        using=str(_require_payload_value(prefetch_payload, "using", label="Hybrid prefetch item")),
        query=_build_provider_prefetch_query(
            prefetch_payload.get("query"),
            qdrant_models,
        ),
        limit=int(_require_payload_value(prefetch_payload, "limit", label="Hybrid prefetch item")),
    )


def _build_provider_prefetch_query(query: object, qdrant_models):
    if isinstance(query, Sequence) and not isinstance(query, (str, bytes)):
        return [float(value) for value in query]

    query_payload = _require_mapping(query, "Hybrid prefetch query")
    if "indices" in query_payload and "values" in query_payload and hasattr(qdrant_models, "SparseVector"):
        return qdrant_models.SparseVector(
            indices=[int(index) for index in query_payload["indices"]],
            values=[float(value) for value in query_payload["values"]],
        )
    if "text" in query_payload and "model" in query_payload and hasattr(qdrant_models, "Document"):
        return qdrant_models.Document(
            text=str(query_payload["text"]),
            model=str(query_payload["model"]),
        )
    return dict(query_payload)


def _build_provider_fusion_query(query: object, qdrant_models):
    query_payload = _require_mapping(query, "Hybrid fusion query")
    if "rrf" in query_payload:
        return qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF)
    return dict(query_payload)


def _payload_to_chunk_metadata(payload: Mapping[str, object]) -> ChunkMetadata:
    return ChunkMetadata(
        source_file=str(_require_payload_value(payload, "source_file")),
        source_path=str(_require_payload_value(payload, "source_path")),
        heading_path=tuple(_coerce_string_sequence(payload.get("heading_path", ()))),
        doc_type=str(payload.get("doc_type", "unknown")),
        topic=str(payload.get("topic", "General")),
        disease_name=_coerce_optional_string(payload.get("disease_name")),
        semantic_graph=tuple(
            _payload_to_semantic_relation(item)
            for item in _coerce_relation_sequence(payload.get("semantic_graph", ()))
        ),
    )


def _payload_to_semantic_relation(item: object) -> SemanticRelation:
    relation_payload = _require_mapping(item, "semantic_graph item")
    return SemanticRelation(
        source=str(_require_payload_value(relation_payload, "source", label="semantic_graph item")),
        type=str(_require_payload_value(relation_payload, "type", label="semantic_graph item")),
        relation=str(
            _require_payload_value(relation_payload, "relation", label="semantic_graph item")
        ),
        target=str(_require_payload_value(relation_payload, "target", label="semantic_graph item")),
    )


def _coerce_string_sequence(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise ValueError(
            "Qdrant hit payload field 'heading_path' must be a sequence of strings, not a string."
        )
    if not _is_sequence_like(value):
        raise TypeError("Qdrant hit payload field 'heading_path' must be a sequence.")
    return tuple(str(item) for item in value)


def _coerce_relation_sequence(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not _is_sequence_like(value):
        raise TypeError(
            "Qdrant hit payload field 'semantic_graph' must be a sequence of relation mappings."
        )
    return tuple(value)


def _coerce_optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    return value


def _require_payload_value(
    payload: Mapping[str, object],
    key: str,
    label: str = "Qdrant hit payload",
) -> object:
    if key not in payload:
        raise ValueError(f"{label} is missing required field '{key}'.")
    return payload[key]


def _build_qdrant_query_filter(qdrant_models, filters: tuple[RetrievalFilter, ...]):
    filter_payload = build_qdrant_filter_payload(filters)
    if not filter_payload:
        return None

    grouped_filters: dict[str, list[str]] = {}
    ordered_fields: list[str] = []
    for field, value in filter_payload:
        if field not in grouped_filters:
            grouped_filters[field] = []
            ordered_fields.append(field)
        grouped_filters[field].append(value)

    must_conditions = []
    for field in ordered_fields:
        values = grouped_filters[field]
        if len(values) == 1:
            must_conditions.append(_build_field_condition(qdrant_models, field, values[0]))
            continue
        must_conditions.append(
            qdrant_models.Filter(
                should=[
                    _build_field_condition(qdrant_models, field, value)
                    for value in values
                ]
            )
        )

    return qdrant_models.Filter(must=must_conditions)


def _build_field_condition(qdrant_models, field: str, value: str):
    return qdrant_models.FieldCondition(
        key=field,
        match=qdrant_models.MatchValue(value=value),
    )


def _to_plain_payload_mapping(payload: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _to_plain_payload_value(value) for key, value in payload.items()}


def _build_point_vector_payload(point: VectorPoint):
    if point.dense_vector_name is None and point.sparse_vector_name is None:
        return point.vector

    vector_payload: dict[str, object] = {}
    if point.dense_vector_name is not None:
        vector_payload[point.dense_vector_name] = point.vector

    if point.sparse_vector_name is not None:
        if point.sparse_document is None:
            raise ValueError(
                f"VectorPoint '{point.point_id}' is missing sparse_document for sparse vector '{point.sparse_vector_name}'."
            )
        vector_payload[point.sparse_vector_name] = {
            "text": point.sparse_document.text,
            "model": point.sparse_document.model,
        }

    return vector_payload


def _to_plain_payload_value(value: object):
    if isinstance(value, Mapping):
        return {str(key): _to_plain_payload_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_plain_payload_value(item) for item in value]
    if isinstance(value, list):
        return [_to_plain_payload_value(item) for item in value]
    return value


def _is_sequence_like(value: object) -> bool:
    return isinstance(value, Sequence)


def _read_collection_config_mapping(collection_info, *paths: tuple[str, ...]) -> Mapping[str, object]:
    for path in paths:
        value = _lookup_path(collection_info, path)
        if isinstance(value, Mapping):
            return value
    return {}


def _lookup_path(value, path: tuple[str, ...]):
    current = value
    for part in path:
        if isinstance(current, Mapping):
            if part not in current:
                return None
            current = current[part]
            continue
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current
