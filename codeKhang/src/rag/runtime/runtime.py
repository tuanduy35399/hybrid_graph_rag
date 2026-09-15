from src.rag.integration.runtime import build_runtime_image_preprocessor
from src.rag.planning.runtime import build_runtime_planner_service
from src.rag.runtime.service import RAGRuntimeService


def build_runtime_service(
    preprocessor_factory=build_runtime_image_preprocessor,
    planner_factory=build_runtime_planner_service,
    runtime_service_cls=RAGRuntimeService,
    text_retrieval_service_factory=None,
    graph_chunk_provider_factory=None,
    text_answering_service_factory=None,
):
    preprocessor = preprocessor_factory()
    planner = planner_factory()
    if (
        text_retrieval_service_factory is None
        and graph_chunk_provider_factory is None
        and text_answering_service_factory is None
    ):
        return runtime_service_cls(preprocessor=preprocessor, planner=planner)

    if (
        text_retrieval_service_factory is None
    ) != (
        graph_chunk_provider_factory is None
    ):
        raise ValueError(
            "text_retrieval_service_factory and graph_chunk_provider_factory must be provided together"
        )

    if text_answering_service_factory is not None and (
        text_retrieval_service_factory is None
        or graph_chunk_provider_factory is None
    ):
        raise ValueError(
            "text_answering_service_factory requires text_retrieval_service_factory and graph_chunk_provider_factory"
        )

    text_retrieval_service = (
        text_retrieval_service_factory()
        if text_retrieval_service_factory is not None
        else None
    )
    graph_chunk_provider = (
        graph_chunk_provider_factory()
        if graph_chunk_provider_factory is not None
        else None
    )

    if text_answering_service_factory is None:
        return runtime_service_cls(
            preprocessor=preprocessor,
            planner=planner,
            text_retrieval_service=text_retrieval_service,
            graph_chunk_provider=graph_chunk_provider,
        )

    text_answering_service = text_answering_service_factory()
    return runtime_service_cls(
        preprocessor=preprocessor,
        planner=planner,
        text_retrieval_service=text_retrieval_service,
        graph_chunk_provider=graph_chunk_provider,
        text_answering_service=text_answering_service,
    )
