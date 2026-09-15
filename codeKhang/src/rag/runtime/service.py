from src.rag.retrieval.request_builder import build_retrieval_request
from src.rag.runtime.types import ImageRuntimeResult, TextRuntimeResult


class RAGRuntimeService:
    def __init__(
        self,
        preprocessor,
        planner,
        text_retrieval_service=None,
        graph_chunk_provider=None,
        text_answering_service=None,
    ):
        self.preprocessor = preprocessor
        self.planner = planner
        self.text_retrieval_service = text_retrieval_service
        self.graph_chunk_provider = graph_chunk_provider
        self.text_answering_service = text_answering_service

    def answer_text_query(self, raw_text: str):
        if self.text_answering_service is None:
            raise ValueError(
                "text_answering_service must be provided for text answer synthesis"
            )

        runtime_result = self.prepare_text_runtime_result(raw_text)
        if runtime_result.retrieval_bundle is None:
            raise ValueError(
                "text runtime answer path requires retrieval_bundle before answer synthesis"
            )

        return self.text_answering_service.answer(runtime_result)

    def prepare_text_runtime_result(self, raw_text: str) -> TextRuntimeResult:
        prepared = self.preprocessor.prepare(raw_text, has_image=False)
        planned = self.planner.plan(prepared)
        retrieval_request = build_retrieval_request(planned)
        retrieval_bundle = None
        has_text_retrieval_service = self.text_retrieval_service is not None
        has_graph_chunk_provider = self.graph_chunk_provider is not None

        if has_text_retrieval_service != has_graph_chunk_provider:
            raise ValueError(
                "text_retrieval_service and graph_chunk_provider must be provided together"
            )

        if has_text_retrieval_service and has_graph_chunk_provider:
            graph_chunks = self.graph_chunk_provider.get_chunks(retrieval_request)
            retrieval_bundle = self.text_retrieval_service.retrieve(
                retrieval_request=retrieval_request,
                graph_chunks=graph_chunks,
            )
        return TextRuntimeResult(
            prepared=prepared,
            planned=planned,
            retrieval_request=retrieval_request,
            retrieval_bundle=retrieval_bundle,
        )

    def prepare_image_runtime_result(
        self,
        raw_text: str,
        disease_hints: tuple[str, ...] = (),
    ) -> ImageRuntimeResult:
        prepared = self.preprocessor.prepare(raw_text, has_image=True)
        planned = self.planner.plan(prepared)
        retrieval_request = build_retrieval_request(
            planned,
            disease_hints=disease_hints,
        )
        normalized_disease_hints = tuple(
            item.value
            for item in retrieval_request.filters
            if item.field == "disease_name"
        )
        return ImageRuntimeResult(
            prepared=prepared,
            planned=planned,
            retrieval_request=retrieval_request,
            disease_hints=normalized_disease_hints,
        )
