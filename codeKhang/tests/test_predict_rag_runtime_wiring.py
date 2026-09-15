import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.integration.types import BranchType, PreparedBranchInput
from src.rag.normalization.types import NormalizationResult, QueryType
from src.rag.planning.types import PlannedQuery, QueryIntent
from src.rag.retrieval.types import RetrievalFilter, RetrievalRequest
from src.rag.runtime.types import ImageRuntimeResult


def test_extract_disease_hints_from_detection_keeps_unique_ranked_class_names():
    from src.rag.runtime.predict_bridge import extract_disease_hints_from_detection

    detections = [
        {"class_id": 2, "confidence": 0.91},
        {"class_id": 1, "confidence": 0.88},
        {"class_id": 2, "confidence": 0.70},
        {"class_id": 9, "confidence": 0.40},
        {"confidence": 0.33},
        {"class_id": "bad"},
        None,
    ]
    class_names = [
        "benh_a",
        "dom_la",
        "ri_sat",
    ]

    hints = extract_disease_hints_from_detection(
        detections=detections,
        class_names=class_names,
        max_hints=3,
    )

    assert hints == ("ri_sat", "dom_la")


def test_extract_disease_hints_from_detection_skips_unknown_and_malformed_rows():
    from src.rag.runtime.predict_bridge import extract_disease_hints_from_detection

    detections = [
        {"class_id": 99, "confidence": 0.95},
        {"class_id": -1, "confidence": 0.90},
        {"confidence": 0.80},
        {"class_id": "nan", "confidence": 0.70},
        [],
        {"class_id": 1, "confidence": 0.60},
        {"class_id": 0, "confidence": 0.50},
    ]
    class_names = ["ri_sat", "dom_la"]

    hints = extract_disease_hints_from_detection(
        detections=detections,
        class_names=class_names,
        max_hints=3,
    )

    assert hints == ("dom_la", "ri_sat")


def test_build_image_runtime_payload_returns_future_retriever_ready_dict():
    from src.rag.runtime.predict_bridge import build_image_runtime_payload

    prepared = PreparedBranchInput(
        branch=BranchType.IMAGE_BRANCH,
        normalization=NormalizationResult(
            normalized_query="la mai bi ri sat va xu ly sao",
            query_type=QueryType.MIXED_QUERY,
            normalization_notes=("rewritten_for_retrieval",),
        ),
    )
    planned = PlannedQuery(
        branch=BranchType.IMAGE_BRANCH,
        normalized_query="la mai bi ri sat va xu ly sao",
        query_type=QueryType.MIXED_QUERY,
        intent=QueryIntent.MIXED_REASONING,
        query_terms=("ri sat", "xu ly"),
        entity_hints=(),
        subqueries=("la mai bi ri sat", "xu ly sao"),
    )
    request = RetrievalRequest(
        query_text="la mai bi ri sat va xu ly sao",
        branch=BranchType.IMAGE_BRANCH,
        query_type=QueryType.MIXED_QUERY,
        intent=QueryIntent.MIXED_REASONING,
        query_terms=("ri sat", "xu ly"),
        entity_hints=(),
        topic_hints=("ri sat", "symptoms", "care"),
        filters=(RetrievalFilter(field="disease_name", value="ri sat"),),
        subqueries=("la mai bi ri sat", "xu ly sao"),
    )
    result = ImageRuntimeResult(
        prepared=prepared,
        planned=planned,
        retrieval_request=request,
        disease_hints=("ri sat",),
    )

    payload = build_image_runtime_payload(result)

    assert payload["branch"] == "image_branch"
    assert payload["normalized_query"] == "la mai bi ri sat va xu ly sao"
    assert payload["query_type"] == "mixed_query"
    assert payload["intent"] == "mixed_reasoning"
    assert payload["query_terms"] == ("ri sat", "xu ly")
    assert payload["topic_hints"] == ("ri sat", "symptoms", "care")
    assert payload["subqueries"] == ("la mai bi ri sat", "xu ly sao")
    assert payload["disease_hints"] == ("ri sat",)
    assert payload["filters"] == (("disease_name", "ri sat"),)


def test_predict_runtime_wiring_combined_helper_uses_same_runtime_service_across_image_seam():
    from src.rag.retrieval.request_builder import build_retrieval_request

    sys.modules.pop("predict", None)
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torchvision_stub = types.ModuleType("torchvision")
    torchvision_stub.ops = types.SimpleNamespace(nms=lambda *args, **kwargs: [])
    cv2_stub = types.ModuleType("cv2")
    matplotlib_stub = types.ModuleType("matplotlib")
    plt_stub = types.ModuleType("matplotlib.pyplot")
    matplotlib_stub.pyplot = plt_stub
    np_stub = types.ModuleType("numpy")
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *args, **kwargs: {}
    pil_stub = types.ModuleType("PIL")
    pil_image_stub = types.ModuleType("PIL.Image")
    pil_draw_stub = types.ModuleType("PIL.ImageDraw")
    pil_font_stub = types.ModuleType("PIL.ImageFont")
    pil_image_stub.open = lambda *args, **kwargs: None
    pil_draw_stub.Draw = lambda *args, **kwargs: None
    pil_font_stub.truetype = lambda *args, **kwargs: None
    pil_font_stub.load_default = lambda: None
    config_model_kwargs_stub = types.ModuleType("src.config.model_kwargs")
    config_model_kwargs_stub.build_model_kwargs = lambda config: {}
    data_transforms_stub = types.ModuleType("src.data.transforms")
    data_transforms_stub.ImageTransform = object
    data_transforms_stub.TextTransform = object
    models_multimodal_yolo_stub = types.ModuleType("src.models.multimodal_yolo")
    models_multimodal_yolo_stub.MultimodalYOLO = object

    stubbed_modules = {
        "cv2": cv2_stub,
        "matplotlib": matplotlib_stub,
        "matplotlib.pyplot": plt_stub,
        "numpy": np_stub,
        "torch": torch_stub,
        "torchvision": torchvision_stub,
        "yaml": yaml_stub,
        "PIL": pil_stub,
        "PIL.Image": pil_image_stub,
        "PIL.ImageDraw": pil_draw_stub,
        "PIL.ImageFont": pil_font_stub,
        "src.config.model_kwargs": config_model_kwargs_stub,
        "src.data.transforms": data_transforms_stub,
        "src.models.multimodal_yolo": models_multimodal_yolo_stub,
    }
    original_modules = {name: sys.modules.get(name) for name in stubbed_modules}
    sys.modules.update(stubbed_modules)

    try:
        from predict import build_image_rag_runtime_seam
    finally:
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    class StubPreprocessor:
        def __init__(self):
            self.calls = []
            self.prepared_results = []

        def prepare(self, raw_text, has_image=False):
            self.calls.append((raw_text, has_image))
            prepared = PreparedBranchInput(
                branch=BranchType.IMAGE_BRANCH,
                normalization=NormalizationResult(
                    normalized_query=f"normalized-pass-{len(self.calls)}",
                    query_type=QueryType.MIXED_QUERY,
                    normalization_notes=("rewritten_for_retrieval",),
                ),
            )
            self.prepared_results.append(prepared)
            return prepared

    class StubPlanner:
        def __init__(self):
            self.calls = []

        def plan(self, prepared):
            self.calls.append(prepared)
            return PlannedQuery(
                branch=BranchType.IMAGE_BRANCH,
                normalized_query=prepared.normalization.normalized_query,
                query_type=QueryType.MIXED_QUERY,
                intent=QueryIntent.MIXED_REASONING,
                query_terms=("ri sat", "xu ly"),
                entity_hints=(),
                subqueries=(prepared.normalization.normalized_query,),
            )

    class StubRuntimeService:
        def __init__(self):
            self.preprocessor = StubPreprocessor()
            self.planner = StubPlanner()

        def prepare_image_runtime_result(self, raw_text: str, disease_hints=()):
            prepared = self.preprocessor.prepare(raw_text, has_image=True)
            planned = self.planner.plan(prepared)
            request = build_retrieval_request(planned, disease_hints=disease_hints)
            return ImageRuntimeResult(
                prepared=prepared,
                planned=planned,
                retrieval_request=request,
                disease_hints=tuple(item.value for item in request.filters),
            )

    class FakeDetections:
        def __init__(self, rows):
            self._rows = rows

        def detach(self):
            return self

        def cpu(self):
            return self._rows

        def __len__(self):
            return len(self._rows)

    detections = FakeDetections(
        [
            ((0.0, 0.0, 10.0, 10.0), 0.95, 0),
            ((5.0, 5.0, 15.0, 15.0), 0.70, 1),
        ]
    )
    class_names = ["ri sat", "dom la"]

    runtime = StubRuntimeService()
    prepared_prompt, payload = build_image_rag_runtime_seam(
        raw_prompt="la mai bi gi va xu ly sao",
        runtime_service=runtime,
        det=detections,
        class_names=class_names,
    )

    assert runtime.preprocessor.calls == [
        ("la mai bi gi va xu ly sao", True)
    ]
    assert runtime.planner.calls == [prepared_prompt.prepared]
    assert prepared_prompt.raw_prompt == "la mai bi gi va xu ly sao"
    assert prepared_prompt.normalized_prompt == "normalized-pass-1"
    assert payload["normalized_query"] == "normalized-pass-1"
    assert payload["subqueries"] == ("normalized-pass-1",)
    assert payload["disease_hints"] == ("ri sat", "dom la")
    assert payload["filters"] == (
        ("disease_name", "ri sat"),
        ("disease_name", "dom la"),
    )
