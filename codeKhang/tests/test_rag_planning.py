import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.integration.types import BranchType, PreparedBranchInput
from src.rag.normalization.types import NormalizationResult, QueryType


def test_query_intent_supports_disease_care_mixed_and_general():
    from src.rag.planning.types import QueryIntent

    assert QueryIntent.DISEASE_LOOKUP.value == "disease_lookup"
    assert QueryIntent.CARE_GUIDANCE.value == "care_guidance"
    assert QueryIntent.MIXED_REASONING.value == "mixed_reasoning"
    assert QueryIntent.GENERAL_LOOKUP.value == "general_lookup"


def test_entity_hint_preserves_kind_and_value():
    from src.rag.planning.types import EntityHint

    hint = EntityHint(kind="symptom", value="dom den")

    assert hint.kind == "symptom"
    assert hint.value == "dom den"


def test_planned_query_wraps_branch_normalization_and_planner_fields():
    from src.rag.planning.types import EntityHint, PlannedQuery, QueryIntent

    prepared = PreparedBranchInput(
        branch=BranchType.TEXT_ONLY_BRANCH,
        normalization=NormalizationResult(
            normalized_query="la mai bi dom den la benh gi",
            query_type=QueryType.DISEASE_QUERY,
            normalization_notes=("rewritten_for_retrieval",),
        ),
    )

    planned = PlannedQuery(
        branch=prepared.branch,
        normalized_query=prepared.normalization.normalized_query,
        query_type=prepared.normalization.query_type,
        intent=QueryIntent.DISEASE_LOOKUP,
        query_terms=("mai", "dom den", "benh"),
        entity_hints=(EntityHint(kind="symptom", value="dom den"),),
        subqueries=("la mai bi dom den la benh gi",),
    )

    assert planned.branch is BranchType.TEXT_ONLY_BRANCH
    assert planned.query_type is QueryType.DISEASE_QUERY
    assert planned.intent is QueryIntent.DISEASE_LOOKUP
    assert planned.entity_hints[0].kind == "symptom"


def test_build_planner_prompt_mentions_schema_and_safety_rules():
    from src.rag.planning.prompt import build_planner_prompt

    prompt = build_planner_prompt(
        normalized_query="la mai bi dom den la benh gi va xu ly sao",
        query_type=QueryType.MIXED_QUERY,
    )

    assert "intent" in prompt
    assert "query_terms" in prompt
    assert "entity_hints" in prompt
    assert "subqueries" in prompt
    assert "Do not answer the question." in prompt
    assert "Do not invent a disease conclusion." in prompt


def test_parse_planner_response_accepts_valid_mixed_query_payload():
    from src.rag.planning.parser import parse_planner_response
    from src.rag.planning.types import QueryIntent

    llm_text = """
    {
      "intent": "mixed_reasoning",
      "query_terms": ["mai", "dom den", "xu ly"],
      "entity_hints": [
        {"kind": "plant_name", "value": "mai"},
        {"kind": "symptom", "value": "dom den"}
      ],
      "subqueries": [
        "la mai bi dom den la benh gi",
        "xu ly sao"
      ]
    }
    """

    planned = parse_planner_response(
        branch=BranchType.TEXT_ONLY_BRANCH,
        normalized_query="la mai bi dom den la benh gi va xu ly sao",
        query_type=QueryType.MIXED_QUERY,
        llm_text=llm_text,
    )

    assert planned.intent is QueryIntent.MIXED_REASONING
    assert planned.subqueries == (
        "la mai bi dom den la benh gi",
        "xu ly sao",
    )
    assert any(
        hint.kind == "symptom" and hint.value == "dom den"
        for hint in planned.entity_hints
    )


def test_parse_planner_response_falls_back_when_output_is_invalid():
    from src.rag.planning.parser import parse_planner_response
    from src.rag.planning.types import QueryIntent

    planned = parse_planner_response(
        branch=BranchType.TEXT_ONLY_BRANCH,
        normalized_query="cach tuoi mai",
        query_type=QueryType.CARE_QUERY,
        llm_text='{"intent":"care_guidance","query_terms":"tuoi mai"}',
    )

    assert planned.intent is QueryIntent.CARE_GUIDANCE
    assert planned.query_terms == ()
    assert planned.entity_hints == ()
    assert planned.subqueries == ("cach tuoi mai",)


def test_parse_planner_response_falls_back_when_intent_conflicts_with_query_type():
    from src.rag.planning.parser import parse_planner_response
    from src.rag.planning.types import QueryIntent

    planned = parse_planner_response(
        branch=BranchType.TEXT_ONLY_BRANCH,
        normalized_query="cach tuoi mai",
        query_type=QueryType.CARE_QUERY,
        llm_text='{"intent":"disease_lookup","query_terms":["benh mai"],"entity_hints":[{"kind":"plant_name","value":"mai"}],"subqueries":["benh mai"]}',
    )

    assert planned.intent is QueryIntent.CARE_GUIDANCE
    assert planned.query_terms == ()
    assert planned.entity_hints == ()
    assert planned.subqueries == ("cach tuoi mai",)


def test_planner_llm_client_wraps_model_generate_content_result():
    from src.rag.planning.gemini_client import PlannerLLMClient

    class StubResponse:
        text = '{"intent":"care_guidance","query_terms":["tuoi mai"],"entity_hints":[],"subqueries":["cach tuoi mai"]}'

    class StubModel:
        def __init__(self):
            self.calls = []

        def generate_content(self, prompt: str):
            self.calls.append(prompt)
            return StubResponse()

    client = PlannerLLMClient(StubModel())
    output = client.generate("plan this")

    assert output == StubResponse.text
    assert client.model.calls == ["plan this"]


def test_plan_prepared_query_uses_client_output_when_valid():
    from src.rag.planning.service import PlannerService
    from src.rag.planning.types import QueryIntent

    class StubClient:
        def generate(self, prompt: str):
            return '{"intent":"mixed_reasoning","query_terms":["mai","dom den"],"entity_hints":[{"kind":"plant_name","value":"mai"}],"subqueries":["la mai bi dom den la benh gi","xu ly sao"]}'

    prepared = PreparedBranchInput(
        branch=BranchType.TEXT_ONLY_BRANCH,
        normalization=NormalizationResult(
            normalized_query="la mai bi dom den la benh gi va xu ly sao",
            query_type=QueryType.MIXED_QUERY,
            normalization_notes=("rewritten_for_retrieval",),
        ),
    )

    planned = PlannerService(StubClient()).plan(prepared)

    assert planned.intent is QueryIntent.MIXED_REASONING
    assert planned.subqueries == ("la mai bi dom den la benh gi", "xu ly sao")


def test_plan_prepared_query_falls_back_conservatively_when_client_raises():
    from src.rag.planning.service import PlannerService
    from src.rag.planning.types import QueryIntent

    class RaisingClient:
        def generate(self, prompt: str):
            raise RuntimeError("planner unavailable")

    prepared = PreparedBranchInput(
        branch=BranchType.TEXT_ONLY_BRANCH,
        normalization=NormalizationResult(
            normalized_query="cach tuoi mai",
            query_type=QueryType.CARE_QUERY,
            normalization_notes=("rewritten_for_retrieval",),
        ),
    )

    planned = PlannerService(RaisingClient()).plan(prepared)

    assert planned.intent is QueryIntent.CARE_GUIDANCE
    assert planned.query_terms == ()
    assert planned.entity_hints == ()
    assert planned.subqueries == ("cach tuoi mai",)


def test_build_gemini_planner_service_uses_env_and_factory_outputs():
    from src.rag.planning.runtime import build_gemini_planner_service

    class StubResponse:
        text = '{"intent":"care_guidance","query_terms":["tuoi mai"],"entity_hints":[],"subqueries":["cach tuoi mai"]}'

    class StubModel:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def generate_content(self, prompt: str):
            return StubResponse()

    class StubGenAI:
        def __init__(self):
            self.configured = []
            self.models = []

        def configure(self, api_key: str):
            self.configured.append(api_key)

        def GenerativeModel(self, model_name: str):
            model = StubModel(model_name)
            self.models.append(model)
            return model

    genai = StubGenAI()
    service = build_gemini_planner_service(
        env_map={"GEMINI_API_KEY": "secret-key"},
        genai_module=genai,
        model_name="gemini-2.5-flash",
    )

    assert genai.configured == ["secret-key"]
    assert len(genai.models) == 1
    assert genai.models[0].model_name == "gemini-2.5-flash"
    assert service.__class__.__name__ == "PlannerService"


def test_planning_package_exports_public_api():
    import src.rag.planning as planning
    from src.rag.planning.prompt import build_planner_prompt
    from src.rag.planning.runtime import (
        build_gemini_planner_service,
        build_runtime_planner_service,
    )
    from src.rag.planning.service import PlannerService
    from src.rag.planning.types import EntityHint, PlannedQuery, QueryIntent

    assert planning.EntityHint is EntityHint
    assert planning.PlannedQuery is PlannedQuery
    assert planning.PlannerService is PlannerService
    assert planning.QueryIntent is QueryIntent
    assert planning.build_gemini_planner_service is build_gemini_planner_service
    assert planning.build_planner_prompt is build_planner_prompt
    assert planning.build_runtime_planner_service is build_runtime_planner_service
    assert planning.__all__ == [
        "EntityHint",
        "PlannedQuery",
        "PlannerService",
        "QueryIntent",
        "build_gemini_planner_service",
        "build_planner_prompt",
        "build_runtime_planner_service",
    ]
