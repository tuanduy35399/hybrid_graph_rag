from src.rag.integration.types import PreparedBranchInput
from src.rag.planning.parser import fallback_planned_query, parse_planner_response
from src.rag.planning.prompt import build_planner_prompt


class PlannerService:
    def __init__(self, client):
        self.client = client

    def plan(self, prepared: PreparedBranchInput):
        prompt = build_planner_prompt(
            normalized_query=prepared.normalization.normalized_query,
            query_type=prepared.normalization.query_type,
        )

        try:
            llm_text = self.client.generate(prompt)
        except Exception:
            return fallback_planned_query(
                branch=prepared.branch,
                normalized_query=prepared.normalization.normalized_query,
                query_type=prepared.normalization.query_type,
            )

        if not isinstance(llm_text, str):
            return fallback_planned_query(
                branch=prepared.branch,
                normalized_query=prepared.normalization.normalized_query,
                query_type=prepared.normalization.query_type,
            )

        return parse_planner_response(
            branch=prepared.branch,
            normalized_query=prepared.normalization.normalized_query,
            query_type=prepared.normalization.query_type,
            llm_text=llm_text,
        )
