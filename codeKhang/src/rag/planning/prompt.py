from src.rag.normalization.types import QueryType


def build_planner_prompt(normalized_query: str, query_type: QueryType) -> str:
    return (
        "You are a retrieval planner for a Vietnamese plant-care RAG system.\n"
        "Return JSON only with this schema:\n"
        "{\n"
        '  "intent": "disease_lookup | care_guidance | mixed_reasoning | general_lookup",\n'
        '  "query_terms": ["..."],\n'
        '  "entity_hints": [{"kind": "...", "value": "..."}],\n'
        '  "subqueries": ["..."]\n'
        "}\n"
        "Rules:\n"
        "- Do not answer the question.\n"
        "- Do not invent a disease conclusion.\n"
        "- Do not add treatment advice.\n"
        "- Preserve the user meaning in retrieval-friendly form.\n"
        f"- query_type hint: {query_type.value}\n"
        f"- normalized_query: {normalized_query}\n"
    )
