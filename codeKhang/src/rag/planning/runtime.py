from pathlib import Path

from src.rag.planning.gemini_client import PlannerLLMClient
from src.rag.planning.service import PlannerService


def read_gemini_api_key_from_env_map(env_map: dict[str, str]) -> str:
    return env_map["GEMINI_API_KEY"]


def load_env_file(env_path: Path) -> dict[str, str]:
    env_map: dict[str, str] = {}
    if not env_path.exists():
        return env_map

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        env_map[key.strip()] = value

    return env_map


def build_gemini_planner_service(
    env_map: dict[str, str],
    genai_module,
    model_name: str = "gemini-2.5-flash",
) -> PlannerService:
    api_key = read_gemini_api_key_from_env_map(env_map)
    genai_module.configure(api_key=api_key)
    model = genai_module.GenerativeModel(model_name)
    client = PlannerLLMClient(model)
    return PlannerService(client)


def build_runtime_planner_service() -> PlannerService:
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError(
            "google-generativeai is required for planner normalization."
        ) from exc

    env_map = load_env_file(Path("src/.env"))
    return build_gemini_planner_service(env_map=env_map, genai_module=genai)
