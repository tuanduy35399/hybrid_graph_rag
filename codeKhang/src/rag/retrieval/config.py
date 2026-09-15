from pathlib import Path

from src.rag.retrieval.provider_types import ProviderConfig

DEFAULT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def build_provider_config(env_map: dict[str, str]) -> ProviderConfig:
    return ProviderConfig(
        qdrant_api_key=_read_required_env(env_map, "QDRANT_API_KEY"),
        qdrant_url=_read_required_env(env_map, "QDRANT_URL"),
        qdrant_collection_name=_read_required_env(env_map, "QDRANT_COLLECTION_NAME"),
        qdrant_dense_vector_name=_read_optional_env(env_map, "QDRANT_DENSE_VECTOR_NAME"),
        qdrant_sparse_vector_name=_read_optional_env(env_map, "QDRANT_SPARSE_VECTOR_NAME"),
    )


def build_runtime_provider_config(env_path: Path | None = None) -> ProviderConfig:
    resolved_env_path = Path(env_path) if env_path is not None else DEFAULT_ENV_PATH
    env_map = _load_env_file(resolved_env_path)
    return build_provider_config(env_map)


def _read_required_env(env_map: dict[str, str], key: str) -> str:
    if key not in env_map:
        raise ValueError(f"Missing required environment variable: {key}")

    value = env_map[key].strip()
    if not value:
        raise ValueError(f"Blank environment variable: {key}")

    return value


def _read_optional_env(env_map: dict[str, str], key: str) -> str | None:
    value = env_map.get(key)
    if value is None:
        return None

    stripped = value.strip()
    return stripped or None


def _load_env_file(env_path: Path) -> dict[str, str]:
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
