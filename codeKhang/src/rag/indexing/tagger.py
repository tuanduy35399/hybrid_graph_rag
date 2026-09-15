from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Protocol, Sequence

from .models import ChunkMetadata, SemanticRelation


class LLMClient(Protocol):
    def generate(self, prompt: str) -> object:
        ...


class GeminiTaggerClient:
    def __init__(self, model):
        self.model = model

    def generate(self, prompt: str) -> object:
        response = self.model.generate_content(prompt)
        return getattr(response, "text", None)


class TaggerService:
    def __init__(
        self,
        client: LLMClient | None,
        fallback_reason: str | None = None,
        debug_chunk_limit: int = 0,
    ):
        self.client = client
        self.fallback_reason = fallback_reason
        self.debug_chunk_limit = max(0, int(debug_chunk_limit))
        self._tag_calls = 0

    def tag(
        self,
        chunk_content: str,
        heading_path: Sequence[str],
        source_file: str,
        source_path: str,
    ) -> ChunkMetadata:
        self._tag_calls += 1
        heading_tuple = _normalize_heading_path(heading_path)
        prompt = build_tagger_prompt(
            chunk_content=chunk_content,
            heading_path=heading_tuple,
            source_file=source_file,
            source_path=source_path,
        )

        if self.client is None:
            return fallback_chunk_metadata(
                source_file=source_file,
                source_path=source_path,
                heading_path=heading_tuple,
            )

        try:
            llm_text = self.client.generate(prompt)
        except Exception as exc:
            self._debug_error(
                source_file=source_file,
                heading_path=heading_tuple,
                error=exc,
            )
            return fallback_chunk_metadata(
                source_file=source_file,
                source_path=source_path,
                heading_path=heading_tuple,
            )

        if not isinstance(llm_text, str):
            llm_text = getattr(llm_text, "text", None)

        if not isinstance(llm_text, str):
            self._debug_message(
                source_file=source_file,
                heading_path=heading_tuple,
                body="raw response missing text payload; using fallback metadata.",
            )
            return fallback_chunk_metadata(
                source_file=source_file,
                source_path=source_path,
                heading_path=heading_tuple,
            )

        self._debug_message(
            source_file=source_file,
            heading_path=heading_tuple,
            body=f"raw response:\n{llm_text}",
        )

        parsed = _try_parse_tagger_response(
            chunk_content=chunk_content,
            heading_path=heading_tuple,
            source_file=source_file,
            source_path=source_path,
            llm_text=llm_text,
        )
        if parsed is None:
            self._debug_message(
                source_file=source_file,
                heading_path=heading_tuple,
                body="parse_ok=False; response did not match expected JSON schema. Using fallback metadata.",
            )
            return fallback_chunk_metadata(
                source_file=source_file,
                source_path=source_path,
                heading_path=heading_tuple,
            )

        self._debug_message(
            source_file=source_file,
            heading_path=heading_tuple,
            body=(
                "parse_ok=True; "
                f"doc_type={parsed.doc_type!r}, topic={parsed.topic!r}, disease_name={parsed.disease_name!r}, "
                f"semantic_graph_count={len(parsed.semantic_graph)}"
            ),
        )
        return parsed

    def _should_debug(self) -> bool:
        return self.debug_chunk_limit > 0 and self._tag_calls <= self.debug_chunk_limit

    def _debug_message(
        self,
        *,
        source_file: str,
        heading_path: Sequence[str],
        body: str,
    ) -> None:
        if not self._should_debug():
            return
        heading_text = " > ".join(heading_path) if heading_path else "General"
        print(
            f"[rag-indexing][tagger-debug][chunk {self._tag_calls}] "
            f"source={source_file} heading={heading_text}\n{body}"
        )

    def _debug_error(
        self,
        *,
        source_file: str,
        heading_path: Sequence[str],
        error: Exception,
    ) -> None:
        self._debug_message(
            source_file=source_file,
            heading_path=heading_path,
            body=f"error={type(error).__name__}: {error}",
        )


def build_gemini_tagger_service(
    env_map: dict[str, str] | None = None,
    genai_module=None,
    model_name: str = "gemini-2.5-flash",
) -> TaggerService:
    if env_map is None:
        env_map = _merge_env_sources(os.environ, _load_env_file(Path("src/.env")))
    api_key = env_map.get("GEMINI_API_KEY")
    if not api_key:
        return TaggerService(None, fallback_reason="Gemini tagger unavailable; GEMINI_API_KEY is missing.")

    if genai_module is None:
        try:
            import google.generativeai as genai_module  # type: ignore
        except Exception:
            return TaggerService(
                None,
                fallback_reason="Gemini tagger unavailable; google.generativeai could not be imported.",
            )

    try:
        genai_module.configure(api_key=api_key)
        model = genai_module.GenerativeModel(model_name)
    except Exception:
        return TaggerService(
            None,
            fallback_reason="Gemini tagger unavailable; Gemini client initialization failed.",
        )

    return TaggerService(
        GeminiTaggerClient(model),
        debug_chunk_limit=_read_debug_chunk_limit(env_map),
    )


def _merge_env_sources(primary_env: dict[str, str], fallback_env: dict[str, str]) -> dict[str, str]:
    merged = dict(fallback_env)
    merged.update(primary_env)
    return merged


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


def _read_debug_chunk_limit(env_map: dict[str, str]) -> int:
    raw_value = env_map.get("RAG_TAGGER_DEBUG_CHUNKS")
    if raw_value is None:
        return 0
    try:
        return max(0, int(raw_value.strip()))
    except ValueError:
        return 0


def build_tagger_prompt(
    chunk_content: str,
    heading_path: Sequence[str],
    source_file: str,
    source_path: str,
) -> str:
    heading_text = " > ".join(heading_path) if heading_path else "General"
    return (
        "Extract structured metadata for a knowledge-base markdown chunk.\n"
        f"Source file: {source_file}\n"
        f"Source path: {source_path}\n"
        f"Heading path: {heading_text}\n"
        "Return JSON with keys: doc_type, topic, disease_name, semantic_graph.\n"
        "semantic_graph must be a list of relation objects with source, type, relation, target.\n"
        "Chunk:\n"
        f"{chunk_content}\n"
    )


def parse_tagger_response(
    chunk_content: str,
    heading_path: Sequence[str],
    source_file: str,
    source_path: str,
    llm_text: str,
) -> ChunkMetadata:
    parsed = _try_parse_tagger_response(
        chunk_content=chunk_content,
        heading_path=heading_path,
        source_file=source_file,
        source_path=source_path,
        llm_text=llm_text,
    )
    if parsed is None:
        return fallback_chunk_metadata(source_file, source_path, _normalize_heading_path(heading_path))
    return parsed


def _try_parse_tagger_response(
    chunk_content: str,
    heading_path: Sequence[str],
    source_file: str,
    source_path: str,
    llm_text: str,
) -> ChunkMetadata | None:
    heading_tuple = _normalize_heading_path(heading_path)
    payload = _load_json_payload(llm_text)
    if payload is None:
        return None

    doc_type = payload.get("doc_type")
    topic = payload.get("topic")
    disease_name = payload.get("disease_name")
    semantic_graph = payload.get("semantic_graph")

    if not isinstance(doc_type, str) or not isinstance(topic, str):
        return None
    if disease_name is not None and not isinstance(disease_name, str):
        return None
    if not isinstance(semantic_graph, list):
        return None

    parsed_relations = []
    for item in semantic_graph:
        if not isinstance(item, dict):
            return None

        source = item.get("source")
        rel_type = item.get("type")
        relation = item.get("relation")
        target = item.get("target")
        if not all(isinstance(value, str) for value in (source, rel_type, relation, target)):
            return None

        parsed_relations.append(
            SemanticRelation(
                source=" ".join(source.strip().split()),
                type=" ".join(rel_type.strip().split()),
                relation=" ".join(relation.strip().split()),
                target=" ".join(target.strip().split()),
            )
        )

    return ChunkMetadata(
        source_file=source_file,
        source_path=source_path,
        heading_path=heading_tuple,
        doc_type=doc_type.strip(),
        topic=topic.strip(),
        disease_name=disease_name.strip() if isinstance(disease_name, str) else None,
        semantic_graph=tuple(parsed_relations),
    )


def fallback_chunk_metadata(
    source_file: str,
    source_path: str,
    heading_path: Sequence[str],
) -> ChunkMetadata:
    heading_tuple = _normalize_heading_path(heading_path)
    return ChunkMetadata(
        source_file=source_file,
        source_path=source_path,
        heading_path=heading_tuple,
        doc_type="unknown",
        topic=heading_tuple[-1] if heading_tuple else "General",
        disease_name=None,
        semantic_graph=(),
    )


def _normalize_heading_path(heading_path: Sequence[str]) -> tuple[str, ...]:
    normalized = []
    for part in heading_path:
        cleaned = " ".join(str(part).strip().split())
        if cleaned:
            normalized.append(cleaned)
    return tuple(normalized)


def _load_json_payload(llm_text: str) -> dict[str, object] | None:
    cleaned = llm_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    try:
        payload = json.loads(cleaned[start : end + 1])
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    return payload
