import json
import re
from collections.abc import Sequence

from .prompt import ALLOWED_NOTES
from .types import NormalizationResult, QueryType, fallback_normalization

_ALLOWED_NOTE_SET = frozenset(ALLOWED_NOTES)
_ALLOWED_QUERY_TYPES = frozenset(query_type.value for query_type in QueryType)
_DIAGNOSIS_CLAIM_PATTERNS = (
    re.compile(r"\bday la benh\s+(?!gi\b|nao\b)\w+", re.IGNORECASE),
    re.compile(r"\bla benh\s+(?!gi\b|nao\b)\w+", re.IGNORECASE),
    re.compile(r"\bbi benh\s+\w+", re.IGNORECASE),
    re.compile(r"\bmac benh\s+\w+", re.IGNORECASE),
    re.compile(r"\bco the la benh\s+(?!gi\b|nao\b)\w+", re.IGNORECASE),
)
_ADVICE_PATTERNS = (
    re.compile(r"\bban nen\b", re.IGNORECASE),
    re.compile(r"\bnen phun\b", re.IGNORECASE),
    re.compile(r"\bnen xit\b", re.IGNORECASE),
    re.compile(r"\bnen tuoi\b", re.IGNORECASE),
    re.compile(r"\bnen bon\b", re.IGNORECASE),
    re.compile(r"\bcan phai\b", re.IGNORECASE),
    re.compile(r"\bdung thuoc\b", re.IGNORECASE),
    re.compile(r"\bdung phan\b", re.IGNORECASE),
    re.compile(r"\bkhong nen\b", re.IGNORECASE),
    re.compile(r"\bgiam tuoi\b", re.IGNORECASE),
)


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.strip().split())


def _filter_notes(normalization_notes: Sequence[str]) -> tuple[str, ...]:
    filtered_notes = tuple(
        note.strip()
        for note in normalization_notes
        if note.strip() in _ALLOWED_NOTE_SET
    )
    if filtered_notes:
        return filtered_notes
    return ("no_major_change",)


def _matches_any_pattern(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


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
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None
    return payload


def _introduces_disallowed_semantics(raw_text: str, normalized_query: str) -> bool:
    raw_compact = _collapse_whitespace(raw_text).lower()
    normalized_compact = normalized_query.lower()

    if _matches_any_pattern(
        normalized_compact,
        _DIAGNOSIS_CLAIM_PATTERNS,
    ) and not _matches_any_pattern(raw_compact, _DIAGNOSIS_CLAIM_PATTERNS):
        return True

    if _matches_any_pattern(
        normalized_compact,
        _ADVICE_PATTERNS,
    ) and not _matches_any_pattern(raw_compact, _ADVICE_PATTERNS):
        return True

    return False


def parse_normalization_response(raw_text: str, llm_text: str) -> NormalizationResult:
    if not isinstance(llm_text, str):
        return fallback_normalization(raw_text)

    payload = _load_json_payload(llm_text)
    if payload is None:
        return fallback_normalization(raw_text)

    normalized_query = payload.get("normalized_query")
    query_type = payload.get("query_type")
    normalization_notes = payload.get("normalization_notes")

    if not isinstance(normalized_query, str):
        return fallback_normalization(raw_text)
    normalized_query = _collapse_whitespace(normalized_query)
    if not normalized_query:
        return fallback_normalization(raw_text)
    if _introduces_disallowed_semantics(raw_text, normalized_query):
        return fallback_normalization(raw_text)

    if not isinstance(query_type, str):
        return fallback_normalization(raw_text)
    query_type = query_type.strip()
    if query_type not in _ALLOWED_QUERY_TYPES:
        return fallback_normalization(raw_text)

    if (
        not isinstance(normalization_notes, Sequence)
        or isinstance(normalization_notes, str)
        or not all(isinstance(note, str) for note in normalization_notes)
    ):
        return fallback_normalization(raw_text)

    return NormalizationResult(
        normalized_query=normalized_query,
        query_type=QueryType(query_type),
        normalization_notes=_filter_notes(normalization_notes),
    )
