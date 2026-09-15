from __future__ import annotations

import re
from typing import Any

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")


def chunk_markdown(content: str, max_length_tokens: int = 1000) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current_heading_path: list[str] = []
    body_lines: list[str] = []

    def token_count(text: str) -> int:
        return len(text.split())

    def split_overlong_text(text: str) -> list[str]:
        cleaned = text.strip()
        if not cleaned:
            return []
        if max_length_tokens <= 0 or token_count(cleaned) <= max_length_tokens:
            return [cleaned]

        paragraphs = [part.strip() for part in cleaned.split("\n\n") if part.strip()]
        if len(paragraphs) > 1:
            pieces: list[str] = []
            for paragraph in paragraphs:
                pieces.extend(split_overlong_text(paragraph))
            return pieces

        lines = [part.strip() for part in cleaned.splitlines() if part.strip()]
        if len(lines) > 1:
            pieces = []
            for line in lines:
                pieces.extend(split_overlong_text(line))
            return pieces

        tokens = cleaned.split()
        return [
            " ".join(tokens[index : index + max_length_tokens])
            for index in range(0, len(tokens), max_length_tokens)
        ]

    def flush_body() -> None:
        if not body_lines:
            return
        body = "\n".join(body_lines).strip()
        body_lines.clear()
        if not body:
            return

        heading_path = current_heading_path[:] if current_heading_path else ["General"]
        for subchunk in split_overlong_text(body):
            chunks.append(
                {
                    "heading_path": heading_path[:],
                    "content": subchunk,
                }
            )

    for raw_line in content.splitlines():
        match = _HEADING_RE.match(raw_line)
        if match:
            flush_body()
            level = len(match.group(1))
            title = match.group(2).strip()
            if not title:
                current_heading_path = current_heading_path[: max(level - 1, 0)]
            elif level <= 1:
                current_heading_path = [title]
            else:
                current_heading_path = current_heading_path[: level - 1] + [title]
            continue

        body_lines.append(raw_line)

    flush_body()
    return chunks
