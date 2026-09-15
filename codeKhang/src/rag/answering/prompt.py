def build_citation_label(chunk) -> str:
    return f"{chunk.metadata.source_file}#{chunk.chunk_id}"


def render_context_block(chunks) -> str:
    rendered_blocks = []
    for chunk in chunks:
        heading = " > ".join(chunk.metadata.heading_path)
        label = build_citation_label(chunk)
        rendered_blocks.append(
            "\n".join(
                (
                    f"[{label}]",
                    f"topic: {chunk.metadata.topic}",
                    f"heading: {heading}" if heading else "heading:",
                    chunk.content,
                )
            )
        )
    return "\n\n".join(rendered_blocks)


def build_grounded_answer_prompt(answer_context) -> str:
    return "\n".join(
        (
            "Ban la tro ly RAG cho cay mai vang.",
            "Chi duoc tra loi dua tren cac doan context da cung cap.",
            "Neu context khong du, hay noi ro la chua du thong tin.",
            "Moi khang dinh quan trong phai kem citation dang [source_file#chunk_id].",
            "",
            "QUESTION:",
            answer_context.retrieval_request.query_text,
            "",
            "CONTEXT:",
            answer_context.rendered_context,
        )
    )
