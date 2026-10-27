"""Adapter for the same Retriever/AnswerGenerator pair used by HybridRAG API."""

import os
import sys
from pathlib import Path

from .common import article_number, context, legal_code, main


def adapt_contexts(nodes):
    out = []
    for rank, item in enumerate(nodes, 1):
        node = getattr(item, "node", item)
        meta = getattr(node, "metadata", {}) or {}
        text = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", None)
        
        # Document identifier: prioritize source / file_name / legal doc_code
        doc_id = meta.get("source") or meta.get("file_name") or legal_code(meta.get("doc_code"))
        if not doc_id and meta.get("file_path"):
            doc_id = Path(meta["file_path"]).name
        if not doc_id:
            doc_id = None

        # Section / article identifier (only when valid article exists)
        article = article_number(meta.get("article"))

        out.append(context(rank, text, doc_id, article, score=getattr(item, "score", None)))
    return out


def build_query():
    candidates = [
        Path(__file__).resolve().parents[2] / "hybirdRAG_nckh",
        Path(__file__).resolve().parents[2] / "hybridRAG",
        Path(__file__).resolve().parents[2] / "hybridRAG_paper",
    ]
    root = None
    for c in candidates:
        if c.exists() and (c / "api" / "main.py").exists():
            root = c
            break
    if root is None:
        raise FileNotFoundError(f"Could not locate hybridRAG directory in {[str(c) for c in candidates]}")

    sys.path.insert(0, str(root))
    previous = Path.cwd()
    try:
        os.chdir(root)  # Existing API resolves BM25/Chroma paths against cwd.
        from api.main import generator, retriever
    finally:
        os.chdir(previous)

    def query(question):
        nodes = retriever.retrieve(question)
        return generator.generate(question=question, contexts=nodes), adapt_contexts(nodes)

    return query


if __name__ == "__main__":
    main("hybridrag", build_query)
