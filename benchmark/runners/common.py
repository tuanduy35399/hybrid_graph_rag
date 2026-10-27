"""Input isolation, output contract, and batch CLI shared by all systems."""

import argparse
import json
import re
import time
from pathlib import Path

from benchmark.adapters.citation_parser import parse_citation_payloads
from benchmark.evaluator.models import SystemOutput


DOC_CODE = re.compile(r"\b\d{1,4}(?:/\d{4})?/[A-ZĐ][\wĐđ]*(?:-[A-ZĐ][\wĐđ]*)*\b", re.I)
ARTICLE = re.compile(r"\bĐiều\s+(\d+[a-zđ]?)\b", re.I)


def load_questions(path):
    """Project untrusted dataset to exactly the two fields systems may see."""
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()

    rows = []
    if p.suffix.lower() == ".jsonl" or ("\n" in text and not text.startswith("[") and not text.startswith("{")):
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    else:
        data = json.loads(text)
        rows = data if isinstance(data, list) else data.get("items", data.get("questions", []))

    if not isinstance(rows, list):
        raise ValueError("dataset must contain a list of questions")
    questions = []
    seen = set()
    for row in rows:
        question_id = row.get("id") or row.get("case_id") or row.get("question_id")
        question = row.get("question") or row.get("user_input") or row.get("query")
        if not isinstance(question_id, str) or not question_id or question_id in seen:
            raise ValueError(f"invalid or duplicate question id: {question_id!r}")
        if not isinstance(question, str) or not question:
            raise ValueError(f"invalid question: {question_id!r}")
        seen.add(question_id)
        questions.append((question_id, question))
    return questions


def legal_code(value):
    match = DOC_CODE.search(value or "")
    return match.group(0) if match else None


def article_number(value):
    if isinstance(value, int) or (isinstance(value, str) and re.fullmatch(r"\d+[a-zđ]?", value, re.I)):
        return str(value)
    match = ARTICLE.search(str(value or ""))
    return match.group(1) if match else None


def context(rank, text, document_id=None, article=None, clause=None, point=None, score=None):
    return dict(rank=rank, document_id=document_id, article=article, clause=clause,
                point=point, text=text, score=score)


def run_question(question_id, question, system, query):
    start = time.perf_counter()
    answer, contexts, error = None, [], None
    try:
        answer, contexts = query(question)
        if not isinstance(answer, str):
            raise TypeError("system answer must be text")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        answer = None
    output = dict(
        question_id=question_id, system=system, answer=answer,
        retrieved_contexts=contexts, citations=parse_citation_payloads(answer),
        latency_ms=(time.perf_counter() - start) * 1000, error=error,
    )
    SystemOutput.from_dict(output)
    return output


def _save_checkpoint(records, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = out_path.with_suffix(out_path.suffix + ".tmp")
    temp_file.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_file.replace(out_path)


def main(system, build_query):
    parser = argparse.ArgumentParser(description=f"Run {system} on a benchmark dataset")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="Resume from existing output file if present (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Do not resume; start fresh and overwrite existing output file")
    parser.add_argument("--samples", "--limit", dest="samples", type=int, default=None,
                        help="Limit number of questions to run (useful for testing)")
    args = parser.parse_args()
    dataset, out = Path(args.dataset).resolve(), Path(args.out).resolve()
    questions = load_questions(dataset)
    if args.samples is not None and args.samples > 0:
        questions = questions[:args.samples]

    # Check for existing checkpoint
    completed_by_id = {}
    if args.resume and out.exists():
        try:
            raw_data = json.loads(out.read_text(encoding="utf-8"))
            records = raw_data if isinstance(raw_data, list) else raw_data.get("outputs", raw_data.get("results", []))
            for r in records:
                qid = r.get("question_id")
                if qid and qid not in completed_by_id:
                    completed_by_id[qid] = r
            if completed_by_id:
                print(f"[{system}] Checkpoint found: loaded {len(completed_by_id)} already completed questions from {out}")
        except Exception as exc:
            print(f"[{system}] Warning: Failed to read existing output file ({exc}), starting fresh.")
            completed_by_id = {}

    query = build_query()  # Startup and index loading are outside latency.
    outputs = []
    total = len(questions)
    try:
        for idx, (qid, question) in enumerate(questions, 1):
            if qid in completed_by_id:
                outputs.append(completed_by_id[qid])
                print(f"[{idx}/{total}] [{system}] (skip) Question {qid} already completed")
                continue

            print(f"[{idx}/{total}] [{system}] Running question {qid}...")
            res = run_question(qid, question, system, query)
            outputs.append(res)

            lat_str = f"{res['latency_ms']:.0f}ms" if res.get("latency_ms") is not None else "N/A"
            status = f"ERROR: {res['error']}" if res.get("error") else f"OK ({lat_str})"
            print(f"[{idx}/{total}] [{system}] Question {qid} -> {status}")

            _save_checkpoint(outputs, out)
    finally:
        close_query = getattr(query, "close", None)
        if callable(close_query):
            close_query()

    _save_checkpoint(outputs, out)
    print(f"[{system}] Completed: wrote {len(outputs)} outputs to {out}")
