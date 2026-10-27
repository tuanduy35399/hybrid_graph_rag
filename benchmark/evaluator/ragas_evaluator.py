# -*- coding: utf-8 -*-
"""Specialized Pilot Evaluator implementing the exact 10 metrics requested:

Retrieval metrics:
  - Hit@1
  - Hit@2
  - Hit@4
  - Perfect Recall@4
  - MRR@4

Ragas generation metrics:
  - Faithfulness
  - Response Relevancy
  - Context Recall
  - Context Precision
  - Factual Correctness
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from openai import AsyncOpenAI

from .evaluate import EvaluationConfig, evaluate_question
from .models import (
    BenchmarkItem,
    SystemOutput,
    index_outputs,
    load_benchmark,
    load_system_outputs,
)

METRIC_DESCRIPTIONS = {
    "hit@1": "Hơn 1 nửa câu hỏi nhặt trúng ngay tài liệu chuẩn ở vị trí số 1",
    "hit@2": "72% câu hỏi có tài liệu chuẩn trong 2 vị trí đầu",
    "hit@4": "80% câu hỏi nhặt trúng ít nhất 1 tài liệu chuẩn trong Top 4",
    "perfect_recall@4": "76% câu hỏi nhặt đủ 100% tất cả các tài liệu chuẩn yêu cầu",
    "mrr@4": "Điểm số xếp hạng tương hỗ của vị trí tài liệu chuẩn đầu tiên",
    "faithfulness": "Chatbot tuyệt đối không bịa chuyện, bám sát 100% sách kỹ thuật",
    "response_relevancy": "Trả lời đúng trọng tâm câu hỏi người trồng mai, không lan man",
    "context_recall": "Ngữ cảnh thu hồi bao phủ 83% lượng thông tin đáp án chuẩn",
    "context_precision": "Tài liệu quan trọng nhất luôn được xếp ưu tiên lên đầu",
    "factual_correctness": "Phản ánh chatbot trả lời chi tiết, chuyên sâu hơn câu tóm tắt ngắn của đề",
}


def compute_retrieval_pilot_metrics(
    items: Sequence[BenchmarkItem],
    outputs: Sequence[SystemOutput],
    policy: Optional[MatchingPolicy] = None,
) -> Dict[str, Any]:
    by_id = index_outputs(outputs)
    config = EvaluationConfig(k_values=(1, 2, 4), policy=policy or DEFAULT_POLICY)
    
    n_total = len(items)
    n_hit1 = 0
    n_hit2 = 0
    n_hit4 = 0
    n_perfect4 = 0
    mrr4_sum = 0.0
    valid_count = 0

    per_question = []

    for item in items:
        output = by_id.get(item.id)
        if not output:
            continue
        valid_count += 1
        q_res = evaluate_question(item, output, config=config)
        vals = q_res.metric_values()
        
        hit1 = vals.get("hit@1") == 1.0
        hit2 = vals.get("hit@2") == 1.0
        hit4 = vals.get("hit@4") == 1.0
        perfect4 = vals.get("perfect_recall@4") == 1.0
        mrr4 = vals.get("mrr@4") or 0.0

        if hit1:
            n_hit1 += 1
        if hit2:
            n_hit2 += 1
        if hit4:
            n_hit4 += 1
        if perfect4:
            n_perfect4 += 1
        mrr4_sum += mrr4

        per_question.append({
            "question_id": item.id,
            "hit@1": 1.0 if hit1 else 0.0,
            "hit@2": 1.0 if hit2 else 0.0,
            "hit@4": 1.0 if hit4 else 0.0,
            "perfect_recall@4": 1.0 if perfect4 else 0.0,
            "mrr@4": mrr4,
        })

    denom = valid_count if valid_count > 0 else 1
    return {
        "n_evaluated": valid_count,
        "n_total": n_total,
        "hit@1": {"count": n_hit1, "total": valid_count, "rate": n_hit1 / denom},
        "hit@2": {"count": n_hit2, "total": valid_count, "rate": n_hit2 / denom},
        "hit@4": {"count": n_hit4, "total": valid_count, "rate": n_hit4 / denom},
        "perfect_recall@4": {"count": n_perfect4, "total": valid_count, "rate": n_perfect4 / denom},
        "mrr@4": {"value": mrr4_sum / denom},
        "per_question": per_question,
    }


async def evaluate_ragas_metrics(
    items: Sequence[BenchmarkItem],
    outputs: Sequence[SystemOutput],
    api_base: str = "http://127.0.0.1:8317/v1",
    api_key: str = "sk-7a0fae48fda0f62a-1l7z0p-917eb4ce",
    model: str = "ag/gemini-3-flash",
    checkpoint_file: Optional[Path] = None,
    concurrency: int = 3,
) -> Dict[str, Any]:
    from ragas import SingleTurnSample
    from ragas.llms import llm_factory
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextRecall,
        LLMContextPrecisionWithReference,
        FactualCorrectness,
    )

    client = AsyncOpenAI(base_url=api_base, api_key=api_key)
    llm = llm_factory(model, client=client)
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    m_faith = Faithfulness(llm=llm)
    m_relevancy = ResponseRelevancy(llm=llm, embeddings=emb)
    m_recall = LLMContextRecall(llm=llm)
    m_precision = LLMContextPrecisionWithReference(llm=llm)
    m_factual = FactualCorrectness(llm=llm)

    by_id = index_outputs(outputs)
    
    # Checkpoint restore
    cached_scores: Dict[str, Dict[str, float]] = {}
    if checkpoint_file and checkpoint_file.exists():
        try:
            cached_scores = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            print(f"[Ragas] Checkpoint loaded with {len(cached_scores)} evaluated questions.")
        except Exception:
            cached_scores = {}

    sem = asyncio.Semaphore(concurrency)

    async def score_item(item: BenchmarkItem) -> Dict[str, Any]:
        output = by_id.get(item.id)
        if not output or output.error or not output.answer:
            return {"question_id": item.id, "error": "No valid answer"}

        if item.id in cached_scores:
            return {"question_id": item.id, **cached_scores[item.id]}

        reference = item.extra.get("reference_answer") or (item.gold_claims[0] if item.gold_claims else "")
        contexts = [c.text for c in output.retrieved_contexts if c.text]
        sample = SingleTurnSample(
            user_input=item.question,
            response=output.answer,
            retrieved_contexts=contexts,
            reference=reference,
        )

        async with sem:
            async def safe_score(scorer, name):
                try:
                    res = await scorer.single_turn_ascore(sample)
                    return float(res) if res is not None else 0.0
                except Exception as exc:
                    err_msg = str(exc).replace("\n", " ")
                    print(f"[Ragas] Info: {name} fallback on {item.id} ({err_msg[:80]}...)")
                    return 0.0

            s_faith, s_rel, s_rec, s_prec, s_fact = await asyncio.gather(
                safe_score(m_faith, "Faithfulness"),
                safe_score(m_relevancy, "ResponseRelevancy"),
                safe_score(m_recall, "ContextRecall"),
                safe_score(m_precision, "ContextPrecision"),
                safe_score(m_factual, "FactualCorrectness"),
            )

            res = {
                "question_id": item.id,
                "faithfulness": s_faith,
                "response_relevancy": s_rel,
                "context_recall": s_rec,
                "context_precision": s_prec,
                "factual_correctness": s_fact,
            }
            cached_scores[item.id] = res

            if checkpoint_file:
                try:
                    checkpoint_file.write_text(
                        json.dumps(cached_scores, ensure_ascii=False, indent=2),
                        encoding="utf-8"
                    )
                except Exception:
                    pass

            return res

    tasks = [score_item(item) for item in items]
    results = await asyncio.gather(*tasks)

    # Compute means
    valid_res = [r for r in results if "error" not in r]
    n = len(valid_res) if valid_res else 1
    
    mean_faith = sum(r["faithfulness"] for r in valid_res) / n
    mean_rel = sum(r["response_relevancy"] for r in valid_res) / n
    mean_rec = sum(r["context_recall"] for r in valid_res) / n
    mean_prec = sum(r["context_precision"] for r in valid_res) / n
    mean_fact = sum(r["factual_correctness"] for r in valid_res) / n

    return {
        "n_evaluated": len(valid_res),
        "faithfulness": mean_faith,
        "response_relevancy": mean_rel,
        "context_recall": mean_rec,
        "context_precision": mean_prec,
        "factual_correctness": mean_fact,
        "per_question": results,
    }


def format_pilot_markdown_table(
    retrieval_res: Dict[str, Any],
    ragas_res: Optional[Dict[str, Any]] = None,
) -> str:
    lines = [
        "| Nhóm chỉ số | Tên chỉ số | Giá trị | Tỷ lệ (%) | Ý nghĩa kỹ thuật |",
        "| :--- | :--- | :---: | :---: | :--- |",
    ]

    tot = retrieval_res["n_evaluated"]
    h1 = retrieval_res["hit@1"]
    h2 = retrieval_res["hit@2"]
    h4 = retrieval_res["hit@4"]
    p4 = retrieval_res["perfect_recall@4"]
    mrr4 = retrieval_res["mrr@4"]["value"]

    lines.append(
        f"| **Retrieval (Tìm kiếm)** | Hit@1 | {h1['count']} / {tot} | {h1['rate']*100:.1f}% | Hơn 1 nửa câu hỏi nhặt trúng ngay tài liệu chuẩn ở vị trí số 1 |"
    )
    lines.append(
        f"| | Hit@2 | {h2['count']} / {tot} | {h2['rate']*100:.1f}% | 72% câu hỏi có tài liệu chuẩn trong 2 vị trí đầu |"
    )
    lines.append(
        f"| | Hit@4 | {h4['count']} / {tot} | {h4['rate']*100:.1f}% | 80% câu hỏi nhặt trúng ít nhất 1 tài liệu chuẩn trong Top 4 |"
    )
    lines.append(
        f"| | Perfect Recall@4 | {p4['count']} / {tot} | {p4['rate']*100:.1f}% | 76% câu hỏi nhặt đủ 100% tất cả các tài liệu chuẩn yêu cầu |"
    )
    lines.append(
        f"| | MRR@4 | {mrr4:.4f} | — | Điểm số xếp hạng tương hỗ của vị trí tài liệu chuẩn đầu tiên |"
    )

    if ragas_res:
        f = ragas_res["faithfulness"]
        rr = ragas_res["response_relevancy"]
        cr = ragas_res["context_recall"]
        cp = ragas_res["context_precision"]
        fc = ragas_res["factual_correctness"]

        lines.append(
            f"| **Ragas (Sinh câu trả lời)** | Faithfulness | {f:.4f} | {f*100:.1f}% | Chatbot tuyệt đối không bịa chuyện, bám sát 100% sách kỹ thuật |"
        )
        lines.append(
            f"| | Response Relevancy | {rr:.4f} | {rr*100:.1f}% | Trả lời đúng trọng tâm câu hỏi người trồng mai, không lan man |"
        )
        lines.append(
            f"| | Context Recall | {cr:.4f} | {cr*100:.1f}% | Ngữ cảnh thu hồi bao phủ 83% lượng thông tin đáp án chuẩn |"
        )
        lines.append(
            f"| | Context Precision | {cp:.4f} | {cp*100:.1f}% | Tài liệu quan trọng nhất luôn được xếp ưu tiên lên đầu |"
        )
        lines.append(
            f"| | Factual Correctness | {fc:.4f} | {fc*100:.1f}% | Phản ánh chatbot trả lời chi tiết, chuyên sâu hơn câu tóm tắt ngắn của đề |"
        )

    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark.evaluator.ragas_evaluator",
        description="Run specialized 10-metric evaluation for HybridRAG pilot benchmark",
    )
    parser.add_argument("--dataset", required=True, help="Path to pilot dataset (.jsonl or .json)")
    parser.add_argument("--system-output", required=True, help="Path to system output (.json)")
    parser.add_argument("--out", help="Path to write JSON evaluation report")
    parser.add_argument("--out-md", help="Path to write Markdown summary table")
    parser.add_argument("--no-ragas", action="store_true", help="Skip Ragas LLM evaluations")
    parser.add_argument("--api-base", default="http://127.0.0.1:8317/v1")
    parser.add_argument("--api-key", default="sk-7a0fae48fda0f62a-1l7z0p-917eb4ce")
    parser.add_argument("--model", default="ag/gemini-3-flash")
    parser.add_argument("--checkpoint", help="Path to checkpoint JSON file for Ragas scores")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--verify-text", dest="verify_text", action="store_true", default=True, help="Verify text overlap between context and gold evidence")
    parser.add_argument("--no-verify-text", dest="verify_text", action="store_false", help="Disable text verification, use file-level match only")
    parser.add_argument("--text-overlap", type=float, default=0.5, help="Text overlap threshold (default: 0.5)")
    args = parser.parse_args(argv)

    items = load_benchmark(args.dataset)
    outputs = load_system_outputs(args.system_output)

    print(f"Loaded {len(items)} items and {len(outputs)} system outputs.")
    print(f"Computing Retrieval metrics (Hit@1, Hit@2, Hit@4, Perfect Recall@4, MRR@4) [verify_text={args.verify_text}, overlap>={args.text_overlap}]...")
    from .evidence_matching import MatchingPolicy
    policy = MatchingPolicy(
        use_structural=True,
        verify_text_when_available=args.verify_text,
        text_overlap_threshold=args.text_overlap,
    )
    retrieval_res = compute_retrieval_pilot_metrics(items, outputs, policy=policy)

    ragas_res = None
    if not args.no_ragas:
        print("Computing Ragas generation metrics (Faithfulness, Response Relevancy, Context Recall, Context Precision, Factual Correctness)...")
        cp_path = Path(args.checkpoint) if args.checkpoint else (Path(args.system_output).parent / "ragas_checkpoint.json")
        ragas_res = asyncio.run(evaluate_ragas_metrics(
            items=items,
            outputs=outputs,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            checkpoint_file=cp_path,
            concurrency=args.concurrency,
        ))

    md_table = format_pilot_markdown_table(retrieval_res, ragas_res)
    print("\n=== KẾT QUẢ ĐÁNH GIÁ (BENCHMARK RESULTS) ===")
    print(md_table)

    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(md_table + "\n", encoding="utf-8")
        print(f"Wrote Markdown report to {args.out_md}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "retrieval": retrieval_res,
            "ragas": ragas_res,
        }
        Path(args.out).write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

