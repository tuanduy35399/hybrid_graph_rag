# -*- coding: utf-8 -*-
"""Markdown report exporter for benchmark evaluation results.

Produces clean, GitHub-flavored Markdown tables for human inspection,
documentation, and papers.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _format_num(val: Optional[float], precision: int = 3) -> str:
    if val is None:
        return "-"
    return f"{val:.{precision}f}"


def _format_ci(low: Optional[float], high: Optional[float], precision: int = 3) -> str:
    if low is None or high is None:
        return "-"
    return f"[{low:.{precision}f}, {high:.{precision}f}]"


def format_report_markdown(report: Any) -> str:
    """Render a SystemReport or parsed report dict as a Markdown string."""
    data: Dict[str, Any]
    if hasattr(report, "to_dict"):
        data = report.to_dict(include_per_question=False)
    elif isinstance(report, dict):
        data = report
    else:
        raise TypeError(f"expected SystemReport or dict, got {type(report).__name__}")

    system = data.get("system", "unknown")
    n_questions = data.get("n_questions", 0)
    n_evaluated = data.get("n_evaluated", 0)
    n_missing = data.get("n_missing_outputs", 0)
    n_errors = data.get("n_errors", 0)

    judge_obj = data.get("judge") or {}
    judge_name = judge_obj.get("judge", "null") if isinstance(judge_obj, dict) else str(judge_obj)
    usage = data.get("judge_usage") or {}
    notes = data.get("notes") or []

    metrics: Dict[str, Any] = data.get("metrics") or {}
    primary_names: Sequence[str] = data.get("primary_metrics") or ()
    diagnostic_names: Sequence[str] = data.get("diagnostic_metrics") or ()
    lower_is_better = set(data.get("lower_is_better") or ())

    lines: List[str] = []
    lines.append(f"# Báo cáo Đánh giá Benchmark: {system.upper()}")
    lines.append("")
    lines.append("## 1. Thông tin chung (Overview)")
    lines.append(f"- **Hệ thống (System):** `{system}`")
    lines.append(f"- **Tổng số câu hỏi:** `{n_questions}`")
    lines.append(f"  - Đã đánh giá (Evaluated): `{n_evaluated}`")
    lines.append(f"  - Thiếu kết quả (Missing outputs): `{n_missing}`")
    lines.append(f"  - Lỗi thực thi (Errors): `{n_errors}`")
    lines.append(f"- **Judge:** `{judge_name}`")
    if usage.get("calls"):
        lines.append(
            f"- **Judge calls:** {usage['calls']} (decided {usage.get('decided', 0)}, unknown {usage.get('unknown', 0)})"
        )
    if notes:
        lines.append("- **Ghi chú (Notes):**")
        for note in notes:
            lines.append(f"  - {note}")
    lines.append("")

    def _render_metric_table(title: str, names: Sequence[str]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Metric | Chiều tốt | Micro | Macro | 95% CI | n | N/A | Unav |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for name in names:
            summary = metrics.get(name)
            if not summary:
                continue
            direction = "↓" if name in lower_is_better else "↑"
            micro = _format_num(summary.get("micro"))
            macro = _format_num(summary.get("macro"))
            ci = _format_ci(summary.get("ci_low"), summary.get("ci_high"))
            n = summary.get("n", 0)
            na = summary.get("n_not_applicable", 0)
            unav = summary.get("n_unavailable", 0)
            lines.append(f"| `{name}` | {direction} | {micro} | {macro} | {ci} | {n} | {na} | {unav} |")
        lines.append("")

    if primary_names:
        _render_metric_table("2. Chỉ số chính (Primary Metrics)", primary_names)
    if diagnostic_names:
        _render_metric_table("3. Chỉ số chẩn đoán (Diagnostic Metrics)", diagnostic_names)

    # Category breakdown for primary metrics
    categories: List[str] = []
    for summary in metrics.values():
        if isinstance(summary, dict) and "by_category" in summary:
            for cat in summary["by_category"]:
                if cat not in categories:
                    categories.append(cat)

    if categories and primary_names:
        lines.append("## 4. Chi tiết theo danh mục (Breakdown by Category - Micro)")
        lines.append("")
        header = "| Metric | " + " | ".join(f"`{c}`" for c in categories) + " |"
        sep = "| :--- | " + " | ".join(":---:" for _ in categories) + " |"
        lines.append(header)
        lines.append(sep)
        for name in primary_names:
            summary = metrics.get(name)
            if not summary or "by_category" not in summary:
                continue
            by_cat = summary.get("by_category", {})
            row = [f"`{name}`"]
            for c in categories:
                row.append(_format_num(by_cat.get(c)))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


def format_comparison_markdown(reports: Sequence[Dict[str, Any]]) -> str:
    """Render a side-by-side comparison table of multiple system reports."""
    lines: List[str] = []
    lines.append("# Báo cáo So sánh Benchmark các hệ thống")
    lines.append("")
    lines.append("## 1. Tổng quan hệ thống")
    lines.append("")
    lines.append("| Hệ thống | Đã đánh giá | Lỗi | Judge |")
    lines.append("| :--- | :---: | :---: | :---: |")
    for r in reports:
        sys_name = r.get("system", "unknown")
        n_eval = r.get("n_evaluated", 0)
        n_err = r.get("n_errors", 0)
        judge = (r.get("judge") or {}).get("judge", "null")
        lines.append(f"| `{sys_name}` | {n_eval} | {n_err} | `{judge}` |")
    lines.append("")

    if not reports:
        return "\n".join(lines)

    first = reports[0]
    primary_names: Sequence[str] = first.get("primary_metrics") or ()
    lower_is_better = set(first.get("lower_is_better") or ())
    sys_names = [r.get("system", f"sys_{i}") for i, r in enumerate(reports)]

    lines.append("## 2. So sánh Primary Metrics (Micro Score)")
    lines.append("")
    header = "| Metric | Chiều tốt | " + " | ".join(f"**{s}**" for s in sys_names) + " |"
    sep = "| :--- | :---: | " + " | ".join(":---:" for _ in sys_names) + " |"
    lines.append(header)
    lines.append(sep)

    for name in primary_names:
        direction = "↓" if name in lower_is_better else "↑"
        row = [f"`{name}`", direction]
        for r in reports:
            m = (r.get("metrics") or {}).get(name, {})
            row.append(_format_num(m.get("micro")))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark.evaluator.export_markdown",
        description="Export benchmark evaluation report JSON(s) to Markdown format.",
    )
    parser.add_argument(
        "--report",
        help="Path to a single report JSON file to convert to Markdown",
    )
    parser.add_argument(
        "--reports",
        nargs="+",
        help="Path to multiple report JSON files for side-by-side comparison",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Target Markdown file path",
    )
    args = parser.parse_args(argv)

    if not args.report and not args.reports:
        parser.error("Either --report or --reports must be specified.")

    if args.reports:
        loaded = []
        for p in args.reports:
            with open(p, "r", encoding="utf-8") as f:
                loaded.append(json.load(f))
        md_text = format_comparison_markdown(loaded)
    else:
        with open(args.report, "r", encoding="utf-8") as f:
            data = json.load(f)
        md_text = format_report_markdown(data)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out_path, "w", encoding="utf-8", newline="\n") as fout:
        fout.write(md_text)
    print(f"Exported Markdown report to: {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
