from typing import Optional, Union, List
from llm.nvidia import NvidiaNimLLM
from llm.prompts import PROMPT, build_prompt
from retrieval.metrics.latency_tracker import LatencyTracker


class AnswerGenerator:

    def __init__(self, llm=None):
        self.llm = llm if llm is not None else NvidiaNimLLM().get_llm()

    def _format_context(self, contexts: Union[str, List]) -> str:
        """Chuẩn hóa ngữ cảnh đầu vào thành dạng chuỗi dễ đọc cho LLM."""
        if isinstance(contexts, str):
            return contexts

        formatted_chunks = []
        for i, item in enumerate(contexts, 1):
            if hasattr(item, "node") and hasattr(item.node, "get_content"):
                text = item.node.get_content().strip()
            elif hasattr(item, "text"):
                text = item.text.strip()
            else:
                text = str(item).strip()
            formatted_chunks.append(f"[Đoạn {i}]:\n{text}")

        return "\n\n".join(formatted_chunks)

    def generate(
        self,
        question: str,
        contexts: Union[str, List],
        tracker: Optional[LatencyTracker] = None
    ) -> str:
        context_str = self._format_context(contexts)

        prompt = build_prompt(
            context=context_str,
            question=question,
        )

        full_prompt = (
            PROMPT
            + "\n\n"
            + prompt
        )

        if tracker:
            with tracker.track("answer_generation"):
                response = self.llm.complete(full_prompt)
        else:
            response = self.llm.complete(full_prompt)

        return response.text