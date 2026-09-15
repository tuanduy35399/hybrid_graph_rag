from typing import Optional, Tuple, List
from retrieval.rewrite.query_rewrite import QueryRewrite
from retrieval.rewrite.hyde import HyDE
from retrieval.search.vector_search import VectorSearch
from retrieval.search.graph_search import GraphSearch
from retrieval.search.hybrid_search import HybridSearch
from retrieval.reranker.bge_reranker import CrossEncoderReranker
from retrieval.metrics.latency_tracker import LatencyTracker


class Retriever:

    def __init__(self, llm, vector_index, graph_index, top_k: int = 5):
        self.rewriter = QueryRewrite(llm)
        self.hyde = HyDE(llm)

        self.vector_search = VectorSearch(
            vector_index=vector_index,
            top_k=top_k
        )

        self.graph_search = GraphSearch(
            graph_index=graph_index,
            top_k=top_k
        )

        self.hybrid = HybridSearch()

        self.reranker = CrossEncoderReranker(
            top_k=top_k
        )

    def retrieve(self, question: str, tracker: Optional[LatencyTracker] = None):
        """
        Truy xuất ngữ cảnh lai (Hybrid Retrieval) kèm đo thời gian từng giai đoạn nếu tracker được truyền vào.
        Giữ nguyên 100% tương thích ngược với code cũ.
        """
        # 1. Query Rewrite
        if tracker:
            with tracker.track("query_rewrite"):
                rewritten_question = self.rewriter.rewrite(question)
        else:
            rewritten_question = self.rewriter.rewrite(question)

        # 2. HyDE Generation
        if tracker:
            with tracker.track("hyde_generation"):
                hyde_query = self.hyde.generate(rewritten_question)
        else:
            hyde_query = self.hyde.generate(rewritten_question)

        # 3. Vector Search
        if tracker:
            with tracker.track("vector_search"):
                vector_results = self.vector_search.search(hyde_query)
        else:
            vector_results = self.vector_search.search(hyde_query)

        # 4. Graph Search
        if tracker:
            with tracker.track("graph_search"):
                graph_results = self.graph_search.search(rewritten_question)
        else:
            graph_results = self.graph_search.search(rewritten_question)

        # 5. Hybrid Merge
        if tracker:
            with tracker.track("hybrid_merge"):
                merged_results = self.hybrid.merge(vector_results, graph_results)
        else:
            merged_results = self.hybrid.merge(vector_results, graph_results)

        # 6. Reranking
        print("Reranking")
        if tracker:
            with tracker.track("reranking"):
                final_results = self.reranker.rerank(rewritten_question, merged_results)
        else:
            final_results = self.reranker.rerank(rewritten_question, merged_results)

        return final_results

    def retrieve_with_latency(self, question: str) -> Tuple[List, LatencyTracker]:
        """
        Truy xuất kèm trả về tracker chứa thời gian chi tiết của từng bước.
        """
        tracker = LatencyTracker()
        results = self.retrieve(question, tracker=tracker)
        return results, tracker