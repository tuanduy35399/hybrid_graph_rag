from typing import Optional
from llama_index.core.indices.property_graph import PropertyGraphIndex


class GraphSearch:

    def __init__(self, graph_index: Optional[PropertyGraphIndex] = None, top_k: int = 5):
        self.graph_index = graph_index
        self.top_k = top_k

    def search(self, question: str):
        if self.graph_index is None:
            return []
        print("Buoc graph search")
        try:
            retriever = self.graph_index.as_retriever(
                similarity_top_k=self.top_k
            )
            return retriever.retrieve(question)
        except Exception as e:
            print(f"[CẢNH BÁO] Graph search gặp lỗi ({e}), bỏ qua kết quả graph.")
            return []