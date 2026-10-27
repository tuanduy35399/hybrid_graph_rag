from typing import Optional, Dict, List
import sys
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from llm.nvidia import NvidiaNimLLM
from llm.answer_generator import AnswerGenerator
from retrieval.retrieve import Retriever
from retrieval.metrics.latency_tracker import LatencyTracker
from database.neo4j_store import load_graph_index
from database.chroma_store import load_vector_index


llm = NvidiaNimLLM().get_llm()

vector_index = load_vector_index()
graph_index = load_graph_index()

retriever = Retriever(
    llm=llm,
    vector_index=vector_index,
    graph_index=graph_index
)

generator = AnswerGenerator(llm=llm)


app = FastAPI(
    title="HybridRAG Mai Vang API",
    description="API hỏi đáp chẩn đoán và điều trị bệnh trên cây mai vàng tích hợp đo độ trễ chi tiết.",
    version="1.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    latencies: Optional[Dict[str, float]] = None
    retrieved_contexts: Optional[List[str]] = None


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Khởi tạo bộ đo thời gian từng giai đoạn
    tracker = LatencyTracker()

    # 1-6. Giai đoạn truy xuất ngữ cảnh (Rewrite, HyDE, Vector, Graph, Merge, Rerank)
    contexts = retriever.retrieve(request.question, tracker=tracker)

    # 7. Giai đoạn sinh câu trả lời
    answer = generator.generate(
        question=request.question,
        contexts=contexts,
        tracker=tracker
    )

    tracker.stop_total()

    # In bảng thống kê chi tiết ra terminal
    print(tracker.summary())

    # Thu thập danh sách context để hỗ trợ debug/frontend/evaluation
    retrieved_texts = []
    for c in contexts:
        if hasattr(c, "node") and hasattr(c.node, "get_content"):
            retrieved_texts.append(c.node.get_content())
        elif hasattr(c, "text"):
            retrieved_texts.append(c.text)
        else:
            retrieved_texts.append(str(c))

    return ChatResponse(
        answer=answer,
        latencies=tracker.to_dict(unit="seconds"),
        retrieved_contexts=retrieved_texts
    )