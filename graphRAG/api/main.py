from fastapi import FastAPI
from pydantic import BaseModel

from llm.ollama import OllamaLLM
from llm.answer_generator import AnswerGenerator

from retrieval.retrieve import Retriever


llm = OllamaLLM().get_llm()


vector_index = None
graph_index = None

retriever = Retriever(
    llm=llm,
    vector_index=vector_index,
    graph_index=graph_index
)

generator = AnswerGenerator(llm)


app = FastAPI(
    title="GraphRAG API",
    version="1.0"
)

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    contexts = retriever.retrieve(request.question)

    answer = generator.generate(
        question=request.question,
        contexts=contexts
    )

    return ChatResponse(
        answer=answer
    )