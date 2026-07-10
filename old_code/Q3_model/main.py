from fastapi import FastAPI
from fastAPI.old_code.Q3_model.llms import ask_llm
import uuid
from fastAPI.old_code.Q3_model.rag import retriever

app = FastAPI()


@app.get("/ask")
def ask(q: str):

    trace_id = str(uuid.uuid4())
    #buoc retrieval
    docs = retriever.invoke(q)
    print("QUESTION:", q, end='\n')
    context = "\n".join(
        d.page_content
        for d in docs
    )
    print("CONTEXT:",context,end='\n')
    

    answer = ask_llm(
        question=q,
        context=context
    )
    print("ANSWER:",answer,end='\n')
    #ghi log lai de danh gia

    return {
        "question": q,
        "context": context,
        "answer": answer
    }
    
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
# from datasets import Dataset

# @app.get("/evaluate")
# def evaluate_rag():

#     dataset = Dataset.from_list([
#         {
#             "question": x["question"],
#             "contexts": x["contexts"],
#             "answer": x["answer"]
#         }
#         for x in RAG_LOGS
#     ])

#     result = evaluate(
#         dataset,
#         metrics=[
#             faithfulness,
#             answer_relevancy,
#             context_precision
#         ]
#     )

#     return result