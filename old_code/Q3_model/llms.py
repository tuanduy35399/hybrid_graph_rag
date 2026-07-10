

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate 

llm = OllamaLLM(model ='qwen2.5:7b')

template = """
Bạn là hệ thống hỏi đáp chuyên về bệnh trên cây mai vàng.

QUY TẮC BẮT BUỘC
1. Chủ yếu sử dụng thông tin trong CONTEXT, nếu không đủ thông tin có thể tìm kiếm thêm thông tin
bằng cách yêu cầu người dùng nhập thêm mô tả chi tiết thông mà bạn cần muốn biết.
2. KHÔNG được suy đoán, không được dùng kiến thức bên ngoài.
3. Nếu CONTEXT không chứa câu trả lời → trả về:
   "Không đủ thông tin trong dữ liệu để trả lời."
4. Không được bịa thêm nguyên nhân, triệu chứng hoặc cách chữa.
5. Không được trích nguyên văn CONTEXT.

CÁCH TRẢ LỜI
- Trả lời 1 cách chi tiết, cụ thể, đúng trọng tâm
- Giải thích dễ hiểu cho người nông dân
- Nếu có bệnh → nêu nguyên nhân + dấu hiệu + hướng xử lý (chỉ nếu có trong context)
- Gợi ý thêm 1 số câu hỏi mở rộng để duy trì cuộc trò chuyện cho đến khi người dùng chấp nhận câu trả lời
CONTEXT:
{context}

CÂU HỎI:
{question}

========================
TRẢ LỜI:
"""
prompt = ChatPromptTemplate.from_template(template)  
#tạo 1 prompt mà LangChain được được, nó hiểu chỉ cần truyền vào 2 biến context và question
#lúc này kq là 1 obj với 2 khóa có gtri đang rỗng là context và question (ta sẽ truyền ở hàm ask_llm)
chain= prompt | llm # dấu | có ý nghĩa là nối các bước xử lý thành 1 pipeline

def ask_llm(question, context):
    

    res = chain.invoke( {
        "context": context,
        "question": question,
    }
    )
    return res
# from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# client = NVIDIAEmbeddings(
#   model="nvidia/nv-embedcode-7b-v1", 
#   api_key="$NVIDIA_API_KEY", 
#   truncate="NONE", 
#   )

# embedding = client.embed_query("What is the capital of France?")
# print(embedding)

