from gemini import GeminiLLM

llm = GeminiLLM()

response = llm.generate(
    "Giới thiệu ngắn về cây mai vàng."
)

print(response)