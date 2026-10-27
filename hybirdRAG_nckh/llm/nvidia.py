# llm/nvidia_nim.py
import os
from dotenv import load_dotenv
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings
from llama_index.core.types import PydanticProgramMode

load_dotenv()


class NvidiaNimLLM:

    def __init__(self):
        model = os.getenv("LLM_MODEL", "ag/gemini-3-flash")
        api_base = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8317/v1")
        api_key = os.getenv("LLM_API_KEY") or os.getenv("NVIDIA_API_KEY") or "sk-7a0fae48fda0f62a-1l7z0p-917eb4ce"

        self.llm = OpenAILike(
            model=model,
            api_base=api_base,
            api_key=api_key,
            is_chat_model=True,
            is_function_calling_model=False,
            pydantic_program_mode=PydanticProgramMode.LLM,
            context_window=16384,
            timeout=60.0,
            max_retries=2,
        )

        Settings.llm = self.llm

    def get_llm(self):
        return self.llm