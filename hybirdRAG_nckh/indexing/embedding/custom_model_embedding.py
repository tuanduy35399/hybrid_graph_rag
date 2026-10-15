from llama_index.core import Settings
from indexing.embedding.bge_m3 import BGE_M3_Embedding

Settings.chunk_size = 1024
Settings.chunk_overlap = 100


class CustomModelEmbedding:
    """
    Wrapper mô hình Embedding dùng BAAI/bge-m3
    - Chạy hoàn toàn offline local trên máy
    - Hỗ trợ tiếng Việt xuất sắc, vector 1024 chiều
    """

    def __init__(self):
        # Khởi tạo BGE-M3 từ class đã có sẵn trong indexing/embedding/bge_m3.py
        self.embed_model = BGE_M3_Embedding().get_model()
        Settings.embed_model = self.embed_model

    def get_model(self):
        Settings.embed_model = self.embed_model
        return self.embed_model


# Alias tương thích ngược để các file gọi NVIDIAEmbeddingWrapper cũ vẫn chạy bình thường
NVIDIAEmbeddingWrapper = CustomModelEmbedding