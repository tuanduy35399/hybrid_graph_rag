from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import SentenceTransformer
import chromadb


_client = chromadb.PersistentClient(
    path="chroma_db_split_markdownheader"
)

_collection = _client.get_collection(
    "cay_mai_md"
)


_model = SentenceTransformer(
    "BAAI/bge-m3"
)

class ChromaRetriever(BaseRetriever):
    k : int = 3
    def _get_relevant_documents(self, query):
        query = (
            "Represent this sentence for searching relevant passages: "
            + query
        )
        q_emb = _model.encode(query).tolist()
        results = _collection.query(
            query_embeddings=[q_emb],
            n_results=self.k,
            include=["documents"]
        )
        docs = []

        for text in results["documents"][0]:
            docs.append(
                Document(
                    page_content=text
                )
            )
        return docs


retriever = ChromaRetriever()