import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings
from typing import List, Dict

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(name="rag_collection")

    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str], embeddings: List[List[float]]):
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_embedding: List[float], n_results: int = 3):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

vector_store = VectorStore()
