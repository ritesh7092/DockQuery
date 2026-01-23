from sentence_transformers import SentenceTransformer
from app.config import settings

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def get_text_embedding(self, text: str):
        return self.model.encode(text).tolist()

    # Placeholder for image embeddings if needed using a multimodal model (e.g. CLIP)
    # def get_image_embedding(self, image_path: str):
    #     ...
        
embedding_service = EmbeddingService()
