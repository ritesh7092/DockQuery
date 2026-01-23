import google.generativeai as genai
from app.config import settings

class RAGAgent:
    def __init__(self):
        if settings.GOOGLE_API_KEY:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            self.model = None

    def generate_response(self, query: str, context: str):
        if not self.model:
            return "Google API Key not configured."
        
        prompt = f"""
        Answer the following query based on the provided context.
        
        Context:
        {context}
        
        Query: {query}
        """
        response = self.model.generate_content(prompt)
        return response.text

rag_agent = RAGAgent()
