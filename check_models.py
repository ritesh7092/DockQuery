"""Script to list available Gemini models."""
import google.generativeai as genai
from app.config import settings

# Configure the API
genai.configure(api_key=settings.GOOGLE_API_KEY)

with open("available_models.txt", "w") as f:
    f.write("Available Gemini models:\n")
    f.write("=" * 80 + "\n\n")
    
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            f.write(f"Name: {model.name}\n")
            f.write(f"Display Name: {model.display_name}\n")
            f.write(f"Description: {model.description}\n")
            f.write(f"Supported methods: {model.supported_generation_methods}\n")
            f.write("=" * 80 + "\n\n")
    
print("Results written to available_models.txt")
