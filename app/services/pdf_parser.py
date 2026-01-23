import fitz  # PyMuPDF
import os
from PIL import Image
import io
from app.config import settings

def parse_pdf(file_path: str, filename: str):
    """
    Extracts text and images from a PDF file.
    """
    doc = fitz.open(file_path)
    text_content = ""
    images = []

    for page_num, page in enumerate(doc):
        text_content += page.get_text()
        
        # Extract images
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image = Image.open(io.BytesIO(image_bytes))
            
            image_filename = f"{filename}_p{page_num}_i{img_index}.{image_ext}"
            image_path = os.path.join(settings.EXTRACTED_DIR, image_filename)
            image.save(image_path)
            images.append(image_path)
            
    return {"text": text_content, "images": images}
