import fitz  # PyMuPDF
import os
import io
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from PIL import Image
import numpy as np
from app.utils.visual_classifier import VisualClassifier
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TextBlock:
    page: int
    text: str
    bbox: List[float]
    block_id: int

@dataclass
class ImageData:
    page: int
    image_path: str
    bbox: List[float]
    type: str  # "chart", "table", "diagram", "image"
    caption_context: Optional[str] = None

@dataclass
class TableData:
    page: int
    bbox: List[float]
    nearby_text: Optional[str] = None
    # We might want to store the table content itself if extraction is successful, 
    # but the requirement focuses on detection/schema primarily.
    # content: Optional[List[List[str]]] = None 

class PDFParser:
    def __init__(self, output_dir: str = "data/extracted_images"):
        """
        Initialize the PDFParser with an output directory for images.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.classifier = VisualClassifier()

    def parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Parses a PDF file to extract text, images, and tables.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            A dictionary containing extracted text blocks, images, and tables.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise ValueError(f"Invalid or corrupted PDF file: {file_path}") from e

        extracted_data = {
            "text_blocks": [],
            "images": [],
            "tables": [],
            "metadata": doc.metadata
        }

        for page_num, page in enumerate(doc):
            # Extract Text Blocks
            text_blocks = self._extract_text_blocks(page, page_num)
            extracted_data["text_blocks"].extend([asdict(tb) for tb in text_blocks])

            # Extract Tables
            tables = self._extract_tables(page, page_num)
            extracted_data["tables"].extend([asdict(t) for t in tables])

            # Extract Images (and classify them)
            # Pass detected table bboxes to avoid classifying table regions as pure images if overlapping,
            # though PyMuPDF handles images and tables differently.
            images = self._extract_images(page, page_num, doc, file_path)
            extracted_data["images"].extend([asdict(img) for img in images])

        doc.close()
        return extracted_data

    def _extract_text_blocks(self, page, page_num: int) -> List[TextBlock]:
        """Extracts text blocks with bounding boxes."""
        blocks = page.get_text("blocks")
        text_blocks = []
        for i, block in enumerate(blocks):
            # Block format: (x0, y0, x1, y1, "lines", block_no, block_type)
            # We want text blocks (type 0)
            if block[6] == 0:
                bbox = list(block[:4])
                text = block[4].strip()
                if text:
                    text_blocks.append(TextBlock(
                        page=page_num + 1,
                        text=text,
                        bbox=bbox,
                        block_id=i
                    ))
        return text_blocks

    def _extract_tables(self, page, page_num: int) -> List[TableData]:
        """
        Detects tables using PyMuPDF's find_tables.
        """
        tables = []
        try:
            # find_tables looks for table structures
            found_tables = page.find_tables()
            for tab in found_tables:
                bbox = list(tab.bbox)
                # Find nearby text (e.g., caption above or below)
                # This is a naive heuristic: look at text slightly above the table
                text_rect = fitz.Rect(bbox[0], bbox[1] - 50, bbox[2], bbox[1])
                nearby_text = page.get_text("text", clip=text_rect).strip()
                
                tables.append(TableData(
                    page=page_num + 1,
                    bbox=bbox,
                    nearby_text=nearby_text if nearby_text else None
                ))
        except Exception as e:
            logger.warning(f"Table extraction failed on page {page_num}: {e}")
        
        return tables

    def _extract_images(self, page, page_num: int, doc, file_path_str: str) -> List[ImageData]:
        """Extracts images and classifies them."""
        extracted_images = []
        image_list = page.get_images(full=True)
        pdf_name = os.path.splitext(os.path.basename(file_path_str))[0]

        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Check minimum size to avoid tiny icons or lines being classified as charts
                if len(image_bytes) < 100: # heuristic, skip very small images
                    continue

                image = Image.open(io.BytesIO(image_bytes))
                
                # Save image
                image_filename = f"{pdf_name}_p{page_num+1}_i{img_index}.{image_ext}"
                image_path = os.path.join(self.output_dir, image_filename)
                image.save(image_path)
                
                # Get bbox of the image on the page
                # PyMuPDF: get_image_rects gives where the image is drawn
                rects = page.get_image_rects(xref)
                vocab_bbox = [0.0, 0.0, 0.0, 0.0]
                if rects:
                     # Just take the first occurrence if multiple
                    vocab_bbox = list(rects[0])

                # Classify
                classification = self.classifier.classify(image)
                img_type = classification["type"]

                # Attempt to find caption (text below image)
                caption = None
                if rects:
                    r = rects[0]
                    caption_rect = fitz.Rect(r.x0, r.y1, r.x1, r.y1 + 50)
                    caption = page.get_text("text", clip=caption_rect).strip()

                extracted_images.append(ImageData(
                    page=page_num + 1,
                    image_path=image_path,
                    bbox=vocab_bbox,
                    type=img_type,
                    caption_context=caption
                ))

            except Exception as e:
                logger.warning(f"Failed to extract image {img_index} on page {page_num}: {e}")
                continue

        return extracted_images

    def classify_visual_element(self, image: Image.Image) -> str:
        """
        [Deprecated] Uses internal classifier now.
        Kept for backward internal compatibility if needed, but delegating.
        """
        return self.classifier.classify(image)["type"]


def parse_pdf(file_path: str, filename: str = None) -> Dict[str, Any]:
    """
    Top-level helper to support the API interface.
    """
    # If filename is provided, it might be used for naming images, 
    # but PDFParser currently uses basename of file_path. 
    # We can stick to defaults for now or update PDFParser to accept a prefix.
    parser = PDFParser(output_dir=os.path.join(settings.EXTRACTED_DIR if hasattr(settings, 'EXTRACTED_DIR') else "data/extracted_images"))
    return parser.parse_pdf(file_path)
