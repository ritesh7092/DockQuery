import os
import shutil
import unittest
import fitz
from PIL import Image
import numpy as np
from app.services.pdf_parser import PDFParser

class TestPDFParser(unittest.TestCase):
    def setUp(self):
        self.output_dir = "tests/test_data/extracted"
        self.pdf_path = "tests/test_data/test.pdf"
        os.makedirs("tests/test_data", exist_ok=True)
        
        # Create a dummy PDF
        self.doc = fitz.open()
        page = self.doc.new_page()
        
        # Add Text
        page.insert_text((50, 50), "Test PDF Document", fontsize=20)
        page.insert_text((50, 100), "Page 1 Content", fontsize=12)
        
        # Add a simple image
        # Create a red square image
        img = Image.new('RGB', (100, 100), color = 'red')
        img.save("tests/test_data/temp_img.png")
        page.insert_image(fitz.Rect(100, 200, 200, 300), filename="tests/test_data/temp_img.png")
        
        self.doc.save(self.pdf_path)
        self.doc.close()

    def tearDown(self):
        if os.path.exists("tests/test_data"):
            shutil.rmtree("tests/test_data")

    def test_parse_pdf_structure(self):
        parser = PDFParser(output_dir=self.output_dir)
        result = parser.parse_pdf(self.pdf_path)
        
        # Check high-level structure
        self.assertIn("text_blocks", result)
        self.assertIn("images", result)
        self.assertIn("tables", result)
        self.assertIn("metadata", result)
        
        # Check Text Extraction
        texts = [b['text'] for b in result['text_blocks']]
        self.assertTrue(any("Test PDF Document" in t for t in texts))
        
        # Check Image Extraction
        # We inserted one image
        self.assertTrue(len(result['images']) >= 1)
        img_data = result['images'][0]
        self.assertEqual(img_data['page'], 1)
        self.assertTrue(os.path.exists(img_data['image_path']))
        
    def test_invalid_file(self):
        parser = PDFParser(output_dir=self.output_dir)
        with self.assertRaises(FileNotFoundError):
            parser.parse_pdf("non_existent.pdf")

if __name__ == '__main__':
    unittest.main()
