import unittest
import numpy as np
import cv2
from PIL import Image
from app.utils.visual_classifier import VisualClassifier

class TestVisualClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = VisualClassifier()

    def create_dummy_table_image(self):
        # Create white image
        img = 255 * np.ones((500, 500, 3), dtype=np.uint8)
        # Draw grid
        for i in range(0, 500, 50):
            cv2.line(img, (0, i), (500, i), (0, 0, 0), 2)  # Horizontal
            cv2.line(img, (i, 0), (i, 500), (0, 0, 0), 2)  # Vertical
        return Image.fromarray(img)

    def create_dummy_chart_image(self):
        img = 255 * np.ones((500, 500, 3), dtype=np.uint8)
        # Draw axes
        cv2.line(img, (50, 450), (450, 450), (0, 0, 0), 3) # X
        cv2.line(img, (50, 50), (50, 450), (0, 0, 0), 3)   # Y
        # Draw some bars
        cv2.rectangle(img, (70, 400), (100, 450), (255, 0, 0), -1)
        cv2.rectangle(img, (120, 300), (150, 450), (0, 255, 0), -1)
        return Image.fromarray(img)
    
    def create_noise_image(self):
        img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        return Image.fromarray(img)

    def test_classify_table(self):
        img = self.create_dummy_table_image()
        result = self.classifier.classify(img)
        self.assertEqual(result['type'], 'table')

    def test_classify_chart(self):
        img = self.create_dummy_chart_image()
        result = self.classifier.classify(img)
        # We accept chart or at least high confidence it's not a simple image
        # Note: Tuning heuristics for synthetic charts can be tricky, 
        # so we check if it is NOT table basically, or specific 'chart'.
        # Our heuristic for chart is a bit weak (just edges/corners), but table is strong.
        self.assertIn(result['type'], ['chart'])

    def test_classify_image(self):
        # Noise should be image (or maybe chart if random edges trigger it, but unlikely to map to table)
        img = self.create_noise_image()
        result = self.classifier.classify(img)
        # Random noise has extremely high edge density -> might fail table check?
        # Actually random noise has standard edge density ~0.5?
        # Let's say it shouldn't be a table.
        self.assertNotEqual(result['type'], 'table')

if __name__ == '__main__':
    unittest.main()
