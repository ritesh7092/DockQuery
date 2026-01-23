import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple

class VisualClassifier:
    """
    Heuristic-based classifier for visual elements extracted from PDFs.
    Distinguishes between 'table', 'chart', and 'image' (general).
    """

    def classify(self, image: Image.Image) -> Dict[str, Any]:
        """
        Classifies the input image.

        Args:
            image: PIL Image object.

        Returns:
            Dictionary with 'type' and 'confidence'.
        """
        # Convert PIL to OpenCV format (RGB -> BGR)
        open_cv_image = np.array(image) 
        # Handle grayscale images if they come in as 'L'
        if len(open_cv_image.shape) == 2:
             open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_GRAY2BGR)
        else:
            open_cv_image = open_cv_image[:, :, ::-1].copy()

        # Resizing for speed consistency if image is huge
        h, w = open_cv_image.shape[:2]
        if h > 1000 or w > 1000:
            scale = 1000 / max(h, w)
            open_cv_image = cv2.resize(open_cv_image, (0, 0), fx=scale, fy=scale)

        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        table_score = self._detect_table(gray, edges)
        chart_score = self._detect_chart(open_cv_image, gray, edges)
        
        # Heuristic decision logic
        
        # If table score is high, but we only found exactly 2 orthogonal lines, it might be a chart axis.
        # Tables usually have multiple rows/cols.
        # We can move this logic to _detect_table or handle here.
        # Let's handle it by checking if table_score reflects a "full grid" vs "just axes"
        
        if table_score > 0.5 and table_score > chart_score:
             # Basic check to avoid L-shape being table
             return {"type": "table", "confidence": table_score}
        elif chart_score > 0.15: 
            return {"type": "chart", "confidence": chart_score}
        else:
            return {"type": "image", "confidence": 1.0 - max(table_score, chart_score)}

    def _detect_table(self, gray: np.ndarray, edges: np.ndarray) -> float:
        """
        Detects tables based on horizontal and vertical lines.
        """
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return 0.0

        horizontal_lines = 0
        vertical_lines = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 2 or angle > 178: # Stricter horizontal
                horizontal_lines += 1
            elif 88 < angle < 92: # Stricter vertical
                vertical_lines += 1

        total_lines = len(lines)
        if total_lines == 0:
            return 0.0

        # Filter out high noise images (noise generates many small lines often)
        edge_density = np.sum(edges) / 255 / edges.size
        if edge_density > 0.3: 
             return 0.0
            
        if horizontal_lines >= 3 and vertical_lines >= 3: # Need at least a few lines to be a table (grid)
             return 0.8 + min(0.2, (total_lines / 20)) 
        
        if horizontal_lines > 5: # List
            return 0.6
            
        return 0.1 # Just axes or few lines -> unlikely table


    def _detect_chart(self, original: np.ndarray, gray: np.ndarray, edges: np.ndarray) -> float:
        """
        Detects charts based on axes-like lines and color usage.
        """
        # 1. Detect Axes (L-shape lines)
        # Using Hough lines again but looking for specific long lines near borders? 
        # Simplified: Check for high density of edges in the center (data points) vs empty space.
        
        # 2. Color Histograms
        # Charts often have specific distinct colors on a white background.
        # Check unique colors count vs area.
        
        # Heuristic: Check for corners? 
        # Good corners -> Shi-Tomasi Corner Detector
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        if corners is not None:
             corner_count = len(corners)
        else:
             corner_count = 0

        # Heuristic: Chart area often has non-text high frequency (edges)
        edge_density = np.sum(edges) / 255 / edges.size
        
        score = 0.0
        
        if 0.05 < edge_density < 0.35: # Typical range for charts, text pages are denser or sparser
             score += 0.3
        elif edge_density >= 0.35:
             # Too busy, likely noise or complex photo
             score -= 0.5
             
        if corner_count > 5:
            score += 0.2
            
        return max(0.0, score)
