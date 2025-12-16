"""
OCR Service for License Plate Character Recognition
Advanced preprocessing techniques for improved accuracy
"""
import cv2
import numpy as np
import re
from typing import Tuple, Optional, List
import easyocr


class OCRService:
    """Service for extracting text from license plate images using EasyOCR."""
    
    _instance: Optional["OCRService"] = None
    _reader: Optional[easyocr.Reader] = None
    
    # Pattern to detect expiry date (MM-YY, MM/YY, MM.YY, or MMYY)
    EXPIRY_DATE_PATTERN = re.compile(r'^(0[1-9]|1[0-2])[-/.]?(\d{2})$')
    
    def __new__(cls):
        """Singleton pattern for reader loading efficiency."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the EasyOCR reader if not already loaded."""
        if OCRService._reader is None:
            self._load_reader()
    
    def _load_reader(self):
        """Load the EasyOCR reader."""
        print("Loading EasyOCR reader...")
        OCRService._reader = easyocr.Reader(['en'], gpu=False)
        print("EasyOCR reader loaded successfully!")
    
    @property
    def reader(self) -> easyocr.Reader:
        """Get the loaded EasyOCR reader."""
        if OCRService._reader is None:
            self._load_reader()
        return OCRService._reader
    
    def is_expiry_date(self, text: str) -> bool:
        """Check if text looks like an expiry date (MM-YY format)."""
        cleaned = text.strip().replace(' ', '').replace('-', '').replace('/', '').replace('.', '')
        
        if len(cleaned) == 4 and cleaned.isdigit():
            month = int(cleaned[:2])
            if 1 <= month <= 12:
                return True
        
        if re.match(r'^(0[1-9]|1[0-2])[-/.\s]?\d{2}$', text.strip()):
            return True
            
        return False
    
    def get_bbox_left(self, bbox) -> float:
        """Get the leftmost x-coordinate of a bounding box."""
        return min(point[0] for point in bbox)
    
    def sort_results_by_position(self, results: List) -> List:
        """Sort OCR results by position: top-to-bottom, then left-to-right."""
        if not results:
            return results
        
        heights = []
        for detection in results:
            bbox = detection[0]
            y_coords = [point[1] for point in bbox]
            heights.append(max(y_coords) - min(y_coords))
        
        avg_height = sum(heights) / len(heights) if heights else 30
        
        sorted_results = []
        remaining = list(results)
        
        while remaining:
            remaining.sort(key=lambda r: min(point[1] for point in r[0]))
            first_item = remaining[0]
            row_y = min(point[1] for point in first_item[0])
            
            row_items = []
            still_remaining = []
            
            for item in remaining:
                item_y = min(point[1] for point in item[0])
                if abs(item_y - row_y) < avg_height * 0.6:
                    row_items.append(item)
                else:
                    still_remaining.append(item)
            
            row_items.sort(key=lambda r: self.get_bbox_left(r[0]))
            sorted_results.extend(row_items)
            remaining = still_remaining
        
        return sorted_results

    # ==================== ADVANCED PREPROCESSING TECHNIQUES ====================
    
    def resize_image(self, image: np.ndarray, target_height: int = 100) -> np.ndarray:
        """
        Resize image to optimal height for OCR while maintaining aspect ratio.
        """
        h, w = image.shape[:2]
        if h < target_height:
            scale = target_height / h
            new_w = int(w * scale)
            image = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
        return image
    
    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew image by detecting and correcting rotation angle.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find coordinates of non-zero pixels
        coords = np.column_stack(np.where(thresh > 0))
        
        if len(coords) < 10:
            return image
        
        # Get the minimum area rectangle
        try:
            angle = cv2.minAreaRect(coords)[-1]
            
            # Adjust angle
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
            
            # Only correct if angle is significant but not too extreme
            if abs(angle) > 0.5 and abs(angle) < 15:
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h), 
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
                return rotated
        except:
            pass
        
        return image
    
    def remove_borders(self, image: np.ndarray) -> np.ndarray:
        """
        Remove black/white borders from plate image.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box of largest contour
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            
            # Only crop if the detected region is reasonable
            orig_h, orig_w = image.shape[:2]
            if w > orig_w * 0.5 and h > orig_h * 0.3:
                # Add small padding
                pad = 5
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(orig_w - x, w + 2 * pad)
                h = min(orig_h - y, h + 2 * pad)
                return image[y:y+h, x:x+w]
        
        return image
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE.
        """
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge and convert back
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from image using Non-Local Means Denoising.
        """
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        return denoised
    
    def sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """
        Sharpen image using unsharp masking.
        """
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        
        # Apply unsharp mask
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        
        return sharpened
    
    def binarize_adaptive(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding for binarization.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply bilateral filter first
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def binarize_otsu(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Otsu's thresholding for binarization.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu's threshold
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def morphological_cleanup(self, image: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up the image.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create kernels
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        # Close small holes
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_close)
        
        # Remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        
        return opened
    
    def invert_if_dark(self, image: np.ndarray) -> np.ndarray:
        """
        Invert image if background is dark (for dark plates with light text).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Check average brightness
        avg_brightness = np.mean(gray)
        
        if avg_brightness < 127:
            # Dark background, invert
            if len(image.shape) == 3:
                return cv2.bitwise_not(image)
            else:
                return cv2.bitwise_not(gray)
        
        return image
    
    # ==================== PREPROCESSING PIPELINES ====================
    
    def preprocess_pipeline_standard(self, image: np.ndarray) -> np.ndarray:
        """Standard preprocessing pipeline."""
        img = self.resize_image(image, target_height=80)
        img = self.enhance_contrast(img)
        img = self.denoise_image(img)
        img = self.sharpen_image(img)
        img = self.binarize_otsu(img)
        return img
    
    def preprocess_pipeline_aggressive(self, image: np.ndarray) -> np.ndarray:
        """Aggressive preprocessing for difficult images."""
        img = self.resize_image(image, target_height=100)
        img = self.deskew_image(img)
        img = self.remove_borders(img)
        img = self.enhance_contrast(img)
        img = self.denoise_image(img)
        img = self.sharpen_image(img)
        img = self.binarize_adaptive(img)
        img = self.morphological_cleanup(img)
        return img
    
    def preprocess_pipeline_dark_plate(self, image: np.ndarray) -> np.ndarray:
        """Pipeline for dark plates with light text."""
        img = self.resize_image(image, target_height=100)
        img = self.enhance_contrast(img)
        img = self.invert_if_dark(img)
        img = self.denoise_image(img)
        img = self.sharpen_image(img)
        img = self.binarize_otsu(img)
        return img
    
    def preprocess_pipeline_light_plate(self, image: np.ndarray) -> np.ndarray:
        """Pipeline for light plates with dark text."""
        img = self.resize_image(image, target_height=100)
        img = self.enhance_contrast(img)
        img = self.sharpen_image(img)
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Otsu threshold
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def preprocess_pipeline_minimal(self, image: np.ndarray) -> np.ndarray:
        """Minimal preprocessing - just resize and enhance."""
        img = self.resize_image(image, target_height=80)
        img = self.enhance_contrast(img)
        return img
    
    # ==================== MAIN OCR METHODS ====================
    
    def filter_by_vertical_position(
        self, 
        results: List, 
        image_height: int,
        top_portion: float = 0.65
    ) -> List:
        """Filter OCR results to focus on the top portion of the image."""
        cutoff_y = image_height * top_portion
        filtered = []
        
        for detection in results:
            bbox, text, confidence = detection
            y_coords = [point[1] for point in bbox]
            center_y = sum(y_coords) / len(y_coords)
            
            if center_y < cutoff_y:
                filtered.append(detection)
        
        return filtered
    
    def _looks_like_plate(self, text: str) -> bool:
        """Check if text looks like a valid Indonesian license plate format."""
        cleaned = text.replace(' ', '').upper()
        
        if len(cleaned) < 4:
            return False
        
        pattern = re.compile(r'^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$')
        return bool(pattern.match(cleaned))
    
    def _run_ocr_on_image(self, image: np.ndarray) -> List:
        """Run EasyOCR on an image and return results."""
        try:
            results = self.reader.readtext(
                image,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                paragraph=False,
                min_size=5,
                text_threshold=0.5,
                low_text=0.3,
                contrast_ths=0.1,
                adjust_contrast=0.5,
            )
            return results
        except Exception as e:
            print(f"OCR error: {e}")
            return []
    
    def extract_text(self, image: np.ndarray, preprocess: bool = True) -> Tuple[str, float]:
        """
        Extract text from a license plate image using multiple preprocessing pipelines.
        
        Args:
            image: Input plate image (BGR or grayscale)
            preprocess: Whether to apply preprocessing
            
        Returns:
            Tuple of (extracted_text, confidence)
        """
        if image is None or image.size == 0:
            return "", 0.0
        
        img_height = image.shape[0]
        
        best_text = ""
        best_confidence = 0.0
        best_score = 0.0
        
        # Define preprocessing pipelines to try
        pipelines = [
            ("minimal", self.preprocess_pipeline_minimal),
            ("standard", self.preprocess_pipeline_standard),
            ("light_plate", self.preprocess_pipeline_light_plate),
            ("dark_plate", self.preprocess_pipeline_dark_plate),
            ("aggressive", self.preprocess_pipeline_aggressive),
            ("original", lambda x: x),
        ]
        
        for pipeline_name, pipeline_func in pipelines:
            try:
                # Apply preprocessing
                processed = pipeline_func(image)
                
                # Run OCR
                results = self._run_ocr_on_image(processed)
                
                if not results:
                    continue
                
                # Filter by vertical position
                proc_height = processed.shape[0] if len(processed.shape) == 2 else processed.shape[0]
                filtered_results = self.filter_by_vertical_position(results, proc_height, top_portion=0.70)
                
                if not filtered_results:
                    filtered_results = results
                
                # Sort results by position
                sorted_results = self.sort_results_by_position(filtered_results)
                
                # Combine detected text
                texts = []
                confidences = []
                
                for detection in sorted_results:
                    bbox, text, confidence = detection
                    
                    if confidence < 0.2:
                        continue
                    
                    if self.is_expiry_date(text):
                        continue
                    
                    clean_text = text.strip().upper()
                    
                    if len(clean_text) < 1:
                        continue
                    
                    texts.append(clean_text)
                    confidences.append(confidence)
                
                if texts:
                    combined_text = " ".join(texts)
                    avg_confidence = sum(confidences) / len(confidences)
                    
                    # Score based on confidence and validity
                    looks_valid = self._looks_like_plate(combined_text)
                    score = avg_confidence * (1.3 if looks_valid else 1.0)
                    
                    # Track best result
                    if score > best_score:
                        best_text = combined_text
                        best_confidence = avg_confidence
                        best_score = score
                        
            except Exception as e:
                print(f"Pipeline '{pipeline_name}' error: {e}")
                continue
        
        return best_text, best_confidence
    
    def extract_text_with_alternatives(
        self, 
        image: np.ndarray
    ) -> list[Tuple[str, float]]:
        """Extract text with multiple preprocessing methods."""
        results = []
        
        text, conf = self.extract_text(image, preprocess=True)
        if text:
            results.append((text, conf))
        
        # Try inverted image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        inverted = cv2.bitwise_not(gray)
        text, conf = self.extract_text(inverted, preprocess=False)
        if text:
            results.append((text, conf))
        
        return results
