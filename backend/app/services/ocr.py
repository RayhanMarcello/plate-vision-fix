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
        Improved to handle tilted license plates better.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold - try both normal and inverted
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find coordinates of non-zero pixels
        coords = np.column_stack(np.where(thresh > 0))
        
        if len(coords) < 10:
            return image
        
        # Get the minimum area rectangle
        try:
            angle = cv2.minAreaRect(coords)[-1]
            
            # Adjust angle to get rotation direction
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
            
            # Correct if angle is significant (increased range to 30 degrees)
            if abs(angle) > 0.5 and abs(angle) < 30:
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # Calculate new bounding box size to avoid cropping
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_w = int((h * sin) + (w * cos))
                new_h = int((h * cos) + (w * sin))
                
                # Adjust the rotation matrix
                M[0, 2] += (new_w / 2) - center[0]
                M[1, 2] += (new_h / 2) - center[1]
                
                rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
                return rotated
        except:
            pass
        
        return image
    
    def deskew_hough(self, image: np.ndarray) -> np.ndarray:
        """
        Alternative deskew using Hough line detection.
        Better for plates with clear edges.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is None or len(lines) == 0:
            return image
        
        # Calculate average angle from detected lines
        angles = []
        for line in lines[:10]:  # Use top 10 lines
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            if -45 < angle < 45:
                angles.append(angle)
        
        if not angles:
            return image
        
        # Use median angle for rotation
        median_angle = np.median(angles)
        
        if abs(median_angle) > 0.5 and abs(median_angle) < 30:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            return rotated
        
        return image
    
    def perspective_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply perspective correction to straighten a skewed plate.
        Uses contour detection to find plate boundary and applies 4-point transform.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Try multiple threshold methods
        methods = [
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        ]
        
        for method in methods:
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, method)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            
            # Get minimum area rectangle
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.float32)
            
            # Get width and height from rect
            width = int(rect[1][0])
            height = int(rect[1][1])
            
            if width < 10 or height < 10:
                continue
            
            # Ensure width > height (plate is horizontal)
            if width < height:
                width, height = height, width
            
            # Order points: top-left, top-right, bottom-right, bottom-left
            # Sort by y first
            sorted_by_y = box[np.argsort(box[:, 1])]
            top_points = sorted_by_y[:2]
            bottom_points = sorted_by_y[2:]
            
            # Sort top by x
            top_points = top_points[np.argsort(top_points[:, 0])]
            # Sort bottom by x
            bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
            
            src_pts = np.array([
                top_points[0],      # top-left
                top_points[1],      # top-right
                bottom_points[1],   # bottom-right
                bottom_points[0]    # bottom-left
            ], dtype=np.float32)
            
            # Destination points
            dst_pts = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)
            
            # Get perspective transform matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            # Apply perspective transform
            warped = cv2.warpPerspective(image, M, (width, height),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
            
            if warped.size > 0:
                return warped
        
        return image
    
    def preprocess_tilted_dark_plate(self, image: np.ndarray) -> np.ndarray:
        """
        Special preprocessing for tilted dark plates.
        Combines perspective correction with dark plate enhancement.
        """
        # First apply perspective correction
        corrected = self.perspective_correction(image)
        
        # Resize for better OCR
        corrected = self.resize_image(corrected, target_height=150)
        
        # Convert to grayscale
        if len(corrected.shape) == 3:
            gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        else:
            gray = corrected.copy()
        
        # Invert for dark plates (white text on black background)
        inverted = cv2.bitwise_not(gray)
        
        # Strong CLAHE
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(inverted)
        
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 8
        )
        
        # Morphological closing to connect broken characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Small dilation to thicken characters
        kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(closed, kernel_d, iterations=1)
        
        return dilated
    
    def preprocess_tilted_white_plate(self, image: np.ndarray) -> np.ndarray:
        """
        Special preprocessing for tilted white plates.
        Combines perspective correction with white plate enhancement.
        """
        # First apply perspective correction
        corrected = self.perspective_correction(image)
        
        # Resize for better OCR
        corrected = self.resize_image(corrected, target_height=150)
        
        # Convert to grayscale
        if len(corrected.shape) == 3:
            gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        else:
            gray = corrected.copy()
        
        # CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Gaussian + Median blur
        gaussian = cv2.GaussianBlur(enhanced, (3, 3), 0)
        blurred = cv2.medianBlur(gaussian, 3)
        
        # Adaptive threshold (inverted for white plate)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10
        )
        
        # Morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return closed
    
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
    
    # ==================== NEW ENHANCED PREPROCESSING ====================
    
    def gamma_correction(self, image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
        """
        Apply gamma correction to brighten dark images or darken bright images.
        gamma > 1 = brighter, gamma < 1 = darker
        """
        if len(image.shape) == 3:
            img = image.copy()
        else:
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                          for i in np.arange(0, 256)]).astype("uint8")
        
        return cv2.LUT(img, table)
    
    def top_hat_transform(self, image: np.ndarray) -> np.ndarray:
        """
        Extract bright regions (white text) from dark background using top-hat.
        Great for dark plates with white text.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create structuring element - sized for typical character dimensions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        
        # White top-hat (tophat) extracts bright objects on dark background
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # Enhance the result
        enhanced = cv2.add(gray, tophat)
        
        return enhanced
    
    def black_hat_transform(self, image: np.ndarray) -> np.ndarray:
        """
        Extract dark regions (black text) from light background using black-hat.
        Great for white/light plates with dark text.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        
        # Black top-hat (blackhat) extracts dark objects on light background
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Subtract from original to enhance dark text
        enhanced = cv2.subtract(gray, blackhat)
        
        return enhanced
    
    def edge_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance edges using Sobel operators for clearer character boundaries.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur first
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Sobel operators
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.uint8(sobel / sobel.max() * 255)
        
        # Add edges to original
        enhanced = cv2.addWeighted(gray, 0.7, sobel, 0.3, 0)
        
        return enhanced
    
    def sauvola_binarization(self, image: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
        """
        Sauvola binarization - better for uneven lighting than Otsu.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Ensure window size is odd
        if window_size % 2 == 0:
            window_size += 1
        
        # Calculate local mean and standard deviation
        mean = cv2.blur(gray.astype(np.float64), (window_size, window_size))
        mean_sq = cv2.blur(gray.astype(np.float64)**2, (window_size, window_size))
        std = np.sqrt(mean_sq - mean**2)
        
        # Sauvola threshold
        R = 128  # Dynamic range of standard deviation
        threshold = mean * (1 + k * (std / R - 1))
        
        # Apply threshold
        binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
        
        return binary
    
    def character_separation(self, image: np.ndarray) -> np.ndarray:
        """
        Improve character separation using morphological operations.
        Helps prevent characters from merging.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Small erosion to separate touching characters
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        eroded = cv2.erode(binary, kernel_erode, iterations=1)
        
        # Small dilation to restore character thickness
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        dilated = cv2.dilate(eroded, kernel_dilate, iterations=1)
        
        return dilated
    
    def detect_plate_type(self, image: np.ndarray) -> str:
        """
        Detect if plate is dark (black background) or light (white background).
        Returns 'dark', 'light', or 'unknown'.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Calculate mean brightness
        avg_brightness = np.mean(gray)
        
        # Check for bimodal distribution (typical for plates)
        dark_pixels = np.sum(hist[:100])
        light_pixels = np.sum(hist[155:])
        
        if avg_brightness < 100:
            return 'dark'
        elif avg_brightness > 150:
            return 'light'
        elif dark_pixels > light_pixels * 1.5:
            return 'dark'
        elif light_pixels > dark_pixels * 1.5:
            return 'light'
        else:
            return 'unknown'
    
    # ==================== STRUCTURED PREPROCESSING PIPELINE ====================
    
    def structured_preprocess(self, image: np.ndarray, for_dark_plate: bool = False) -> np.ndarray:
        """
        Structured preprocessing pipeline as requested:
        1. Grayscale conversion
        2. Noise Reduction (Gaussian + Median Blur)
        3. Thresholding (Otsu + Adaptive)
        4. Morphological operations
        """
        # Step 0: Resize for better OCR
        img = self.resize_image(image, target_height=150)
        
        # Step 1: GRAYSCALE CONVERSION
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Step 2: NOISE REDUCTION
        # Gaussian blur for general noise
        gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
        # Median blur for salt-and-pepper noise
        denoised = cv2.medianBlur(gaussian, 3)
        
        # Step 3: THRESHOLDING
        # Try Otsu first
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Also try Adaptive for comparison
        adaptive = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 10
        )
        
        # Choose based on plate type
        if for_dark_plate:
            # For dark plates, invert
            binary = cv2.bitwise_not(otsu)
        else:
            # For light plates, check if text is dark (should be inverted)
            if np.mean(otsu) > 127:
                binary = cv2.bitwise_not(otsu)
            else:
                binary = otsu
        
        # Step 4: MORPHOLOGICAL OPERATIONS
        # Closing to fill small holes in characters
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        
        # Opening to remove small noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        
        # Optional: Dilation to thicken characters slightly
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        final = cv2.dilate(opened, kernel_dilate, iterations=1)
        
        return final
    
    def structured_preprocess_v2(self, image: np.ndarray) -> np.ndarray:
        """
        Alternative structured pipeline with stronger processing.
        """
        # Resize
        img = self.resize_image(image, target_height=180)
        
        # Grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # CLAHE for better contrast before noise reduction
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Noise reduction with bilateral filter (preserves edges better)
        denoised = cv2.bilateralFilter(enhanced, 11, 75, 75)
        
        # Combined blur
        gaussian = cv2.GaussianBlur(denoised, (5, 5), 0)
        median = cv2.medianBlur(gaussian, 3)
        
        # Thresholding - adaptive works better for uneven lighting
        binary = cv2.adaptiveThreshold(
            median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 15
        )
        
        # Morphology - more aggressive cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
        
        return binary
    
    def structured_preprocess_v3(self, image: np.ndarray) -> np.ndarray:
        """
        Third variant with edge preservation focus.
        """
        img = self.resize_image(image, target_height=160)
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Sharpen first to enhance edges
        kernel_sharpen = np.array([[-1, -1, -1],
                                    [-1,  9, -1],
                                    [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
        
        # Light gaussian blur
        blurred = cv2.GaussianBlur(sharpened, (3, 3), 0)
        
        # Otsu threshold
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if needed
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    # ==================== CHARACTER CORRECTION ====================
    
    # Indonesian license plate region codes
    INDONESIAN_REGION_CODES = [
        'A', 'B', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'W', 'Z',
        'AA', 'AB', 'AD', 'AE', 'AG', 'BA', 'BB', 'BD', 'BE', 'BG', 'BH', 'BK', 'BL', 'BM',
        'BN', 'BP', 'BR', 'BS', 'BT', 'DA', 'DB', 'DC', 'DD', 'DE', 'DG', 'DH', 'DK', 'DL',
        'DM', 'DN', 'DP', 'DR', 'DS', 'DT', 'DW', 'EA', 'EB', 'ED', 'EE', 'EG', 'EP',
        'GA', 'KB', 'KH', 'KT', 'KU', 'PA', 'PB', 'PG'
    ]
    
    # Common OCR misreadings
    OCR_CORRECTIONS = {
        # Letters that look like numbers
        '0': 'O', 'O': 'O',
        '1': 'I', 'I': 'I',
        '2': 'Z', 
        '5': 'S',
        '8': 'B',
        '6': 'G',
        # Numbers that look like letters
        'O': '0',
        'I': '1',
        'Z': '2',
        'S': '5',
        'B': '8',
        'G': '6',
        # Other common mistakes
        'Q': 'O',
        'D': 'D',
    }
    
    def correct_indonesian_plate(self, text: str) -> str:
        """
        Apply character correction for Indonesian license plates.
        Format: [Region Code 1-2 letters] [Numbers 1-4 digits] [Letters 1-3]
        Example: B 1234 ABC, AB 12 CD
        """
        if not text:
            return text
        
        # Remove spaces and clean
        cleaned = text.replace(' ', '').upper()
        
        if len(cleaned) < 4:
            return text
        
        # Try to identify structure
        result = []
        
        # Identify parts
        # First part should be 1-2 letters (region code)
        # Middle part should be 1-4 numbers
        # Last part should be 1-3 letters
        
        i = 0
        part = 'region'  # region -> number -> suffix
        
        region_chars = []
        number_chars = []
        suffix_chars = []
        
        for char in cleaned:
            if part == 'region':
                # First 1-2 characters should be letters
                if char.isalpha() or char in '068':
                    # Correct numbers that should be letters in region
                    if char == '0':
                        region_chars.append('O')
                    elif char == '6':
                        region_chars.append('G')
                    elif char == '8':
                        region_chars.append('B')
                    else:
                        region_chars.append(char)
                    
                    if len(region_chars) >= 2:
                        part = 'number'
                elif char.isdigit():
                    # If we hit a digit, we might have only 1 letter region
                    # or the first char was misread
                    if len(region_chars) == 0:
                        # First char should be letter, try to correct
                        if char == '0':
                            region_chars.append('O')
                        elif char == '6':
                            region_chars.append('G')
                        elif char == '8':
                            region_chars.append('B')
                        elif char == '1':
                            region_chars.append('I')
                        else:
                            region_chars.append(char)
                    else:
                        # Already have region, this starts numbers
                        number_chars.append(char)
                        part = 'number'
                        
            elif part == 'number':
                # Middle part should be numbers (1-4 digits)
                if char.isdigit():
                    # Only add up to 4 digits
                    if len(number_chars) < 4:
                        number_chars.append(char)
                    else:
                        # More than 4 digits means this should be suffix
                        # Convert last digit if it looks like a letter
                        if char == '0':
                            suffix_chars.append('O')
                        elif char == '8':
                            suffix_chars.append('B')
                        elif char == '6':
                            suffix_chars.append('G')
                        else:
                            suffix_chars.append(char)
                        part = 'suffix'
                elif char.isalpha():
                    # If we have at least 1 number, letters start the suffix
                    if len(number_chars) >= 1:
                        # This is the start of suffix - keep as letter
                        suffix_chars.append(char)
                        part = 'suffix'
                    else:
                        # No numbers yet, try to correct letter to number
                        if char == 'O':
                            number_chars.append('0')
                        elif char == 'I' or char == 'L':
                            number_chars.append('1')
                        elif char == 'Z':
                            number_chars.append('2')
                        elif char == 'S':
                            number_chars.append('5')
                        elif char == 'B':
                            number_chars.append('8')
                        elif char == 'G':
                            number_chars.append('6')
                        else:
                            # Unknown letter with no numbers - might be suffix
                            suffix_chars.append(char)
                            part = 'suffix'
                        
            elif part == 'suffix':
                # Last part should be letters
                if char.isalpha():
                    suffix_chars.append(char)
                elif char.isdigit():
                    # Correct numbers that should be letters in suffix
                    if char == '0':
                        suffix_chars.append('O')
                    elif char == '1':
                        suffix_chars.append('I')
                    elif char == '6':
                        suffix_chars.append('G')
                    elif char == '8':
                        suffix_chars.append('B')
                    else:
                        suffix_chars.append(char)
        
        # Reconstruct with proper spacing
        region = ''.join(region_chars)
        numbers = ''.join(number_chars)
        suffix = ''.join(suffix_chars)
        
        # IMPORTANT: Limit suffix to max 3 characters (Indonesian plate format)
        # This prevents false detection of extra characters like 'J' at the end
        if len(suffix) > 3:
            suffix = suffix[:3]
        
        # Validate and correct region code
        if region and region not in self.INDONESIAN_REGION_CODES:
            # Try single letter if double didn't match
            if len(region) == 2 and region[0] in self.INDONESIAN_REGION_CODES:
                # Move second char to numbers if it's a number-like letter
                second = region[1]
                region = region[0]
                if second in 'OIZSBG':
                    if second == 'O':
                        numbers = '0' + numbers
                    elif second == 'I':
                        numbers = '1' + numbers
                    elif second == 'Z':
                        numbers = '2' + numbers
                    elif second == 'S':
                        numbers = '5' + numbers
                    elif second == 'B':
                        numbers = '8' + numbers
                    elif second == 'G':
                        numbers = '6' + numbers
        
        # Format result
        parts = []
        if region:
            parts.append(region)
        if numbers:
            parts.append(numbers)
        if suffix:
            parts.append(suffix)
        
        return ' '.join(parts) if parts else text
    
    def apply_character_correction(self, text: str, confidence: float) -> Tuple[str, float]:
        """
        Apply character correction and potentially boost confidence.
        """
        if not text or confidence < 0.1:
            return text, confidence
        
        # Apply Indonesian plate correction
        corrected = self.correct_indonesian_plate(text)
        
        # Check if correction made the plate valid
        cleaned = corrected.replace(' ', '')
        
        # Indonesian plate pattern
        pattern = re.compile(r'^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$')
        
        # Boost confidence if correction resulted in valid format
        if pattern.match(cleaned):
            # Valid format, boost confidence slightly
            new_confidence = min(confidence * 1.15, 0.99)
            return corrected, new_confidence
        
        return corrected, confidence
    
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
    
    def preprocess_pipeline_dark_enhanced(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced pipeline for dark plates (black background, white text).
        Uses top-hat transform and gamma correction for better results.
        """
        # Resize for better OCR
        img = self.resize_image(image, target_height=120)
        
        # Apply gamma correction to brighten
        img = self.gamma_correction(img, gamma=1.8)
        
        # Top-hat transform to extract white text
        enhanced = self.top_hat_transform(img)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(enhanced)
        
        # Denoise
        enhanced = cv2.fastNlMeansDenoising(enhanced, None, 8, 7, 21)
        
        # Sharpen
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 2)
        sharpened = cv2.addWeighted(enhanced, 1.8, blurred, -0.8, 0)
        
        # Adaptive threshold for uneven lighting
        binary = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 4
        )
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    def preprocess_pipeline_dark_tophat(self, image: np.ndarray) -> np.ndarray:
        """
        Alternative dark plate pipeline using strong top-hat transform.
        """
        img = self.resize_image(image, target_height=100)
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Strong top-hat with larger kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 7))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # Normalize
        tophat = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
        
        # Threshold
        _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)
        
        return binary
    
    def preprocess_pipeline_light_plate(self, image: np.ndarray) -> np.ndarray:
        """Pipeline for light plates with dark text."""
        img = self.resize_image(image, target_height=120)
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(enhanced, 11, 50, 50)
        
        # Otsu threshold
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if needed (dark text on light background should become white text on black)
        # Check if background is light
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        
        return binary
    
    def preprocess_pipeline_light_enhanced(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced pipeline for light/white plates with dark text.
        Optimized for better character recognition.
        """
        # Resize larger for better detail
        img = self.resize_image(image, target_height=150)
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Apply strong CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        
        # Sharpen the image
        kernel_sharpen = np.array([[-1, -1, -1],
                                    [-1,  9, -1],
                                    [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
        
        # Bilateral filter
        filtered = cv2.bilateralFilter(sharpened, 9, 75, 75)
        
        # Adaptive threshold - better for uneven lighting on white plates
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def preprocess_enhanced_white_plate(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing pipeline for white plates with dark text.
        Implements the following steps:
        1. Grayscale conversion
        2. Gaussian + Median Blur to reduce noise
        3. CLAHE for contrast enhancement
        4. Adaptive Thresholding (better than simple threshold)
        5. Morphological Closing to connect broken characters
        """
        # Step 0: Resize for better OCR
        img = self.resize_image(image, target_height=150)
        
        # Step 1: GRAYSCALE CONVERSION
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Step 2: GAUSSIAN + MEDIAN BLUR to reduce noise
        # Gaussian blur for general smoothing
        gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
        # Median blur to remove salt-and-pepper noise (preserves edges better)
        blurred = cv2.medianBlur(gaussian, 3)
        
        # Step 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Improves local contrast significantly
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Step 4: ADAPTIVE THRESHOLDING
        # Much better than Otsu for uneven lighting conditions
        # Using Gaussian method with larger block size for smoother results
        binary = cv2.adaptiveThreshold(
            enhanced, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,  # Invert: black text becomes white
            21,  # Block size (must be odd)
            10   # Constant subtracted from mean
        )
        
        # Step 5: MORPHOLOGICAL CLOSING to connect broken characters
        # Closing = Dilation followed by Erosion
        # This fills small holes and gaps in the characters
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        
        # Additional: Small opening to remove tiny noise specks
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        final = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        
        return final
    
    def preprocess_enhanced_white_plate_v2(self, image: np.ndarray) -> np.ndarray:
        """
        Alternative enhanced pipeline for white plates.
        Uses stronger parameters for difficult cases.
        """
        # Resize larger for more detail
        img = self.resize_image(image, target_height=180)
        
        # Grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Strong CLAHE first
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        
        # Bilateral filter - reduces noise while keeping edges sharp
        bilateral = cv2.bilateralFilter(enhanced, 11, 75, 75)
        
        # Gaussian + Median blur
        gaussian = cv2.GaussianBlur(bilateral, (5, 5), 0)
        blurred = cv2.medianBlur(gaussian, 3)
        
        # Adaptive threshold with larger block
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,  # Larger block size
            12   # Higher constant
        )
        
        # Strong morphological closing
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        
        # Light dilation to thicken characters slightly
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(closed, kernel_dilate, iterations=1)
        
        # Opening to clean noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        final = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel_open)
        
        return final
    
    def preprocess_for_thin_chars(self, image: np.ndarray) -> np.ndarray:
        """
        Special preprocessing to enhance detection of thin characters like 'I', '1', 'L'.
        Uses horizontal dilation to thicken thin vertical strokes.
        """
        # Resize to larger height for better detail
        img = self.resize_image(image, target_height=200)
        
        # Grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Strong CLAHE for maximum contrast
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            8
        )
        
        # KEY: Horizontal dilation to thicken thin vertical characters like 'I'
        # Using a horizontal kernel (wider than tall) to expand thin strokes
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        dilated_h = cv2.dilate(binary, kernel_h, iterations=1)
        
        # Small vertical dilation to connect any broken parts
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        dilated_v = cv2.dilate(dilated_h, kernel_v, iterations=1)
        
        # Morphological closing to fill gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(dilated_v, cv2.MORPH_CLOSE, kernel_close)
        
        return closed
    
    
    def preprocess_pipeline_light_aggressive(self, image: np.ndarray) -> np.ndarray:
        """
        Aggressive pipeline for difficult white plates.
        Uses multiple enhancement steps.
        """
        # Resize 
        img = self.resize_image(image, target_height=140)
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Gamma correction - darken slightly
        inv_gamma = 1.0 / 0.7
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                          for i in np.arange(0, 256)]).astype("uint8")
        darkened = cv2.LUT(gray, table)
        
        # Strong CLAHE
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(darkened)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # Morphological gradient to find edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(denoised, cv2.MORPH_GRADIENT, kernel)
        
        # Add gradient to enhance edges
        edge_enhanced = cv2.addWeighted(denoised, 0.8, gradient, 0.2, 0)
        
        # Adaptive threshold with larger block size
        binary = cv2.adaptiveThreshold(
            edge_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 8
        )
        
        # Clean up noise
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)
        
        # Small dilation to thicken characters
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        binary = cv2.dilate(binary, kernel_dilate, iterations=1)
        
        return binary
    
    def preprocess_pipeline_light_canny(self, image: np.ndarray) -> np.ndarray:
        """
        Edge-based pipeline for white plates using Canny edge detection.
        """
        img = self.resize_image(image, target_height=120)
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Canny edge detection
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Dilate edges to make them thicker
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Close gaps
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        
        return closed
    
    def remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """
        Remove shadows using illumination normalization (division by background).
        """
        if len(image.shape) == 3:
            planes = cv2.split(image)
            result_planes = []
            
            for plane in planes:
                # 1. Dilate to remove text features (get background)
                dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
                
                # 2. Median blur to smooth the background
                bg_img = cv2.medianBlur(dilated_img, 21)
                
                # 3. Calculate difference (255 - abs(plane - bg))
                # diff_img = 255 - cv2.absdiff(plane, bg_img)
                
                # 3. Normalize: raw / background
                # Avoid division by zero
                diff_img = 255 - cv2.absdiff(plane, bg_img)
                norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                
                result_planes.append(norm_img)
                
            result = cv2.merge(result_planes)
        else:
            # Gray scale version
            dilated_img = cv2.dilate(image, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(image, bg_img)
            result = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            
        return result


    def remove_small_noise(self, image: np.ndarray, min_area: int = 10, max_area: int = 150) -> np.ndarray:
        """
        Remove small noise (like screws/bolts/dirt) from binary image.
        """
        # Ensure binary
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            binary = image.copy()
            
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.ones(binary.shape, dtype=np.uint8) * 255
        
        cleaned = binary.copy()
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # If contour is too small (noise/screw), remove it (shape it black)
            if area < min_area:
                cv2.drawContours(cleaned, [cnt], -1, 0, -1)
            # Optional: Remove very large blobs that aren't text
            # elif area > 5000:
            #     cv2.drawContours(cleaned, [cnt], -1, 0, -1)
                
        return cleaned

    def preprocess_pipeline_shadows(self, image: np.ndarray) -> np.ndarray:
        """
        Pipeline specifically designed for images with strong shadows/variable lighting.
        Includes Noise Bolt Removal.
        """
        img = self.resize_image(image, target_height=120)
        
        # 1. Remove shadows
        img = self.remove_shadows(img)
        
        # 2. Convert to gray
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        # 3. Enhance contrast (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 4. Sharpen
        enhanced = self.sharpen_image(enhanced)
        
        # 5. Otsu thresholding
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 6. Clean up small noise (screws/bolts)
        # Invert first if needed (we want white text on black background for finding contours properly)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
            
        # Use smaller min_area to avoid removing parts of letters (like 'H')
        binary = self.remove_small_noise(binary, min_area=15)
        
        # 7. Morphological cleanup - USE CLOSE instead of OPEN to connect broken characters
        # "H" was breaking into "I I" because of Opening (erosion). Closing (dilation) fixes this.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
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
    
    def _run_ocr_on_image(self, image: np.ndarray, use_strict: bool = False) -> List:
        """
        Run EasyOCR on an image and return results.
        
        Args:
            image: Preprocessed image
            use_strict: If True, use stricter parameters for cleaner images
        """
        try:
            if use_strict:
                # Stricter parameters for clean/preprocessed images
                results = self.reader.readtext(
                    image,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    paragraph=False,
                    min_size=10,
                    text_threshold=0.6,
                    low_text=0.4,
                    contrast_ths=0.2,
                    adjust_contrast=0.7,
                    width_ths=0.8,
                    decoder='beamsearch',
                    beamWidth=5,
                )
            else:
                # Standard parameters
                results = self.reader.readtext(
                    image,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    paragraph=False,
                    min_size=8,
                    text_threshold=0.55,
                    low_text=0.35,
                    contrast_ths=0.15,
                    adjust_contrast=0.6,
                    width_ths=0.7,
                )
            return results
        except Exception as e:
            print(f"OCR error: {e}")
            return []
    
    def extract_text(self, image: np.ndarray, preprocess: bool = True) -> Tuple[str, float]:
        """
        Extract text from a license plate image.
        Uses improved preprocessing to reduce false character detection.
        
        Args:
            image: Input plate image (BGR or grayscale) - already cropped plate
            preprocess: Whether to apply preprocessing
            
        Returns:
            Tuple of (extracted_text, confidence)
        """
        if image is None or image.size == 0:
            return "", 0.0
        
        best_text = ""
        best_confidence = 0.0
        
        # Resize to standard height
        img = self.resize_image(image.copy(), target_height=80)
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # List of preprocessing approaches to try
        preprocessed_images = []
        
        # Approach 1: Original grayscale (minimal processing)
        preprocessed_images.append(("original", gray))
        
        # Approach 2: Bilateral filter (best for noise reduction while keeping edges)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        preprocessed_images.append(("bilateral", bilateral))
        
        # Approach 3: CLAHE + bilateral for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_bilateral = cv2.bilateralFilter(enhanced, 9, 50, 50)
        preprocessed_images.append(("clahe_bilateral", enhanced_bilateral))
        
        # Approach 4: Binary with Otsu (from bilateral filtered)
        _, binary_otsu = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Clean up small noise with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary_clean = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, kernel)
        preprocessed_images.append(("binary_clean", binary_clean))
        
        # Approach 5: NEW - Enhanced white plate preprocessing
        # Grayscale → Gaussian/Median Blur → CLAHE → Adaptive Threshold → Morphological Closing
        enhanced_white = self.preprocess_enhanced_white_plate(image.copy())
        preprocessed_images.append(("enhanced_white", enhanced_white))
        
        # Approach 6: NEW - Enhanced white plate v2 (stronger parameters)
        enhanced_white_v2 = self.preprocess_enhanced_white_plate_v2(image.copy())
        preprocessed_images.append(("enhanced_white_v2", enhanced_white_v2))
        
        # Approach 7: NEW - Special preprocessing for thin characters like 'I', '1', 'L'
        # Uses horizontal dilation to thicken thin vertical strokes
        thin_chars = self.preprocess_for_thin_chars(image.copy())
        preprocessed_images.append(("thin_chars", thin_chars))
        
        # Approach 8: DARK PLATE - Simple invert for black plates with white text
        inverted = cv2.bitwise_not(gray)
        preprocessed_images.append(("inverted", inverted))
        
        # Approach 9: DARK PLATE - Inverted with CLAHE
        clahe_inv = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        inverted_enhanced = clahe_inv.apply(inverted)
        preprocessed_images.append(("inverted_clahe", inverted_enhanced))
        
        # Approach 10: DARK PLATE - Enhanced dark plate pipeline
        dark_enhanced = self.preprocess_pipeline_dark_enhanced(image.copy())
        preprocessed_images.append(("dark_enhanced", dark_enhanced))
        
        # Approach 11: DARK PLATE - Top-hat transform for dark plates
        dark_tophat = self.preprocess_pipeline_dark_tophat(image.copy())
        preprocessed_images.append(("dark_tophat", dark_tophat))
        
        # Approach 12: DESKEWED - Original with deskewing for tilted plates
        deskewed = self.deskew_image(image.copy())
        if len(deskewed.shape) == 3:
            deskewed_gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
        else:
            deskewed_gray = deskewed
        preprocessed_images.append(("deskewed", deskewed_gray))
        
        # Approach 13: DESKEWED + CLAHE - Deskewed with contrast enhancement
        clahe_desk = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        deskewed_enhanced = clahe_desk.apply(deskewed_gray)
        preprocessed_images.append(("deskewed_clahe", deskewed_enhanced))
        
        # Approach 14: DESKEWED + Adaptive Threshold
        deskewed_blur = cv2.GaussianBlur(deskewed_enhanced, (3, 3), 0)
        deskewed_binary = cv2.adaptiveThreshold(
            deskewed_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10
        )
        preprocessed_images.append(("deskewed_binary", deskewed_binary))
        
        # Approach 15: DESKEWED DARK - Deskewed and inverted for dark plates
        deskewed_inverted = cv2.bitwise_not(deskewed_gray)
        deskewed_inv_clahe = clahe_desk.apply(deskewed_inverted)
        preprocessed_images.append(("deskewed_dark", deskewed_inv_clahe))
        
        # Approach 16: PERSPECTIVE CORRECTED - For tilted dark plates
        tilted_dark = self.preprocess_tilted_dark_plate(image.copy())
        preprocessed_images.append(("tilted_dark", tilted_dark))
        
        # Approach 17: PERSPECTIVE CORRECTED - For tilted white plates
        tilted_white = self.preprocess_tilted_white_plate(image.copy())
        preprocessed_images.append(("tilted_white", tilted_white))
        
        # Approach 18: PERSPECTIVE + simple grayscale
        perspective = self.perspective_correction(image.copy())
        if len(perspective.shape) == 3:
            persp_gray = cv2.cvtColor(perspective, cv2.COLOR_BGR2GRAY)
        else:
            persp_gray = perspective
        persp_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        persp_enhanced = persp_clahe.apply(persp_gray)
        preprocessed_images.append(("perspective", persp_enhanced))
        
        # Approach 19: PERSPECTIVE + inverted for dark plates
        persp_inverted = cv2.bitwise_not(persp_gray)
        persp_inv_clahe = persp_clahe.apply(persp_inverted)
        preprocessed_images.append(("perspective_dark", persp_inv_clahe))
        
        # Approach 20: SHADOW REMOVAL - Explicit shadow removal pipeline
        shadow_removed = self.preprocess_pipeline_shadows(image.copy())
        preprocessed_images.append(("shadow_removed", shadow_removed))
        
        
        # Try each preprocessed image
        candidates = []
        
        print(f"--- Starting OCR Extraction on {len(preprocessed_images)} pipelines ---")
        
        for name, processed in preprocessed_images:
            try:
                # Run OCR with parameters optimized for thin characters like 'I'
                results = self.reader.readtext(
                    processed,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    paragraph=False,
                    min_size=5,            # Reduced to catch thin characters like 'I'
                    text_threshold=0.4,    # Lower = more sensitive text detection
                    low_text=0.3,          # Lower = better for thin characters
                    link_threshold=0.3,    # For linking text boxes
                    contrast_ths=0.1,      # Lower contrast threshold for thin chars
                    adjust_contrast=0.5,   # Contrast adjustment
                    mag_ratio=2.0,         # Higher magnification for thin chars
                    width_ths=0.5,         # Width threshold for character merging
                )
                
                if not results:
                    continue
                
                # Sort by position (left to right)
                sorted_results = self.sort_results_by_position(results)
                
                # Filter by vertical position (focus on main text)
                proc_height = processed.shape[0]
                filtered = []
                for detection in sorted_results:
                    bbox, text, conf = detection
                    y_coords = [p[1] for p in bbox]
                    center_y = sum(y_coords) / len(y_coords)
                    # Keep only top 65% of plate (main number)
                    if center_y < proc_height * 0.65:
                        filtered.append(detection)
                
                if not filtered:
                    filtered = sorted_results
                
                # Collect text with confidence filtering
                texts = []
                confidences = []
                
                for detection in filtered:
                    bbox, text, conf = detection
                    
                    # Base confidence threshold
                    if conf < 0.35:
                        continue
                    
                    # Skip expiry dates
                    if self.is_expiry_date(text):
                        continue
                    
                    clean = text.strip().upper()
                    
                    # Stricter threshold for single characters 
                    # (they are more likely to be noise)
                    if len(clean) == 1 and conf < 0.5:
                        continue
                    
                    # Skip common noise characters that appear at edges
                    if len(clean) == 1 and clean in 'JQ':
                        # These are rarely used in Indonesian plates
                        # Require higher confidence
                        if conf < 0.7:
                            continue
                    
                    if len(clean) >= 1:
                        texts.append(clean)
                        confidences.append(conf)
                
                if texts:
                    combined = " ".join(texts)
                    avg_conf = sum(confidences) / len(confidences)
                    
                    # Apply correction immediately to check validity
                    corrected_text, boost_conf = self.apply_character_correction(combined, avg_conf)
                    
                    # Check validity
                    cleaned = corrected_text.replace(' ', '')
                    match = re.match(r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$', cleaned)
                    is_valid = False
                    
                    if match:
                        region, numbers, suffix = match.groups()
                        # STRICT CHECK: Region code must exist in Indonesia
                        if region in self.INDONESIAN_REGION_CODES:
                            is_valid = True
                        else:
                            # Demote invalid regions (like "IL")
                            is_valid = False
                    
                    # Store candidate
                    candidates.append({
                        'text': corrected_text,
                        'original_text': combined,
                        'confidence': avg_conf,
                        'boost_conf': boost_conf,
                        'is_valid': is_valid,
                        'method': name,
                        'raw_length': len(cleaned)
                    })
                    print(f"Pipeline '{name}': '{corrected_text}' (conf={avg_conf:.2f}, valid={is_valid})")
                        
            except Exception as e:
                print(f"OCR '{name}' error: {e}")
                continue
        
        # === SELECTION LOGIC ===
        
        if not candidates:
            return "", 0.0
            
        # 1. Prioritize Valid Indonesian Plates
        valid_candidates = [c for c in candidates if c['is_valid']]
        
        if valid_candidates:
            # Sort by confidence
            valid_candidates.sort(key=lambda x: x['boost_conf'], reverse=True)
            best = valid_candidates[0]
            print(f"SELECTED VALID: '{best['text']}' from '{best['method']}'")
            return best['text'], best['boost_conf']
            
        # 2. Fallback: Look for "plausible" plates even if strictly invalid
        # e.g. "B 1234" (missing suffix) or "1234 ABC" (missing region)
        # But filter out obvious noise (single letters, very short strings)
        
        plausible_candidates = [
            c for c in candidates 
            if c['raw_length'] >= 3  # Ignore single/double characters
            and not (c['raw_length'] == 1) # Double check
        ]
        
        if plausible_candidates:
            # Sort by confidence
            plausible_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            best = plausible_candidates[0]
            print(f"SELECTED PLAUSIBLE: '{best['text']}' from '{best['method']}'")
            return best['text'], best['confidence']
            
        # 3. Last Resort: Just take the highest confidence non-empty result
        # But try to avoid single-letter noise if possible
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        best = candidates[0]
        print(f"SELECTED FALLBACK: '{best['text']}' from '{best['method']}'")
        
        return best['text'], best['confidence']
    
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