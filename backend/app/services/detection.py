"""
YOLO Detection Service for License Plate Detection
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from ultralytics import YOLO
from ..config import get_settings

settings = get_settings()


class DetectionService:
    """Service for detecting license plates using YOLO model."""
    
    _instance: Optional["DetectionService"] = None
    _model: Optional[YOLO] = None
    
    def __new__(cls):
        """Singleton pattern for model loading efficiency."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the YOLO model if not already loaded."""
        if DetectionService._model is None:
            self._load_model()
    
    def _load_model(self):
        """Load the YOLO model from the configured path."""
        model_path = Path(__file__).parent.parent.parent.parent / "best.pt"
        print(f"Loading YOLO model from: {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        DetectionService._model = YOLO(str(model_path))
        print("YOLO model loaded successfully!")
    
    @property
    def model(self) -> YOLO:
        """Get the loaded YOLO model."""
        if DetectionService._model is None:
            self._load_model()
        return DetectionService._model
    
    def detect_plates(
        self, 
        image: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> List[Tuple[np.ndarray, List[int], float]]:
        """
        Detect license plates in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of tuples: (cropped_plate_image, bbox, confidence)
            bbox format: [x1, y1, x2, y2]
        """
        results = self.model(image, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for i, box in enumerate(boxes):
                confidence = float(box.conf[0])
                
                if confidence < confidence_threshold:
                    continue
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Ensure coordinates are within image bounds
                h, w = image.shape[:2]
                
                # Add padding around the plate (5% of plate dimensions)
                # This helps capture plates that are slightly cut off
                pad_x = int((x2 - x1) * 0.05)
                pad_y = int((y2 - y1) * 0.05)
                
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                
                # Crop the plate region
                cropped = image[y1:y2, x1:x2].copy()
                
                if cropped.size > 0:
                    detections.append((cropped, [x1, y1, x2, y2], confidence))
        
        return detections
    
    def detect_and_annotate(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, List[int], float]]]:
        """
        Detect plates and return annotated image with detections.
        
        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Tuple of (annotated_image, detections)
        """
        results = self.model(image, verbose=False)
        detections = []
        annotated_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            # Get annotated frame from YOLO
            annotated_image = result.plot()
            
            for box in boxes:
                confidence = float(box.conf[0])
                
                if confidence < confidence_threshold:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                h, w = image.shape[:2]
                
                # Add padding around the plate (5% of plate dimensions)
                pad_x = int((x2 - x1) * 0.05)
                pad_y = int((y2 - y1) * 0.05)
                
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                
                cropped = image[y1:y2, x1:x2].copy()
                
                if cropped.size > 0:
                    detections.append((cropped, [x1, y1, x2, y2], confidence))
        
        return annotated_image, detections
    
    def process_image_file(self, image_path: str) -> Tuple[np.ndarray, List[Tuple[np.ndarray, List[int], float]]]:
        """
        Process an image file and return detections.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (original_image, detections)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        detections = self.detect_plates(image)
        return image, detections
