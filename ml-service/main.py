"""
PlateVision ML Service - Hugging Face Spaces
YOLO Detection + EasyOCR for Indonesian License Plates
"""
import base64
import cv2
import numpy as np
import re
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import easyocr

# Initialize FastAPI
app = FastAPI(
    title="PlateVision ML Service",
    description="License Plate Detection and OCR for Indonesian Plates",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class DetectionRequest(BaseModel):
    image: str  # Base64 encoded image

class DetectionResult(BaseModel):
    plate_number: str
    raw_text: str
    confidence: float
    is_valid: bool
    bbox: List[int]
    plate_image: str  # Base64 encoded cropped plate


# Global model instances
yolo_model: Optional[YOLO] = None
ocr_reader: Optional[easyocr.Reader] = None

INDONESIAN_REGION_CODES = [
    'A', 'B', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'W', 'Z',
    'AA', 'AB', 'AD', 'AE', 'AG', 'BA', 'BB', 'BD', 'BE', 'BG', 'BH', 'BK', 'BL', 'BM',
    'BN', 'BP', 'BR', 'BS', 'BT', 'DA', 'DB', 'DC', 'DD', 'DE', 'DG', 'DH', 'DK', 'DL',
    'DM', 'DN', 'DP', 'DR', 'DS', 'DT', 'DW', 'EA', 'EB', 'ED', 'EE', 'EG', 'EP',
    'GA', 'KB', 'KH', 'KT', 'KU', 'PA', 'PB', 'PG'
]


def load_models():
    """Load YOLO and EasyOCR models."""
    global yolo_model, ocr_reader
    
    if yolo_model is None:
        model_path = Path(__file__).parent / "best.pt"
        if model_path.exists():
            yolo_model = YOLO(str(model_path))
            print("YOLO model loaded")
        else:
            print(f"Warning: YOLO model not found at {model_path}")
    
    if ocr_reader is None:
        ocr_reader = easyocr.Reader(['en'], gpu=False)
        print("EasyOCR reader initialized")


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to OpenCV image."""
    # Remove data URI prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_bytes = base64.b64decode(base64_string)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def encode_image_base64(image: np.ndarray) -> str:
    """Encode OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_str}"


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Preprocess plate image for OCR."""
    # Resize
    h, w = image.shape[:2]
    target_height = 80
    scale = target_height / h
    new_w = int(w * scale)
    resized = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Bilateral filter
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return filtered


def validate_plate(text: str) -> tuple:
    """Validate and normalize Indonesian plate number."""
    if not text:
        return "", False
    
    # Clean text
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    if len(cleaned) < 4:
        return text, False
    
    # Check pattern: 1-2 letters + 1-4 digits + 0-3 letters
    pattern = r'^([A-Z]{1,2})(\d{1,4})([A-Z]{0,3})$'
    match = re.match(pattern, cleaned)
    
    if match:
        region = match.group(1)
        numbers = match.group(2)
        suffix = match.group(3)
        
        is_valid = region in INDONESIAN_REGION_CODES
        formatted = f"{region} {numbers} {suffix}".strip()
        
        return formatted, is_valid
    
    return text, False


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_models()


@app.post("/api/detect", response_model=List[DetectionResult])
async def detect_plates(request: DetectionRequest):
    """Detect license plates in uploaded image."""
    global yolo_model, ocr_reader
    
    if yolo_model is None or ocr_reader is None:
        load_models()
    
    if yolo_model is None:
        raise HTTPException(status_code=500, detail="YOLO model not loaded")
    
    try:
        # Decode image
        image = decode_base64_image(request.image)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Run YOLO detection
        results = yolo_model(image, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                confidence = float(box.conf[0])
                
                if confidence < 0.4:
                    continue
                
                # Get bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Add padding
                h, w = image.shape[:2]
                pad_x = int((x2 - x1) * 0.05)
                pad_y = int((y2 - y1) * 0.05)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                
                # Crop plate
                cropped = image[y1:y2, x1:x2].copy()
                
                if cropped.size == 0:
                    continue
                
                # Preprocess and run OCR
                processed = preprocess_for_ocr(cropped)
                ocr_results = ocr_reader.readtext(
                    processed,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    min_size=10,
                    text_threshold=0.5,
                    low_text=0.4
                )
                
                # Extract text
                if ocr_results:
                    raw_text = ' '.join([r[1] for r in ocr_results])
                    ocr_conf = sum([r[2] for r in ocr_results]) / len(ocr_results)
                else:
                    raw_text = ""
                    ocr_conf = 0.0
                
                # Validate plate
                plate_number, is_valid = validate_plate(raw_text)
                
                # Combined confidence
                final_conf = (confidence + ocr_conf) / 2
                
                # Encode cropped image
                plate_image = encode_image_base64(cropped)
                
                detections.append(DetectionResult(
                    plate_number=plate_number if plate_number else raw_text,
                    raw_text=raw_text,
                    confidence=round(final_conf, 4),
                    is_valid=is_valid,
                    bbox=[x1, y1, x2, y2],
                    plate_image=plate_image
                ))
        
        return detections
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "yolo_loaded": yolo_model is not None,
        "ocr_loaded": ocr_reader is not None
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "PlateVision ML Service",
        "version": "1.0.0",
        "endpoints": {
            "detect": "POST /api/detect",
            "health": "GET /health"
        }
    }
