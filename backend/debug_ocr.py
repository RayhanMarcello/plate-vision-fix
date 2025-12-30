import sys
import os
import cv2
import numpy as np

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("Attempting to import OCRService...")
    from app.services.ocr import OCRService
    print("Import successful.")

    print("Initializing OCRService...")
    ocr = OCRService()
    print("OCRService initialized.")

    # Create a dummy image (black image with some white text)
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    cv2.putText(img, "B 1234 ABC", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    print("Running extract_text...")
    text, conf = ocr.extract_text(img)
    print(f"Result: '{text}' with confidence {conf}")

except Exception as e:
    import traceback
    traceback.print_exc()
