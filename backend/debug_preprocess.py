"""
Debug script for OCR preprocessing
Saves intermediate images to see what's happening at each step
"""
import cv2
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.ocr import AdvancedPreprocessor, OCRService

def debug_preprocessing(image_path: str, output_dir: str = "debug_output"):
    """Debug the preprocessing pipeline by saving intermediate images."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Image loaded: {image.shape}")
    cv2.imwrite(f"{output_dir}/00_original.png", image)
    
    # Initialize preprocessor with different target heights
    for target_height in [80, 100, 120, 150]:
        print(f"\n=== Testing target_height={target_height} ===")
        preprocessor = AdvancedPreprocessor(target_height=target_height)
        
        # Step by step
        img = preprocessor.step1_resize(image.copy())
        cv2.imwrite(f"{output_dir}/01_resize_h{target_height}.png", img)
        
        gray = preprocessor.step2_grayscale(img)
        cv2.imwrite(f"{output_dir}/02_grayscale_h{target_height}.png", gray)
        
        denoised = preprocessor.step3_denoise_edge_preserving(gray)
        cv2.imwrite(f"{output_dir}/03_denoised_h{target_height}.png", denoised)
        
        enhanced = preprocessor.step4_enhance_contrast(denoised)
        cv2.imwrite(f"{output_dir}/04_enhanced_h{target_height}.png", enhanced)
        
        binary = preprocessor.step5_adaptive_threshold(enhanced)
        cv2.imwrite(f"{output_dir}/05_binary_h{target_height}.png", binary)
        
        closed = preprocessor.step6_morphological_closing(binary)
        cv2.imwrite(f"{output_dir}/06_closed_h{target_height}.png", closed)
        
        opened = preprocessor.step7_morphological_opening(closed)
        cv2.imwrite(f"{output_dir}/07_opened_h{target_height}.png", opened)
        
        # Final
        final, quality = preprocessor.process(image.copy())
        cv2.imwrite(f"{output_dir}/08_final_h{target_height}.png", final)
        print(f"Quality metrics: {quality}")
    
    # Try different CLAHE parameters
    print("\n=== Testing different CLAHE clipLimit ===")
    for clip_limit in [2.0, 3.0, 4.0, 5.0]:
        preprocessor = AdvancedPreprocessor(target_height=100)
        preprocessor.clahe_clip_limit = clip_limit
        final, quality = preprocessor.process(image.copy())
        cv2.imwrite(f"{output_dir}/clahe_{clip_limit}.png", final)
        
    # Try different adaptive threshold parameters
    print("\n=== Testing different adaptive threshold block sizes ===")
    for block_size in [11, 15, 21, 31, 41]:
        preprocessor = AdvancedPreprocessor(target_height=100)
        preprocessor.adaptive_block_size = block_size
        final, quality = preprocessor.process(image.copy())
        cv2.imwrite(f"{output_dir}/block_{block_size}.png", final)
        
    # Try Otsu instead of adaptive
    print("\n=== Testing Otsu threshold ===")
    preprocessor = AdvancedPreprocessor(target_height=100)
    img = preprocessor.step1_resize(image.copy())
    gray = preprocessor.step2_grayscale(img)
    denoised = preprocessor.step3_denoise_edge_preserving(gray)
    enhanced = preprocessor.step4_enhance_contrast(denoised)
    
    # Otsu threshold
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f"{output_dir}/otsu_normal.png", otsu)
    
    _, otsu_inv = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(f"{output_dir}/otsu_inv.png", otsu_inv)
    
    print(f"\nDebug images saved to: {output_dir}/")
    print("Please check the images to see which preprocessing works best.")
    
    # Now test OCR on best candidates
    print("\n=== Running OCR on preprocessed images ===")
    ocr = OCRService()
    
    # Test on a few variants
    test_images = [
        ("final_h100", f"{output_dir}/08_final_h100.png"),
        ("final_h120", f"{output_dir}/08_final_h120.png"),
        ("otsu_inv", f"{output_dir}/otsu_inv.png"),
    ]
    
    for name, path in test_images:
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            results = ocr.reader.readtext(
                img,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                paragraph=False,
                detail=1
            )
            print(f"\n{name}:")
            for bbox, text, conf in results:
                print(f"  '{text}' (conf={conf:.2f})")

if __name__ == "__main__":
    # Use the test image
    test_image = sys.argv[1] if len(sys.argv) > 1 else "test_plate.png"
    debug_preprocessing(test_image)
