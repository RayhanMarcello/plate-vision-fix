"""
PlateVision REST API Routes
"""
import base64
import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..database import get_db
from ..models.detection import DetectionResult, SourceType
from ..schemas.detection import (
    DetectionResultResponse,
    DetectionResultList,
    DetectionStatistics,
    PlateDetectionResult
)
from ..services.detection import DetectionService
from ..services.ocr import OCRService
from ..services.validator import PlateValidator

router = APIRouter(prefix="/api", tags=["detection"])


def get_detection_service() -> DetectionService:
    """Dependency to get detection service instance."""
    return DetectionService()


def get_ocr_service() -> OCRService:
    """Dependency to get OCR service instance."""
    return OCRService()


def image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 data URI string."""
    _, buffer = cv2.imencode('.jpg', image)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_str}"


@router.post("/detect/upload", response_model=list[PlateDetectionResult])
async def detect_from_upload(
    file: UploadFile = File(...),
    save_to_db: bool = Query(True, description="Save results to database"),
    db: Session = Depends(get_db),
    detection_service: DetectionService = Depends(get_detection_service),
    ocr_service: OCRService = Depends(get_ocr_service)
):
    """
    Upload an image and detect license plates.
    
    Returns list of detected plates with their OCR results.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read file content
    content = await file.read()
    
    # Convert to numpy array
    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Convert original image to base64
    original_base64 = image_to_base64(image)
    
    # Detect plates
    try:
        detections = detection_service.detect_plates(image)
        
        results = []
        for i, (cropped_plate, bbox, det_confidence) in enumerate(detections):
            # Run OCR on cropped plate
            raw_text, ocr_confidence = ocr_service.extract_text(cropped_plate)
            
            # Validate and normalize plate number
            plate_number, is_valid, region = PlateValidator.process_ocr_result(raw_text)
            
            # Combined confidence (average of detection and OCR)
            confidence = (det_confidence + ocr_confidence) / 2
            
            # Convert cropped plate to base64
            plate_base64 = image_to_base64(cropped_plate)
            
            # Create result object
            result = PlateDetectionResult(
                plate_number=plate_number if plate_number else raw_text,
                raw_ocr_text=raw_text,
                confidence=round(confidence, 4),
                is_valid=is_valid,
                bbox=bbox,
                cropped_image_data=plate_base64
            )
            results.append(result)
            
            # Save to database if requested
            if save_to_db and plate_number:
                db_record = DetectionResult(
                    plate_number=plate_number,
                    raw_ocr_text=raw_text,
                    confidence=confidence,
                    source_type=SourceType.UPLOAD,
                    image_data=plate_base64,
                    original_image_data=original_base64,
                    is_valid=is_valid
                )
                db.add(db_record)
        
        if save_to_db:
            db.commit()
        
        return results

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.get("/detections", response_model=DetectionResultList)
async def list_detections(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None, description="Search by plate number"),
    source_type: Optional[str] = Query(None, description="Filter by source: upload or camera"),
    is_valid: Optional[bool] = Query(None, description="Filter by validation status"),
    start_date: Optional[datetime] = Query(None, description="Filter from date"),
    end_date: Optional[datetime] = Query(None, description="Filter to date"),
    db: Session = Depends(get_db)
):
    """
    Get paginated list of detection results with optional filters.
    """
    query = db.query(DetectionResult)
    
    # Apply filters
    if search:
        query = query.filter(DetectionResult.plate_number.contains(search))
    
    if source_type:
        query = query.filter(DetectionResult.source_type == source_type)
    
    if is_valid is not None:
        query = query.filter(DetectionResult.is_valid == is_valid)
    
    if start_date:
        query = query.filter(DetectionResult.detected_at >= start_date)
    
    if end_date:
        query = query.filter(DetectionResult.detected_at <= end_date)
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    offset = (page - 1) * page_size
    items = query.order_by(DetectionResult.detected_at.desc()).offset(offset).limit(page_size).all()
    
    # Calculate total pages
    total_pages = (total + page_size - 1) // page_size
    
    return DetectionResultList(
        items=[DetectionResultResponse.model_validate(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )


@router.get("/detections/{detection_id}", response_model=DetectionResultResponse)
async def get_detection(detection_id: int, db: Session = Depends(get_db)):
    """Get a single detection result by ID."""
    result = db.query(DetectionResult).filter(DetectionResult.id == detection_id).first()
    
    if not result:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    return DetectionResultResponse.model_validate(result)


@router.delete("/detections/{detection_id}")
async def delete_detection(detection_id: int, db: Session = Depends(get_db)):
    """Delete a detection result by ID."""
    result = db.query(DetectionResult).filter(DetectionResult.id == detection_id).first()
    
    if not result:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    db.delete(result)
    db.commit()
    
    return {"message": "Detection deleted successfully"}


@router.get("/statistics", response_model=DetectionStatistics)
async def get_statistics(db: Session = Depends(get_db)):
    """Get detection statistics."""
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago = today - timedelta(days=7)
    
    # Total counts
    total = db.query(func.count(DetectionResult.id)).scalar() or 0
    valid = db.query(func.count(DetectionResult.id)).filter(DetectionResult.is_valid == True).scalar() or 0
    invalid = total - valid
    
    # Source counts
    upload_count = db.query(func.count(DetectionResult.id)).filter(
        DetectionResult.source_type == SourceType.UPLOAD
    ).scalar() or 0
    camera_count = db.query(func.count(DetectionResult.id)).filter(
        DetectionResult.source_type == SourceType.CAMERA
    ).scalar() or 0
    
    # Average confidence
    avg_conf = db.query(func.avg(DetectionResult.confidence)).scalar() or 0.0
    
    # Time-based counts
    today_count = db.query(func.count(DetectionResult.id)).filter(
        DetectionResult.detected_at >= today
    ).scalar() or 0
    
    week_count = db.query(func.count(DetectionResult.id)).filter(
        DetectionResult.detected_at >= week_ago
    ).scalar() or 0
    
    return DetectionStatistics(
        total_detections=total,
        valid_detections=valid,
        invalid_detections=invalid,
        upload_count=upload_count,
        camera_count=camera_count,
        average_confidence=round(float(avg_conf), 4),
        today_count=today_count,
        this_week_count=week_count
    )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
