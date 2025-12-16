"""
Pydantic Schemas for Detection API
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
from enum import Enum


class SourceType(str, Enum):
    """Source type for detection."""
    UPLOAD = "upload"
    CAMERA = "camera"


class PlateDetectionResult(BaseModel):
    """Schema for individual plate detection in an image."""
    plate_number: str
    raw_ocr_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    is_valid: bool
    bbox: List[int] = Field(description="Bounding box [x1, y1, x2, y2]")
    cropped_image_path: Optional[str] = None


class DetectionResultCreate(BaseModel):
    """Schema for creating a new detection result."""
    plate_number: str = Field(max_length=20)
    raw_ocr_text: Optional[str] = Field(default=None, max_length=50)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    source_type: SourceType = SourceType.UPLOAD
    image_path: Optional[str] = None
    original_image_path: Optional[str] = None
    is_valid: bool = False


class DetectionResultResponse(BaseModel):
    """Schema for detection result response."""
    id: int
    plate_number: str
    raw_ocr_text: Optional[str]
    confidence: float
    source_type: SourceType
    image_path: Optional[str]
    original_image_path: Optional[str]
    is_valid: bool
    detected_at: datetime
    
    class Config:
        from_attributes = True


class DetectionResultList(BaseModel):
    """Schema for paginated list of detection results."""
    items: List[DetectionResultResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class DetectionStatistics(BaseModel):
    """Schema for detection statistics."""
    total_detections: int
    valid_detections: int
    invalid_detections: int
    upload_count: int
    camera_count: int
    average_confidence: float
    today_count: int
    this_week_count: int
