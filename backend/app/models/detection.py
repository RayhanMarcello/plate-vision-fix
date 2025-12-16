"""
Detection Result Model
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Enum
from sqlalchemy.sql import func
from ..database import Base
import enum


class SourceType(str, enum.Enum):
    """Source type for detection."""
    UPLOAD = "upload"
    CAMERA = "camera"


class DetectionResult(Base):
    """Model for storing plate detection results."""
    
    __tablename__ = "detection_results"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Plate information
    plate_number = Column(String(20), index=True, nullable=False)
    raw_ocr_text = Column(String(50), nullable=True)
    confidence = Column(Float, default=0.0)
    
    # Source information
    source_type = Column(Enum(SourceType), default=SourceType.UPLOAD)
    
    # Image paths
    image_path = Column(String(255), nullable=True)  # Cropped plate image
    original_image_path = Column(String(255), nullable=True)  # Original image
    
    # Validation
    is_valid = Column(Boolean, default=False)
    
    # Timestamps
    detected_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<DetectionResult(id={self.id}, plate='{self.plate_number}')>"
