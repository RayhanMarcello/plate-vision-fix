"""
PlateVision Configuration Settings
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    database_url: str = "mysql+pymysql://app:root@localhost:3306/platevision"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # File Storage
    upload_dir: str = "uploads"
    detection_dir: str = "detections"
    
    # Camera Settings
    camera_index: int = 0
    detection_interval_ms: int = 500
    
    # YOLO Model
    model_path: str = "../best.pt"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @property
    def upload_path(self) -> Path:
        """Get absolute path for uploads directory."""
        path = Path(__file__).parent.parent / self.upload_dir
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def detection_path(self) -> Path:
        """Get absolute path for detections directory."""
        path = Path(__file__).parent.parent / self.detection_dir
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
