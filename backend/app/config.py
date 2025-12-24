"""
PlateVision Configuration Settings
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    database_url: str = "mysql+pymysql://app:root@localhost:3306/platevision"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Camera Settings
    camera_index: int = 0
    detection_interval_ms: int = 500
    
    # YOLO Model
    model_path: str = "../best.pt"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

