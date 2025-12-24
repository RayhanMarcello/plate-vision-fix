
import sys
import os

# Add parent directory to path so we can import app
sys.path.append(os.getcwd())

try:
    from app.database import engine, Base
    from app.models.detection import DetectionResult
    
    print("Dropping detection_results table...")
    DetectionResult.__table__.drop(bind=engine, checkfirst=True)
    
    print("Creating detection_results table...")
    Base.metadata.create_all(bind=engine)
    
    print("Database migration completed successfully.")
except Exception as e:
    print(f"Error during migration: {e}")
    sys.exit(1)
