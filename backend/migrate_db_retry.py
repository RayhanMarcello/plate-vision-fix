
import sys
import os

# Add parent directory to path
sys.path.append(os.getcwd())

from app.database import engine, Base
from app.models.detection import DetectionResult

def migrate():
    try:
        print("Dropping existing table...")
        DetectionResult.__table__.drop(bind=engine, checkfirst=True)
        
        print("Creating new table...")
        Base.metadata.create_all(bind=engine)
        print("Migration successful!")
    except Exception as e:
        print(f"Migration failed: {e}")

if __name__ == "__main__":
    migrate()
