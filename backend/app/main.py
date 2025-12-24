"""
PlateVision Backend - FastAPI Application Entry Point
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .database import init_db
from .api.routes import router
from .api.websocket import camera_websocket

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print("Starting PlateVision Backend...")
    
    # Initialize database tables
    init_db()
    print("Database initialized")
    
    yield
    
    # Shutdown
    print("Shutting down PlateVision Backend...")


# Create FastAPI application
app = FastAPI(
    title="PlateVision API",
    description="AI-Based License Plate Detection and Recognition System for Indonesian Plates",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router)

# WebSocket endpoint
app.websocket("/ws/camera")(camera_websocket)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "PlateVision API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
