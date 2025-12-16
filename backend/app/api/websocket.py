"""
WebSocket Handler for Real-time Camera Detection
"""
import asyncio
import base64
import json
import cv2
import numpy as np
from datetime import datetime
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models.detection import DetectionResult, SourceType
from ..services.detection import DetectionService
from ..services.ocr import OCRService
from ..services.validator import PlateValidator
from ..config import get_settings

settings = get_settings()


class CameraManager:
    """Manages camera connections and streaming."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.camera: Optional[cv2.VideoCapture] = None
        self.is_streaming = False
        self.detection_service: Optional[DetectionService] = None
        self.ocr_service: Optional[OCRService] = None
        self._stream_task: Optional[asyncio.Task] = None
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total clients: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"Client disconnected. Total clients: {len(self.active_connections)}")
        
        # Stop camera if no clients
        if not self.active_connections:
            self.stop_camera()
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    def start_camera(self) -> bool:
        """Start the camera capture."""
        if self.camera is not None and self.camera.isOpened():
            return True
        
        self.camera = cv2.VideoCapture(settings.camera_index)
        
        if not self.camera.isOpened():
            print(f"Failed to open camera at index {settings.camera_index}")
            return False
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize services
        if self.detection_service is None:
            self.detection_service = DetectionService()
        if self.ocr_service is None:
            self.ocr_service = OCRService()
        
        print("Camera started successfully")
        return True
    
    def stop_camera(self):
        """Stop the camera capture."""
        self.is_streaming = False
        
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
        
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        
        print("Camera stopped")
    
    async def stream_frames(self):
        """Stream camera frames with detection."""
        if not self.start_camera():
            await self.broadcast({
                "type": "error",
                "message": "Failed to start camera"
            })
            return
        
        self.is_streaming = True
        detection_interval = settings.detection_interval_ms / 1000.0
        last_detection_time = 0
        
        await self.broadcast({
            "type": "camera:status",
            "status": "started"
        })
        
        try:
            while self.is_streaming and self.active_connections:
                ret, frame = self.camera.read()
                
                if not ret:
                    await asyncio.sleep(0.1)
                    continue
                
                current_time = asyncio.get_event_loop().time()
                
                # Run detection at specified interval
                if current_time - last_detection_time >= detection_interval:
                    last_detection_time = current_time
                    
                    # Detect and annotate
                    annotated_frame, detections = self.detection_service.detect_and_annotate(frame)
                    
                    # Process detections
                    for cropped_plate, bbox, det_confidence in detections:
                        await self._process_detection(cropped_plate, bbox, det_confidence, frame)
                else:
                    annotated_frame = frame
                
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Broadcast frame
                await self.broadcast({
                    "type": "camera:frame",
                    "data": frame_base64
                })
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            print(f"Stream error: {e}")
        finally:
            self.is_streaming = False
            await self.broadcast({
                "type": "camera:status",
                "status": "stopped"
            })
    
    async def _process_detection(
        self, 
        cropped_plate: np.ndarray, 
        bbox: list, 
        det_confidence: float,
        original_frame: np.ndarray
    ):
        """Process a detected plate: OCR, validate, save to DB."""
        try:
            # Run OCR
            raw_text, ocr_confidence = self.ocr_service.extract_text(cropped_plate)
            
            if not raw_text:
                return
            
            # Validate and normalize
            plate_number, is_valid, region = PlateValidator.process_ocr_result(raw_text)
            
            if not plate_number:
                return
            
            confidence = (det_confidence + ocr_confidence) / 2
            
            # Save images
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            crop_filename = f"{timestamp}_plate.jpg"
            crop_path = settings.detection_path / crop_filename
            cv2.imwrite(str(crop_path), cropped_plate)
            
            original_filename = f"{timestamp}_frame.jpg"
            original_path = settings.upload_path / original_filename
            cv2.imwrite(str(original_path), original_frame)
            
            # Save to database
            db = SessionLocal()
            try:
                db_record = DetectionResult(
                    plate_number=plate_number,
                    raw_ocr_text=raw_text,
                    confidence=confidence,
                    source_type=SourceType.CAMERA,
                    image_path=f"/detections/{crop_filename}",
                    original_image_path=f"/uploads/{original_filename}",
                    is_valid=is_valid
                )
                db.add(db_record)
                db.commit()
                db.refresh(db_record)
                
                # Broadcast new detection
                await self.broadcast({
                    "type": "detection:new",
                    "data": {
                        "id": db_record.id,
                        "plate_number": plate_number,
                        "raw_ocr_text": raw_text,
                        "confidence": round(confidence, 4),
                        "is_valid": is_valid,
                        "region": region,
                        "bbox": bbox,
                        "image_path": f"/detections/{crop_filename}",
                        "detected_at": db_record.detected_at.isoformat()
                    }
                })
            finally:
                db.close()
                
        except Exception as e:
            print(f"Detection processing error: {e}")


# Global camera manager instance
camera_manager = CameraManager()


async def camera_websocket(websocket: WebSocket):
    """WebSocket endpoint for camera streaming."""
    await camera_manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            event_type = message.get("type", "")
            
            if event_type == "camera:start":
                if not camera_manager.is_streaming:
                    camera_manager._stream_task = asyncio.create_task(
                        camera_manager.stream_frames()
                    )
                else:
                    await websocket.send_json({
                        "type": "camera:status",
                        "status": "already_running"
                    })
            
            elif event_type == "camera:stop":
                camera_manager.stop_camera()
            
            elif event_type == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        camera_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        camera_manager.disconnect(websocket)
