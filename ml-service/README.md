---
title: PlateVision ML
emoji: 🚗
colorFrom: blue  
colorTo: green
sdk: docker
pinned: false
license: mit
---

# PlateVision ML Service

AI-based Indonesian license plate detection and OCR service.

## API Endpoints

- `POST /api/detect` - Detect license plates from image
- `GET /health` - Health check

## Usage

Send a POST request with base64 encoded image:

```json
{
  "image": "base64_encoded_image_here"
}
```

Response:
```json
[
  {
    "plate_number": "B 1234 ABC",
    "raw_text": "B1234ABC",
    "confidence": 0.95,
    "is_valid": true,
    "bbox": [x1, y1, x2, y2],
    "plate_image": "data:image/jpeg;base64,..."
  }
]
```
