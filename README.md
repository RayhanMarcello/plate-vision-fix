# PlateVision - Sistema Deteksi Plat Nomor Kendaraan Indonesia

Sistema deteksi dan pengenalan plat nomor kendaraan Indonesia berbasis AI menggunakan YOLO, EasyOCR, dan validasi format plat Indonesia.

## 🚀 Fitur Utama

- **Deteksi Real-time** - Streaming kamera langsung dengan deteksi otomatis
- **Upload Gambar** - Upload foto kendaraan untuk deteksi batch
- **OCR Cerdas** - Pengenalan karakter dengan EasyOCR
- **Validasi Indonesia** - Normalisasi dan validasi format plat Indonesia
- **Database MySQL** - Penyimpanan hasil deteksi
- **Dashboard Admin** - Antarmuka web modern untuk manajemen
- **Statistik Real-time** - Analytics dan visualisasi performa

## 📋 Persyaratan Sistem

- Python 3.10+
- Node.js 18+
- MySQL 8.0+
- Webcam/camera (untuk fitur real-time)

## 🛠️ Instalasi

### 1. Clone Repository

```bash
cd d:\plateVision -ippl
```

### 2. Setup Database MySQL

Buka MySQL CLI dan buat database:

```sql
CREATE DATABASE platevision;
CREATE USER 'app'@'localhost' IDENTIFIED BY 'root';
GRANT ALL PRIVILEGES ON platevision.* TO 'app'@'localhost';
FLUSH PRIVILEGES;
```

### 3. Install Backend (Python)

```bash
cd backend
pip install -r requirements.txt
```

Jalankan backend:

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend akan berjalan di: http://localhost:8000
API Docs (Swagger): http://localhost:8000/docs

### 4. Install Frontend (React)

Buka terminal baru:

```bash
cd frontend
npm install
npm run dev
```

Frontend akan berjalan di: http://localhost:5173

## 📁 Struktur Proyek

```
plateVision-ippl/
├── best.pt                          # YOLO model weights
├── detect.py                        # Original detection script
│
├── backend/                         # Python Backend (FastAPI)
│   ├── app/
│   │   ├── main.py                  # Entry point
│   │   ├── config.py                # Settings
│   │   ├── database.py              # MySQL connection
│   │   ├── models/                  # SQLAlchemy models
│   │   ├── schemas/                 # Pydantic schemas
│   │   ├── services/                # Business logic
│   │   │   ├── detection.py         # YOLO detection
│   │   │   ├── ocr.py               # EasyOCR
│   │   │   └── validator.py         # Plate validation
│   │   └── api/                     # REST & WebSocket
│   ├── uploads/                     # Uploaded images
│   ├── detections/                  # Detected plates
│   ├── requirements.txt
│   └── .env
│
└── frontend/                        # React Frontend (Vite)
    ├── src/
    │   ├── main.jsx
    │   ├── App.jsx
    │   ├── index.css                # Design system
    │   ├── services/api.js          # API client
    │   ├── hooks/useWebSocket.js    # WebSocket hook
    │   ├── components/Layout/
    │   └── pages/
    │       ├── Dashboard.jsx        # Home
    │       ├── LiveCamera.jsx       # Real-time camera
    │       ├── Upload.jsx           # Image upload
    │       ├── Detections.jsx       # Data management
    │       └── Statistics.jsx       # Analytics
    ├── package.json
    └── vite.config.js
```

## 🎯 Penggunaan

### Upload Gambar
1. Buka http://localhost:5173/upload
2. Drag & drop atau klik untuk memilih gambar
3. Klik "Detect Plate"
4. Hasil deteksi akan ditampilkan dan disimpan ke database

### Kamera Real-time
1. Buka http://localhost:5173/camera
2. Klik "Start Camera"
3. Arahkan kamera ke plat nomor kendaraan
4. Sistem akan otomatis mendeteksi dan menyimpan hasil

### Manajemen Data
1. Buka http://localhost:5173/detections
2. Gunakan search untuk mencari plat nomor
3. Filter berdasarkan sumber atau validitas
4. Hapus data yang tidak diperlukan

### Statistik
1. Buka http://localhost:5173/statistics
2. Lihat grafik validasi, distribusi sumber, dan timeline

## 🔧 API Endpoints

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/detect/upload` | Upload image for detection |
| GET | `/api/detections` | List all detections (paginated) |
| GET | `/api/detections/{id}` | Get single detection |
| DELETE | `/api/detections/{id}` | Delete detection |
| GET | `/api/statistics` | Get system statistics |


### WebSocket

- **Endpoint**: `ws://localhost:8000/ws/camera`
- **Events**:
  - `camera:start` - Mulai streaming
  - `camera:stop` - Stop streaming
  - `camera:frame` - Frame video (base64)
  - `detection:new` - Deteksi baru

## 🧩 Format Plat Indonesia

Sistem mendukung validasi format plat nomor Indonesia:

**Format**: `[Kode Wilayah 1-2 huruf] [Angka 1-4 digit] [Seri 1-3 huruf]`

**Contoh**:
- B 1234 ABC (Jakarta)
- D 1 A (Bandung)
- AB 12 CD (Yogyakarta)
- L 5678 XYZ (Surabaya)

### Normalisasi Karakter

Sistem otomatis menormalisasi kesalahan OCR:
- O → 0 (dalam angka)
- 0 → O (dalam huruf)
- I/l → 1 (dalam angka)
- 1 → I (dalam huruf)

## 🎨 Teknologi

### Backend
- FastAPI - Modern Python web framework
- SQLAlchemy - ORM untuk MySQL
- Ultralytics YOLO - Object detection
- EasyOCR - Optical character recognition
- OpenCV - Image processing

### Frontend
- React 18 - UI library
- Vite - Build tool
- Recharts - Data visualization
- Lucide React - Icons
- Axios - HTTP client

## 📊 Database Schema

```sql
detection_results:
  - id (INT, PK)
  - plate_number (VARCHAR(20))
  - raw_ocr_text (VARCHAR(50))
  - confidence (FLOAT)
  - source_type (ENUM: 'upload', 'camera')
  - image_path (VARCHAR(255))
  - original_image_path (VARCHAR(255))
  - is_valid (BOOLEAN)
  - detected_at (DATETIME)
```

## 🐛 Troubleshooting

### Backend tidak bisa start
```bash
# Pastikan semua dependencies terinstall
pip install -r backend/requirements.txt

# Cek apakah MySQL running
mysql -u app -p

# Cek port 8000 tidak digunakan
netstat -ano | findstr :8000
```

### Frontend tidak bisa connect ke backend
- Pastikan backend running di port 8000
- Cek proxy settings di `vite.config.js`
- Clear browser cache

### EasyOCR download model lambat
- Model akan otomatis download saat pertama kali dijalankan
- Model disimpan di user home directory
- Butuh koneksi internet yang stabil


