# 🚀 PlateVision Deployment Guide

Panduan lengkap untuk deploy PlateVision ke:
- **Frontend** → Netlify
- **Backend + Database** → Railway (MySQL)
- **ML Model** → Hugging Face Spaces

---

## 📋 Ringkasan Project

| Komponen | Teknologi | Target Deploy |
|----------|-----------|---------------|
| Frontend | React + Vite | Netlify |
| Backend | FastAPI + Python | Railway |
| Database | MySQL | Railway (MySQL Plugin) |
| ML Model | YOLO + EasyOCR | Hugging Face Spaces |

### Struktur Project
```
plateVision/
├── frontend/          # React + Vite
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── backend/           # FastAPI
│   ├── app/
│   ├── best.pt        # YOLO model
│   ├── requirements.txt
│   └── Dockerfile
└── best.pt            # YOLO model (copy)
```

---

## 🔧 BAGIAN 1: Deploy Backend + Database ke Railway

### Step 1.1: Persiapan Railway

1. **Buat akun Railway**
   - Kunjungi [railway.app](https://railway.app)
   - Sign up menggunakan GitHub

2. **Buat Project Baru**
   - Klik **"New Project"**
   - Pilih **"Empty Project"**

### Step 1.2: Setup MySQL Database

1. **Tambahkan MySQL**
   - Di project, klik **"+ New"** → **"Database"** → **"MySQL"**
   - Railway akan otomatis membuat database MySQL

2. **Catat Credentials**
   - Klik MySQL service → **"Variables"** tab
   - Catat nilai berikut:
     ```
     MYSQL_HOST=xxx.railway.internal
     MYSQL_PORT=3306
     MYSQL_USER=root
     MYSQL_PASSWORD=xxxxx
     MYSQL_DATABASE=railway
     ```

3. **Buat Database `platevision`**
   - Klik **"Data"** tab di MySQL service
   - Jalankan query:
     ```sql
     CREATE DATABASE IF NOT EXISTS platevision;
     ```

### Step 1.3: Deploy Backend

1. **Push Backend ke GitHub**
   - Pastikan folder `backend/` sudah di GitHub repository

2. **Tambahkan Backend Service**
   - Di Railway project, klik **"+ New"** → **"GitHub Repo"**
   - Pilih repository Anda
   - Set **Root Directory**: `backend`

3. **Konfigurasi Environment Variables**
   - Klik backend service → **"Variables"** tab
   - Tambahkan:
     ```
     DATABASE_URL=mysql+pymysql://root:PASSWORD@HOST:PORT/platevision
     HOST=0.0.0.0
     PORT=8000
     DEBUG=false
     ```
   
   > **Catatan**: Ganti `PASSWORD`, `HOST`, `PORT` dengan nilai dari MySQL service.
   > Format lengkap: `mysql+pymysql://root:xxxxxx@xxx.railway.internal:3306/platevision`

4. **Konfigurasi Build**
   - Klik **"Settings"** tab
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

5. **Generate Domain**
   - Klik **"Settings"** → **"Networking"**
   - Klik **"Generate Domain"**
   - Catat URL: `https://platevision-backend-xxxx.up.railway.app`

### Step 1.4: Alternatif - Deploy dengan Dockerfile

Jika ingin menggunakan Dockerfile, update `backend/Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads detections

# Railway menggunakan PORT environment variable
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

---

## 🌐 BAGIAN 2: Deploy Frontend ke Netlify

### Step 2.1: Update API Configuration

1. **Buat file `frontend/.env.production`**:
   ```env
   VITE_API_URL=https://platevision-backend-xxxx.up.railway.app
   ```

2. **Update `frontend/src/services/api.js`**:
   ```javascript
   import axios from 'axios';

   // Gunakan environment variable atau default ke relative path
   const API_BASE = import.meta.env.VITE_API_URL 
     ? `${import.meta.env.VITE_API_URL}/api` 
     : '/api';

   const api = axios.create({
     baseURL: API_BASE,
     headers: {
       'Content-Type': 'application/json',
     },
   });

   // ... rest of the code
   ```

3. **Update `frontend/vite.config.js`** untuk production:
   ```javascript
   import { defineConfig } from 'vite'
   import react from '@vitejs/plugin-react'

   export default defineConfig({
     plugins: [react()],
     server: {
       port: 5173,
       proxy: {
         '/api': {
           target: 'http://localhost:8000',
           changeOrigin: true
         },
         '/ws': {
           target: 'ws://localhost:8000',
           ws: true
         }
       }
     },
     build: {
       outDir: 'dist'
     }
   })
   ```

### Step 2.2: Deploy ke Netlify

1. **Buat akun Netlify**
   - Kunjungi [netlify.com](https://netlify.com)
   - Sign up menggunakan GitHub

2. **Import Project**
   - Klik **"Add new site"** → **"Import an existing project"**
   - Pilih **GitHub** dan authorize
   - Pilih repository Anda

3. **Konfigurasi Build Settings**
   - Base directory: `frontend`
   - Build command: `npm run build`
   - Publish directory: `frontend/dist`

4. **Set Environment Variables**
   - Klik **"Site settings"** → **"Environment variables"**
   - Tambahkan:
     ```
     VITE_API_URL=https://platevision-backend-xxxx.up.railway.app
     ```

5. **Buat file `frontend/netlify.toml`**:
   ```toml
   [build]
     base = "frontend"
     command = "npm run build"
     publish = "dist"

   [[redirects]]
     from = "/api/*"
     to = "https://platevision-backend-xxxx.up.railway.app/api/:splat"
     status = 200
     force = true

   [[redirects]]
     from = "/*"
     to = "/index.html"
     status = 200
   ```

6. **Deploy**
   - Klik **"Deploy site"**
   - Tunggu proses build selesai
   - Akses URL: `https://your-site-name.netlify.app`

---

## 🤖 BAGIAN 3: Deploy ML Model ke Hugging Face Spaces

### Step 3.1: Persiapan

1. **Buat akun Hugging Face**
   - Kunjungi [huggingface.co](https://huggingface.co)
   - Sign up

2. **Buat New Space**
   - Klik **"New Space"**
   - Name: `platevision`
   - SDK: **Docker**
   - Visibility: Public atau Private

### Step 3.2: Persiapan Files

1. **Struktur folder untuk HF Space**:
   ```
   platevision-space/
   ├── app/
   │   ├── __init__.py
   │   ├── main.py
   │   ├── config.py
   │   ├── api/
   │   ├── services/
   │   ├── models/
   │   └── schemas/
   ├── best.pt
   ├── requirements.txt
   ├── Dockerfile
   └── README.md
   ```

2. **Update `Dockerfile` untuk Hugging Face**:
   ```dockerfile
   FROM python:3.10-slim

   WORKDIR /app

   RUN apt-get update && apt-get install -y \
       libgl1 \
       libglib2.0-0 \
       && rm -rf /var/lib/apt/lists/*

   COPY requirements.txt .
   RUN pip install --no-cache-dir --upgrade pip && \
       pip install --no-cache-dir -r requirements.txt

   COPY . .

   RUN mkdir -p uploads detections

   # Hugging Face Spaces uses port 7860
   ENV PORT=7860
   ENV HOST=0.0.0.0
   
   # Use SQLite for demo (no external DB needed)
   ENV DATABASE_URL=sqlite:///./platevision.db

   CMD ["sh", "-c", "uvicorn app.main:app --host $HOST --port $PORT"]
   ```

3. **Buat `README.md` untuk Space**:
   ```markdown
   ---
   title: PlateVision
   emoji: 🚗
   colorFrom: blue
   colorTo: green
   sdk: docker
   pinned: false
   license: mit
   ---

   # PlateVision - License Plate Detection API

   AI-based Indonesian license plate detection and OCR system.

   ## API Endpoints

   - `POST /api/detect/upload` - Upload image for plate detection
   - `GET /api/detections` - List all detections
   - `GET /api/statistics` - Get detection statistics
   - `GET /docs` - Swagger API documentation
   ```

### Step 3.3: Deploy ke Hugging Face

1. **Clone Space Repository**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/platevision
   cd platevision
   ```

2. **Copy Files**
   ```bash
   # Copy seluruh isi backend/
   cp -r /path/to/plateVision/backend/* .
   ```

3. **Install Git LFS** (untuk file besar seperti `best.pt`)
   ```bash
   git lfs install
   git lfs track "*.pt"
   ```

4. **Push ke Hugging Face**
   ```bash
   git add .
   git commit -m "Initial deployment"
   git push
   ```

5. **Akses Space**
   - URL: `https://huggingface.co/spaces/YOUR_USERNAME/platevision`
   - API: `https://YOUR_USERNAME-platevision.hf.space`

---

## 🔄 BAGIAN 4: Integrasi Semua Komponen

### Arsitektur Final

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     Netlify     │     │     Railway     │     │  Hugging Face   │
│   (Frontend)    │────▶│   (Backend)     │     │    (ML Demo)    │
│                 │     │                 │     │                 │
│  React + Vite   │     │  FastAPI + DB   │     │  FastAPI + ML   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │
                              ▼
                        ┌───────────┐
                        │  Railway  │
                        │   MySQL   │
                        └───────────┘
```

### Konfigurasi CORS di Backend

Update `backend/app/main.py` untuk production:

```python
# Configure CORS for production
origins = [
    "https://your-site.netlify.app",
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Atau ["*"] untuk development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## ✅ Checklist Deployment

### Backend (Railway)
- [ ] MySQL database created
- [ ] `platevision` database created
- [ ] Backend service deployed
- [ ] Environment variables configured
- [ ] Domain generated and working
- [ ] API endpoints accessible (`/docs`)

### Frontend (Netlify)
- [ ] `VITE_API_URL` environment variable set
- [ ] `netlify.toml` created
- [ ] Build successful
- [ ] Site accessible
- [ ] API calls working

### ML (Hugging Face)
- [ ] Space created with Docker SDK
- [ ] `best.pt` uploaded with Git LFS
- [ ] Build successful
- [ ] API accessible

---

## 🐛 Troubleshooting

### Backend tidak bisa connect ke MySQL
- Pastikan `DATABASE_URL` menggunakan internal Railway URL
- Format: `mysql+pymysql://root:PASSWORD@HOST:PORT/platevision`

### Frontend 404 pada refresh
- Pastikan `netlify.toml` ada dengan redirect rule ke `index.html`

### CORS Error
- Update CORS origins di `main.py` dengan domain Netlify

### Hugging Face build timeout
- Reduce dependencies di `requirements.txt`
- Hapus PaddleOCR jika tidak diperlukan (berat)

---

## 📞 Support

Jika ada masalah, pastikan:
1. Semua environment variables sudah benar
2. Database sudah dibuat dengan nama `platevision`
3. CORS sudah dikonfigurasi untuk domain frontend
