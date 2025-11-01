# AI Surveillance — Theft Detection

A Flask-based PWA for real-time theft detection with YOLO, OpenCV, Socket.IO, recordings, and alerting.

## Features
- Real-time detection with YOLO (Ultralytics)
- RTSP camera support
- Recent events with recorded MP4 (H.264)
- PWA: service worker, installable
- Frontend alarm sound with toggle
- Onboarding wizard (settings + camera)
- Registration and login (SQLite)

## Requirements
- Python 3.9+
- Windows (tested) or Linux/macOS
- FFmpeg (for some camera streams) optional

## Quick Start

```bash
# 1) Create and activate venv
python -m venv .venv
.venv\Scripts\activate  # Windows

# 2) Install deps
pip install -r requirements.txt  # if exists, otherwise see below

# 3) Run the app
python app_pwa.py

# 4) Open in browser
http://localhost:5000
```

If you don't have a requirements file yet, typical packages:
```
flask
flask-socketio
python-socketio
eventlet
werkzeug
opencv-python
ultralytics
pillow
transformers
torch  # optional, depending on CUDA/CPU availability
```

## Default Admin
- Username: `admin`
- Password: `admin123`

## Onboarding
- Step 1: Contact & notifications (phone, emergency, toggles)
- Step 2: Camera RTSP + test connection
- Step 3: Complete → Dashboard

## Project Structure (key files)
```
app_pwa.py                  # Flask app, routes, sockets
static/sw.js                # Service worker
templates/*.html            # UI pages (login, index, events, onboarding)
theft_detection.py          # Detection (backend)
videos/                     # Recorded theft events (ignored by Git)
```

## Environment & Secrets
Create a `.env` if you add secrets (kept out of Git by .gitignore):
```
SECRET_KEY=your-secret
```

## Git
Initialize and push to GitHub (see below commands).

## Troubleshooting
- If service worker cache errors: unregister and hard refresh (Ctrl+Shift+F5)
- If alarm sound doesn’t play first time: click once to allow audio, ensure `alarm.wav` exists
- If video not playing: ensure codec H.264 (`avc1`) and source element uses `type="video/mp4"`

## License
MIT
