"""
AI Surveillance PWA - Theft Detection System
Flask backend with SocketIO for real-time video streaming and notifications
"""

import os
import cv2
import time
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for, send_from_directory
from flask_socketio import SocketIO, emit
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import numpy as np
from pathlib import Path
import base64

# Import theft detection system
from theft_detection import TheftDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['VIDEO_FOLDER'] = 'videos'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VIDEO_FOLDER'], exist_ok=True)
os.makedirs('static/sounds', exist_ok=True)

# Global variables
detector = None
detection_thread = None
is_detecting = False
current_frame = None
detection_active = False
video_writer = None
recording_theft = False
theft_start_time = None

# Database setup
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect('surveillance.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT,
        phone TEXT,
        notifications_enabled INTEGER DEFAULT 1,
        emergency_auto INTEGER DEFAULT 0,
        emergency_contact TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Members table (registered faces)
    c.execute('''CREATE TABLE IF NOT EXISTS members (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        photo TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Settings table
    c.execute('''CREATE TABLE IF NOT EXISTS settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        rtsp_url TEXT,
        model_path TEXT DEFAULT 'yolo11n.pt',
        imgsz INTEGER DEFAULT 320,
        frame_skip INTEGER DEFAULT 2,
        recognition_interval INTEGER DEFAULT 60,
        proximity_threshold INTEGER DEFAULT 100,
        display_width INTEGER DEFAULT 640,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )''')
    
    # Theft events table
    c.execute('''CREATE TABLE IF NOT EXISTS theft_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        person_name TEXT,
        bag_count INTEGER,
        video_path TEXT,
        snapshot_path TEXT,
        alert_sent INTEGER DEFAULT 0
    )''')
    
    # Create default admin user if not exists
    c.execute("SELECT * FROM users WHERE username='admin'")
    if not c.fetchone():
        hashed_pw = generate_password_hash('admin123')
        c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                 ('admin', hashed_pw, 'admin@surveillance.com'))
    
    conn.commit()
    conn.close()

init_db()


# Helper functions
def get_db():
    """Get database connection"""
    conn = sqlite3.connect('surveillance.db')
    conn.row_factory = sqlite3.Row
    return conn


def login_required(f):
    """Decorator to require login"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def get_user_settings(user_id):
    """Get user settings with proper defaults"""
    defaults = {
        'rtsp_url': 'rtsp://192.168.100.13:8080/h264.sdp',
        'model_path': 'yolo11n.pt',
        'imgsz': 320,
        'frame_skip': 2,
        'recognition_interval': 60,
        'proximity_threshold': 100,
        'display_width': 640
    }
    
    db = get_db()
    settings = db.execute('SELECT * FROM settings WHERE user_id = ?', (user_id,)).fetchone()
    db.close()
    
    if settings:
        result = dict(settings)
        # Ensure all required keys exist and have correct types
        for key, default_value in defaults.items():
            if key not in result or result[key] is None:
                result[key] = default_value
            elif isinstance(default_value, int) and not isinstance(result[key], int):
                result[key] = int(result[key])
        return result
    
    return defaults


def start_video_recording(filename):
    """Start recording video for theft event"""
    global video_writer, recording_theft, theft_start_time
    
    video_path = os.path.join(app.config['VIDEO_FOLDER'], filename)
    # Use H.264 codec for better browser compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 360))
    recording_theft = True
    theft_start_time = time.time()
    return video_path


def stop_video_recording():
    """Stop recording video"""
    global video_writer, recording_theft
    if video_writer:
        video_writer.release()
        video_writer = None
    recording_theft = False


def send_notification(message, event_type='info'):
    """Send notification to connected clients"""
    socketio.emit('notification', {
        'message': message,
        'type': event_type,
        'timestamp': datetime.now().isoformat()
    })


def trigger_emergency_call(user_id, theft_event_id):
    """Trigger emergency call if enabled"""
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    
    if user and user['emergency_auto']:
        contact = user['emergency_contact']
        # In production, integrate with Twilio or similar service
        socketio.emit('emergency_call', {
            'contact': contact,
            'event_id': theft_event_id,
            'timestamp': datetime.now().isoformat()
        })
        print(f"[EMERGENCY] Auto-calling {contact} for theft event #{theft_event_id}")
    
    db.close()


# Routes
@app.route('/')
@login_required
def index():
    """Main dashboard"""
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    recent_events = db.execute(
        'SELECT * FROM theft_events ORDER BY timestamp DESC LIMIT 10'
    ).fetchall()
    members_count = db.execute('SELECT COUNT(*) as count FROM members').fetchone()['count']
    db.close()
    
    return render_template('index_pwa.html', 
                         user=dict(user), 
                         recent_events=[dict(e) for e in recent_events],
                         members_count=members_count)


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        db.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            return jsonify({'success': True, 'redirect': url_for('index')})
        
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    
    return render_template('login_pwa.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Register new user"""
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        # Validation
        if not username or len(username) < 3:
            return jsonify({'success': False, 'message': 'Username must be at least 3 characters'}), 400
        
        if not password or len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters'}), 400
        
        if not email:
            return jsonify({'success': False, 'message': 'Email is required'}), 400
        
        db = get_db()
        
        # Check if username exists
        existing_user = db.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
        if existing_user:
            db.close()
            return jsonify({'success': False, 'message': 'Username already exists'}), 400
        
        # Check if email exists
        existing_email = db.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
        if existing_email:
            db.close()
            return jsonify({'success': False, 'message': 'Email already registered'}), 400
        
        # Create new user
        hashed_password = generate_password_hash(password)
        try:
            db.execute('''INSERT INTO users (username, password, email, created_at)
                         VALUES (?, ?, ?, CURRENT_TIMESTAMP)''',
                      (username, hashed_password, email))
            db.commit()
            user_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]
            
            # Create default settings for new user
            db.execute('''INSERT INTO settings (user_id, rtsp_url)
                         VALUES (?, ?)''',
                      (user_id, 'rtsp://192.168.100.13:8080/h264.sdp'))
            db.commit()
            db.close()
            
            # Auto-login the new user
            session['user_id'] = user_id
            session['username'] = username
            
            return jsonify({'success': True, 'redirect': url_for('onboarding')})
        except Exception as e:
            db.close()
            return jsonify({'success': False, 'message': 'Registration failed'}), 500
    
    return render_template('register_pwa.html')


@app.route('/logout')
def logout():
    """Logout"""
    session.clear()
    return redirect(url_for('login'))


@app.route('/onboarding')
@login_required
def onboarding():
    """Onboarding page for new users"""
    return render_template('onboarding_pwa.html')


@app.route('/api/test-rtsp', methods=['POST'])
@login_required
def test_rtsp():
    """Test RTSP connection and return a frame"""
    try:
        data = request.get_json()
        rtsp_url = data.get('rtsp_url')
        
        if not rtsp_url:
            return jsonify({'success': False, 'message': 'RTSP URL required'})
        
        # Try to capture a frame
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            return jsonify({'success': False, 'message': 'Cannot connect to camera'})
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'success': False, 'message': 'Failed to read frame'})
        
        # Resize and encode frame
        frame = cv2.resize(frame, (480, 360))
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({'success': True, 'frame': frame_base64})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/complete-onboarding', methods=['POST'])
@login_required
def complete_onboarding():
    """Complete onboarding and save user preferences"""
    try:
        data = request.get_json()
        phone = data.get('phone', '')
        emergency_contact = data.get('emergency_contact', '')
        emergency_auto = data.get('emergency_auto', 0)
        notifications_enabled = data.get('notifications_enabled', 1)
        rtsp_url = data.get('rtsp_url', '')
        
        user_id = session['user_id']
        db = get_db()
        
        # Update user profile with contact and notification settings
        db.execute('''UPDATE users SET 
                     phone = ?,
                     emergency_contact = ?,
                     emergency_auto = ?,
                     notifications_enabled = ?
                     WHERE id = ?''',
                  (phone, emergency_contact, emergency_auto, notifications_enabled, user_id))
        
        # Update user settings with RTSP URL
        db.execute('''UPDATE settings 
                     SET rtsp_url = ?
                     WHERE user_id = ?''',
                  (rtsp_url, user_id))
        
        db.commit()
        db.close()
        
        return jsonify({'success': True, 'message': 'Setup completed'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile page"""
    db = get_db()
    
    if request.method == 'POST':
        data = request.get_json()
        
        db.execute('''UPDATE users SET 
                     email = ?, 
                     phone = ?, 
                     notifications_enabled = ?, 
                     emergency_auto = ?,
                     emergency_contact = ?
                     WHERE id = ?''',
                  (data.get('email'), data.get('phone'), 
                   data.get('notifications_enabled', 1),
                   data.get('emergency_auto', 0),
                   data.get('emergency_contact'),
                   session['user_id']))
        db.commit()
        db.close()
        
        return jsonify({'success': True, 'message': 'Profile updated'})
    
    user = db.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    db.close()
    
    return render_template('profile_pwa.html', user=dict(user))


@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    """Settings page"""
    db = get_db()
    
    if request.method == 'POST':
        data = request.get_json()
        
        # Check if settings exist
        existing = db.execute('SELECT id FROM settings WHERE user_id = ?', 
                            (session['user_id'],)).fetchone()
        
        if existing:
            db.execute('''UPDATE settings SET 
                         rtsp_url = ?,
                         model_path = ?,
                         imgsz = ?,
                         frame_skip = ?,
                         recognition_interval = ?,
                         proximity_threshold = ?,
                         display_width = ?
                         WHERE user_id = ?''',
                      (data.get('rtsp_url'), data.get('model_path'),
                       data.get('imgsz'), data.get('frame_skip'),
                       data.get('recognition_interval'), data.get('proximity_threshold'),
                       data.get('display_width'), session['user_id']))
        else:
            db.execute('''INSERT INTO settings 
                         (user_id, rtsp_url, model_path, imgsz, frame_skip, 
                          recognition_interval, proximity_threshold, display_width)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                      (session['user_id'], data.get('rtsp_url'), data.get('model_path'),
                       data.get('imgsz'), data.get('frame_skip'),
                       data.get('recognition_interval'), data.get('proximity_threshold'),
                       data.get('display_width')))
        
        db.commit()
        db.close()
        
        return jsonify({'success': True, 'message': 'Settings saved'})
    
    user_settings = get_user_settings(session['user_id'])
    user = db.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    db.close()
    
    return render_template('settings_pwa.html', settings=user_settings, user=dict(user))


@app.route('/members', methods=['GET'])
@login_required
def members():
    """Members page"""
    db = get_db()
    members_list = db.execute('SELECT * FROM members ORDER BY created_at DESC').fetchall()
    db.close()
    
    return render_template('members_pwa.html', members=[dict(m) for m in members_list])


@app.route('/events', methods=['GET'])
@login_required
def events():
    """Recent theft events page"""
    db = get_db()
    events_list = db.execute('SELECT * FROM theft_events ORDER BY timestamp DESC LIMIT 50').fetchall()
    db.close()
    
    return render_template('events_pwa.html', events=[dict(e) for e in events_list])


@app.route('/api/members', methods=['GET', 'POST', 'DELETE'])
@login_required
def api_members():
    """Members API"""
    db = get_db()
    
    if request.method == 'GET':
        members_list = db.execute('SELECT * FROM members ORDER BY created_at DESC').fetchall()
        db.close()
        return jsonify([dict(m) for m in members_list])
    
    elif request.method == 'POST':
        data = request.get_json()
        name = data.get('name')
        
        # Add to members database
        db.execute('INSERT INTO members (name) VALUES (?)', (name,))
        db.commit()
        member_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]
        db.close()
        
        # Also register face if detector is running and has current frame
        face_registered = False
        if detector and current_frame is not None:
            try:
                success = detector.register_user(current_frame, name)
                face_registered = success
            except Exception as e:
                print(f"[WARNING] Failed to register face: {e}")
        
        message = f'Member {name} added'
        if face_registered:
            message += ' and face registered'
        else:
            message += ' (face not registered - ensure camera is running and person is visible)'
        
        return jsonify({'success': True, 'id': member_id, 'face_registered': face_registered, 'message': message})
    
    elif request.method == 'DELETE':
        member_id = request.args.get('id')
        
        # Get member name before deleting
        member = db.execute('SELECT name FROM members WHERE id = ?', (member_id,)).fetchone()
        member_name = member['name'] if member else None
        
        # Delete from members database
        db.execute('DELETE FROM members WHERE id = ?', (member_id,))
        db.commit()
        db.close()
        
        # Also remove from face database if detector exists
        if detector and member_name:
            try:
                # Remove from known_names and known_embeddings
                if member_name in detector.known_names:
                    idx = detector.known_names.index(member_name)
                    detector.known_names.pop(idx)
                    detector.known_embeddings.pop(idx)
                    detector.save_face_database()
                    print(f"[INFO] Removed {member_name} from face database")
            except Exception as e:
                print(f"[WARNING] Failed to remove from face database: {e}")
        
        return jsonify({'success': True, 'message': 'Member deleted'})


@app.route('/api/current-frame')
@login_required
def api_current_frame():
    """Get current camera frame for preview"""
    global current_frame
    
    if current_frame is not None:
        try:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', current_frame)
            if ret:
                return Response(buffer.tobytes(), mimetype='image/jpeg')
        except Exception as e:
            print(f"[ERROR] Failed to encode frame: {e}")
    
    return '', 404


@app.route('/api/theft-events')
@login_required
def api_theft_events():
    """Get theft events"""
    db = get_db()
    events = db.execute(
        'SELECT * FROM theft_events ORDER BY timestamp DESC LIMIT 50'
    ).fetchall()
    db.close()
    
    return jsonify([dict(e) for e in events])


@app.route('/api/start-detection', methods=['POST'])
@login_required
def start_detection():
    """Start theft detection"""
    global detector, detection_thread, is_detecting
    
    if is_detecting:
        return jsonify({'success': False, 'message': 'Detection already running'})
    
    try:
        user_settings = get_user_settings(session['user_id'])
        
        # Debug logging
        print(f"[DEBUG] Starting detection with settings: {user_settings}")
        
        detector = TheftDetector(
            model_path=user_settings['model_path'],
            imgsz=int(user_settings['imgsz']),
            frame_skip=int(user_settings['frame_skip']),
            rtsp_url=user_settings['rtsp_url'],
            recognition_interval=int(user_settings['recognition_interval']),
            proximity_threshold=int(user_settings['proximity_threshold']),
            display_width=int(user_settings['display_width']) if user_settings['display_width'] else None
        )
        
        is_detecting = True
        detection_thread = threading.Thread(target=detection_loop, daemon=True)
        detection_thread.start()
        
        return jsonify({'success': True, 'message': 'Detection started'})
    
    except KeyError as e:
        error_msg = f'Missing setting: {str(e)}'
        print(f"[ERROR] {error_msg}")
        return jsonify({'success': False, 'message': error_msg}), 500
    except Exception as e:
        error_msg = f'Failed to start: {str(e)}'
        print(f"[ERROR] {error_msg}")
        return jsonify({'success': False, 'message': error_msg}), 500


@app.route('/api/stop-detection', methods=['POST'])
@login_required
def stop_detection():
    """Stop theft detection"""
    global is_detecting
    
    is_detecting = False
    stop_video_recording()
    
    return jsonify({'success': True, 'message': 'Detection stopped'})


@app.route('/api/register-face', methods=['POST'])
@login_required
def register_face():
    """Register a face from current frame"""
    global current_frame, detector
    
    if current_frame is None or detector is None:
        return jsonify({'success': False, 'message': 'No frame available'}), 400
    
    data = request.get_json()
    name = data.get('name')
    
    if not name:
        return jsonify({'success': False, 'message': 'Name is required'}), 400
    
    success = detector.register_user(current_frame, name)
    
    if success:
        # Add to members database
        db = get_db()
        db.execute('INSERT INTO members (name) VALUES (?)', (name,))
        db.commit()
        db.close()
        
        return jsonify({'success': True, 'message': f'User {name} registered'})
    
    return jsonify({'success': False, 'message': 'Registration failed'}), 500


@app.route('/videos/<filename>')
@login_required
def serve_video(filename):
    """Serve video files"""
    response = send_from_directory(app.config['VIDEO_FOLDER'], filename)
    response.headers['Content-Type'] = 'video/mp4'
    response.headers['Accept-Ranges'] = 'bytes'
    return response


@app.route('/manifest.json')
def manifest():
    """PWA manifest"""
    return jsonify({
        "name": "AI Surveillance - Theft Detection",
        "short_name": "AI Surveillance",
        "description": "AI-powered theft detection surveillance system",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#1a1d2e",
        "theme_color": "#2c3e50",
        "icons": [
            {
                "src": "/static/icon-192.png",
                "sizes": "192x192",
                "type": "image/png"
            },
            {
                "src": "/static/icon-512.png",
                "sizes": "512x512",
                "type": "image/png"
            }
        ]
    })


@app.route('/sw.js')
def service_worker():
    """Service worker"""
    return send_from_directory('static', 'sw.js')


def detection_loop():
    """Main detection loop running in background thread"""
    global detector, is_detecting, current_frame, recording_theft, video_writer, theft_start_time
    
    if not detector:
        print("[ERROR] Detector not initialized")
        return
    
    # Connect to RTSP stream
    src = detector.rtsp_url
    if isinstance(src, str) and src.startswith('rtsp://'):
        cap = detector.connect_rtsp(url=src)
    else:
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        socketio.emit('error', {'message': 'Failed to connect to camera'})
        is_detecting = False
        return
    
    frame_count = 0
    prev_time = time.time()
    
    while is_detecting:
        ret, frame = cap.read()
        
        if not ret:
            time.sleep(0.1)
            continue
        
        current_frame = frame.copy()
        frame_count += 1
        
        # Run YOLO detection
        run_infer = True
        if detector.frame_skip > 0 and (frame_count % (detector.frame_skip + 1)) != 1:
            run_infer = False
        
        if run_infer:
            results = detector.model(
                frame, conf=0.5,
                classes=[detector.person_class_id, detector.backpack_class_id,
                        detector.handbag_class_id, detector.suitcase_class_id],
                imgsz=detector.imgsz, device=detector.device,
                half=detector.half, verbose=False
            )
            detector._last_results = results
        else:
            results = detector._last_results if detector._last_results is not None else []
        
        # Run recognition
        run_recognition_now = (frame_count % detector.recognition_interval) == 0
        
        # Draw detections and check for theft
        annotated_frame, person_count, bag_count, theft_count = detector.draw_detections(
            frame, results, recognize=True, run_recognition=run_recognition_now
        )
        
        # Handle theft event
        if theft_count > 0:
            if not recording_theft:
                # Start recording
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                video_filename = f'theft_{timestamp}.mp4'
                video_path = start_video_recording(video_filename)
                
                # Save to database
                db = get_db()
                db.execute('''INSERT INTO theft_events 
                             (person_name, bag_count, video_path)
                             VALUES (?, ?, ?)''',
                          ('Unknown', bag_count, video_filename))
                db.commit()
                event_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]
                db.close()
                
                # Send notifications
                send_notification(f'🚨 THEFT ALERT! Unknown person detected with {bag_count} bag(s)', 'danger')
                socketio.emit('theft_alert', {
                    'timestamp': timestamp,
                    'person_count': person_count,
                    'bag_count': bag_count,
                    'video_path': video_filename
                })
                
                # Trigger emergency call if auto-enabled (use first user by default)
                trigger_emergency_call(1, event_id)
        
        # Record video if theft in progress
        if recording_theft and video_writer:
            # Resize for recording
            record_frame = cv2.resize(annotated_frame, (640, 360))
            video_writer.write(record_frame)
            
            # Stop recording after 60 seconds
            if time.time() - theft_start_time > 60:
                stop_video_recording()
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
        prev_time = current_time
        
        # Resize for streaming
        if detector.display_width:
            height, width = annotated_frame.shape[:2]
            display_height = int(height * (detector.display_width / width))
            annotated_frame = cv2.resize(annotated_frame, (detector.display_width, display_height))
        
        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Emit frame to clients
        socketio.emit('video_frame', {
            'frame': frame_base64,
            'fps': round(fps, 1),
            'persons': person_count,
            'bags': bag_count,
            'alerts': theft_count,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        time.sleep(0.01)  # Small delay
    
    cap.release()
    stop_video_recording()


# SocketIO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('[INFO] Client connected')
    emit('connected', {'message': 'Connected to server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('[INFO] Client disconnected')


if __name__ == '__main__':
    print("[INFO] Starting AI Surveillance PWA...")
    print("[INFO] Access at http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
