"""
Test script to verify settings and detector initialization
"""
import sqlite3

# Connect to database
conn = sqlite3.connect('surveillance.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Check users
print("=== USERS ===")
users = cursor.execute("SELECT * FROM users").fetchall()
for user in users:
    print(f"ID: {user['id']}, Username: {user['username']}")

# Check settings
print("\n=== SETTINGS ===")
settings = cursor.execute("SELECT * FROM settings").fetchall()
if len(settings) == 0:
    print("No settings found. Creating default settings for user_id=1...")
    cursor.execute('''INSERT INTO settings 
                     (user_id, rtsp_url, model_path, imgsz, frame_skip, 
                      recognition_interval, proximity_threshold, display_width)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  (1, 'rtsp://192.168.100.13:8080/h264.sdp', 'yolo11n.pt',
                   320, 2, 60, 100, 640))
    conn.commit()
    print("✅ Default settings created!")
    settings = cursor.execute("SELECT * FROM settings").fetchall()

for setting in settings:
    print(f"\nUser ID: {setting['user_id']}")
    print(f"  RTSP URL: {setting['rtsp_url']}")
    print(f"  Model: {setting['model_path']}")
    print(f"  Image Size: {setting['imgsz']} (type: {type(setting['imgsz'])})")
    print(f"  Frame Skip: {setting['frame_skip']}")
    print(f"  Recognition Interval: {setting['recognition_interval']}")
    print(f"  Proximity: {setting['proximity_threshold']}")
    print(f"  Display Width: {setting['display_width']}")

conn.close()

# Test TheftDetector initialization
print("\n=== TESTING DETECTOR INITIALIZATION ===")
try:
    from theft_detection import TheftDetector
    
    detector = TheftDetector(
        model_path='yolo11n.pt',
        imgsz=320,
        frame_skip=2,
        rtsp_url='rtsp://192.168.100.13:8080/h264.sdp',
        recognition_interval=60,
        proximity_threshold=100,
        display_width=640
    )
    print("✅ TheftDetector initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize TheftDetector: {e}")

print("\n=== TEST COMPLETE ===")
