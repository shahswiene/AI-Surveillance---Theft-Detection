"""
Person Tracking and Face Recognition System
Tracks people using YOLO11, detects faces with Haar Cascades, and recognizes registered users
"""

import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import os
import pickle
import face_recognition


class PersonTracker:
    def __init__(self, model_path='yolo11n.pt', device=None, imgsz=640, 
                 frame_skip=0, half=False, rtsp_url=None, recognition_interval=30):
        """
        Initialize person tracker with face recognition
        
        Args:
            model_path: Path to YOLO11 model
            device: Inference device
            imgsz: Inference image size
            frame_skip: Frames to skip
            half: Use FP16
            rtsp_url: RTSP stream URL
            recognition_interval: Run face recognition every N frames (default: 30)
        """
        print(f"[INFO] Loading YOLO11 model: {model_path}")
        self.model = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.frame_skip = max(0, int(frame_skip))
        self.half = bool(half)
        self.rtsp_url = rtsp_url
        self.recognition_interval = max(1, int(recognition_interval))
        
        # COCO class ID for person is 0
        self.person_class_id = 0
        
        # Load Haar Cascades
        cascade_path_frontal = 'haarcascade_frontalface_default.xml'
        cascade_path_profile = 'haarcascade_profileface.xml'
        
        if not os.path.exists(cascade_path_frontal):
            print(f"[WARNING] {cascade_path_frontal} not found, downloading...")
            cascade_path_frontal = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if not os.path.exists(cascade_path_profile):
            print(f"[WARNING] {cascade_path_profile} not found, downloading...")
            cascade_path_profile = cv2.data.haarcascades + 'haarcascade_profileface.xml'
        
        self.face_cascade_frontal = cv2.CascadeClassifier(cascade_path_frontal)
        self.face_cascade_profile = cv2.CascadeClassifier(cascade_path_profile)
        
        print("[INFO] Haar Cascades loaded")
        
        # Face recognition database
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_db_path = 'face_database.pkl'
        self.load_face_database()
        
        # Statistics
        self.frame_count = 0
        self.person_count = 0
        self.fps = 0
        self._last_results = None
        
        # Tracking
        self.tracked_persons = {}
        self.next_person_id = 1
        self.last_recognition_names = {}  # Cache recognition results
        
    def load_face_database(self):
        """Load registered faces from database"""
        if os.path.exists(self.face_db_path):
            try:
                with open(self.face_db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print(f"[INFO] Loaded {len(self.known_face_names)} registered users")
            except Exception as e:
                print(f"[WARNING] Failed to load face database: {e}")
        else:
            print("[INFO] No face database found, starting fresh")
    
    def save_face_database(self):
        """Save registered faces to database"""
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }
            with open(self.face_db_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"[INFO] Face database saved with {len(self.known_face_names)} users")
        except Exception as e:
            print(f"[ERROR] Failed to save face database: {e}")
    
    def register_user(self, frame, name):
        """
        Register a new user from current frame
        
        Args:
            frame: Current video frame
            name: Name of the user to register
        """
        print(f"[INFO] Attempting to register user: {name}")
        
        # Debug: print frame info
        print(f"[DEBUG] Frame shape: {frame.shape}, dtype: {frame.dtype}")
        
        # Ensure frame is in correct format (BGR, uint8, 3 channels)
        if frame is None or frame.size == 0:
            print("[ERROR] Invalid frame")
            return False
            
        # Make a copy to avoid modifying original
        frame_copy = frame.copy()
        
        # Ensure uint8 type
        if frame_copy.dtype != np.uint8:
            print(f"[DEBUG] Converting from {frame_copy.dtype} to uint8")
            frame_copy = (frame_copy * 255).astype(np.uint8) if frame_copy.max() <= 1.0 else frame_copy.astype(np.uint8)
        
        # Ensure 3 channels (BGR)
        if len(frame_copy.shape) == 2:
            print("[DEBUG] Converting grayscale to BGR")
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_GRAY2BGR)
        elif len(frame_copy.shape) == 3 and frame_copy.shape[2] == 4:
            print("[DEBUG] Converting BGRA to BGR")
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGRA2BGR)
        
        print(f"[DEBUG] After conversion - shape: {frame_copy.shape}, dtype: {frame_copy.dtype}")
        
        # Convert to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        
        # Ensure contiguous array
        rgb_frame = np.ascontiguousarray(rgb_frame)
        
        print(f"[DEBUG] RGB frame - shape: {rgb_frame.shape}, dtype: {rgb_frame.dtype}, contiguous: {rgb_frame.flags['C_CONTIGUOUS']}")
        
        # Find faces
        try:
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')  # Use HOG model (CPU)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        except RuntimeError as e:
            print(f"[ERROR] face_recognition failed: {e}")
            print("[ERROR] This is a known issue with dlib-binary on Windows.")
            print("[ERROR] Registration is not available. Face recognition during tracking may also fail.")
            print("[INFO] To fix: Install Visual Studio Build Tools and reinstall dlib from source")
            return False
        
        if len(face_encodings) == 0:
            print("[WARNING] No face detected in frame")
            return False
        
        if len(face_encodings) > 1:
            print("[WARNING] Multiple faces detected, using the first one")
        
        # Register the first face
        self.known_face_encodings.append(face_encodings[0])
        self.known_face_names.append(name)
        self.save_face_database()
        
        print(f"[SUCCESS] User '{name}' registered successfully!")
        return True
    
    def recognize_faces(self, frame, person_boxes):
        """
        Recognize faces in detected person bounding boxes
        
        Args:
            frame: Current video frame
            person_boxes: List of person bounding boxes
            
        Returns:
            Dictionary mapping box index to recognized name
        """
        if len(self.known_face_encodings) == 0:
            return {}
        
        recognized = {}
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        for idx, (x1, y1, x2, y2) in enumerate(person_boxes):
            # Extract person ROI
            person_roi = rgb_frame[y1:y2, x1:x2]
            
            if person_roi.size == 0:
                continue
            
            # Find faces in person ROI
            try:
                face_locations = face_recognition.face_locations(person_roi, model='hog')
                face_encodings = face_recognition.face_encodings(person_roi, face_locations)
            except RuntimeError:
                # dlib-binary compatibility issue, skip this person
                continue
            
            for face_encoding in face_encodings:
                # Compare with known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding, tolerance=0.6
                )
                name = "Unknown"
                
                # Use the known face with smallest distance
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                
                recognized[idx] = name
                break  # Only recognize first face in person box
        
        return recognized
    
    def detect_faces_haar(self, frame, person_boxes):
        """
        Detect faces using Haar Cascades within person bounding boxes
        
        Args:
            frame: Current video frame
            person_boxes: List of person bounding boxes
            
        Returns:
            List of face bounding boxes (x, y, w, h) with person box index
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []
        
        for idx, (x1, y1, x2, y2) in enumerate(person_boxes):
            # Extract person ROI
            person_roi = gray[y1:y2, x1:x2]
            
            if person_roi.size == 0:
                continue
            
            # Detect frontal faces
            frontal_faces = self.face_cascade_frontal.detectMultiScale(
                person_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            # Detect profile faces
            profile_faces = self.face_cascade_profile.detectMultiScale(
                person_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            # Combine and convert to absolute coordinates
            for (x, y, w, h) in frontal_faces:
                faces.append((x1 + x, y1 + y, w, h, idx, 'frontal'))
            
            for (x, y, w, h) in profile_faces:
                faces.append((x1 + x, y1 + y, w, h, idx, 'profile'))
        
        return faces
    
    def connect_rtsp(self, url=None):
        """Connect to RTSP stream"""
        target_url = url or self.rtsp_url
        if not target_url:
            raise ConnectionError(f"No RTSP URL provided")
        print(f"[INFO] Connecting to RTSP stream: {target_url}")
        cap = cv2.VideoCapture(target_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            raise ConnectionError(f"Failed to connect to RTSP stream: {target_url}")
        
        print("[INFO] Successfully connected to RTSP stream")
        return cap
    
    def draw_detections(self, frame, results, recognize=True, run_recognition=False, detect_faces=True):
        """
        Draw person detections and face recognition results
        
        Args:
            frame: Input frame
            results: YOLO detection results
            recognize: Whether to perform face recognition
            run_recognition: Whether to actually run recognition this frame
            detect_faces: Whether to run Haar Cascade face detection
            
        Returns:
            Annotated frame and person count
        """
        person_count = 0
        person_boxes = []
        
        # Extract person bounding boxes
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if class_id == self.person_class_id and confidence >= 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_boxes.append((x1, y1, x2, y2))
                    person_count += 1
        
        # Recognize faces if enabled and it's time to run recognition
        recognized = {}
        if recognize and len(person_boxes) > 0 and run_recognition:
            recognized = self.recognize_faces(frame, person_boxes)
            self.last_recognition_names = recognized
        elif recognize:
            # Reuse last recognition results
            recognized = self.last_recognition_names
        
        # Detect faces with Haar Cascades (only if enabled and running recognition)
        faces = []
        if detect_faces and run_recognition:
            faces = self.detect_faces_haar(frame, person_boxes)
        
        # Draw person boxes
        for idx, (x1, y1, x2, y2) in enumerate(person_boxes):
            # Person box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label with recognition result
            label = f"Person {idx + 1}"
            if idx in recognized:
                label = f"{recognized[idx]}"
                color = (0, 255, 0) if recognized[idx] != "Unknown" else (0, 165, 255)
            else:
                color = (0, 255, 0)
            
            # Draw label
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (x1, y1 - label_height - 10),
                         (x1 + label_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw face detections
        for (x, y, w, h, person_idx, face_type) in faces:
            color = (255, 0, 0) if face_type == 'frontal' else (255, 165, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, face_type, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame, person_count
    
    def add_info_overlay(self, frame, person_count):
        """Add information overlay to frame"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info_text = [
            f"Persons Detected: {person_count}",
            f"Registered Users: {len(self.known_face_names)}",
            f"FPS: {self.fps:.1f}",
            f"Frame: {self.frame_count}",
            f"Time: {timestamp}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            y_offset += 25
        
        return frame
    
    def run_tracking(self, source=0, display=True, recognize=True, detect_faces=True):
        """
        Run person tracking and face recognition
        
        Args:
            source: Video source
            display: Whether to display video
            recognize: Whether to perform face recognition
        """
        src = self.rtsp_url if self.rtsp_url else source
        
        if isinstance(src, str) and src.startswith('rtsp://'):
            cap = self.connect_rtsp(url=src)
        else:
            cap = cv2.VideoCapture(src)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video source: {src}")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"[INFO] Video properties: {frame_width}x{frame_height} @ {fps_input} FPS")
        print("[INFO] Starting tracking... Press 'q' to quit, 'r' to register user")
        
        prev_time = time.time()
        registration_mode = False
        original_frame = None  # Store original unmodified frame
        
        try:
            while True:
                ret, frame = cap.read()
                
                if ret:
                    # Save original frame before any modifications
                    original_frame = frame.copy()
                
                if not ret:
                    print("[WARNING] Failed to read frame")
                    if isinstance(src, str) and src.startswith('rtsp://'):
                        cap.release()
                        time.sleep(2)
                        cap = self.connect_rtsp(url=src)
                        continue
                    else:
                        break
                
                self.frame_count += 1
                
                # Run inference
                run_infer = True
                if self.frame_skip > 0 and (self.frame_count % (self.frame_skip + 1)) != 1:
                    run_infer = False
                
                if run_infer:
                    results = self.model(
                        frame, conf=0.5, classes=[self.person_class_id],
                        imgsz=self.imgsz, device=self.device,
                        half=self.half, verbose=False
                    )
                    self._last_results = results
                else:
                    results = self._last_results if self._last_results is not None else []
                
                # Determine if we should run recognition this frame
                run_recognition_now = (self.frame_count % self.recognition_interval) == 0
                
                # Draw detections
                annotated_frame, person_count = self.draw_detections(
                    frame, results, recognize=recognize, run_recognition=run_recognition_now,
                    detect_faces=detect_faces
                )
                
                if person_count > 0:
                    self.person_count += 1
                
                # Calculate FPS
                current_time = time.time()
                self.fps = 1 / (current_time - prev_time)
                prev_time = current_time
                
                # Add info overlay
                annotated_frame = self.add_info_overlay(annotated_frame, person_count)
                
                # Registration mode indicator
                if registration_mode:
                    cv2.putText(annotated_frame, "REGISTRATION MODE - Press SPACE to capture",
                               (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 0, 255), 2)
                
                # Display
                if display:
                    cv2.imshow('Person Tracking & Recognition', annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        print("[INFO] Quit signal received")
                        break
                    elif key == ord('r'):
                        registration_mode = True
                        print("[INFO] Registration mode activated - Press SPACE to capture face")
                    elif key == ord(' ') and registration_mode:
                        name = input("Enter name for registration: ")
                        if name:
                            # Use original unmodified frame for registration
                            success = self.register_user(original_frame if original_frame is not None else frame, name)
                            if success:
                                registration_mode = False
                        else:
                            print("[WARNING] Name cannot be empty")
                
                # Print info every 30 frames
                if self.frame_count % 30 == 0:
                    print(f"[INFO] Frame {self.frame_count} | Persons: {person_count} | FPS: {self.fps:.1f}")
        
        except KeyboardInterrupt:
            print("\n[INFO] Tracking interrupted by user")
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            
            print(f"\n[INFO] Tracking Summary:")
            print(f"  Total Frames: {self.frame_count}")
            print(f"  Frames with Persons: {self.person_count}")
            print(f"  Registered Users: {len(self.known_face_names)}")
            print(f"  Average FPS: {self.fps:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Person Tracking with Face Recognition')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                        help='Path to YOLO11 model (default: yolo11n.pt)')
    parser.add_argument('--rtsp', type=str, default=None,
                        help='RTSP stream URL')
    parser.add_argument('--source', type=str, default='0',
                        help='Video source: 0 for webcam (default: 0)')
    parser.add_argument('--device', type=str, default=None,
                        help="Inference device, e.g. 'cpu', '0' for CUDA:0")
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Inference image size (default: 640)')
    parser.add_argument('--frame-skip', type=int, default=0,
                        help='Frames to skip between inferences (default: 0)')
    parser.add_argument('--half', action='store_true',
                        help='Use half precision (FP16) if supported')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable video display')
    parser.add_argument('--no-recognize', action='store_true',
                        help='Disable face recognition')
    parser.add_argument('--recognition-interval', type=int, default=30,
                        help='Run face recognition every N frames (default: 30)')
    parser.add_argument('--no-face-detect', action='store_true',
                        help='Disable Haar Cascade face detection (faster)')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a number
    source = args.source
    if source.isdigit():
        source = int(source)
    
    # Initialize tracker
    tracker = PersonTracker(
        model_path=args.model,
        device=args.device,
        imgsz=args.imgsz,
        frame_skip=args.frame_skip,
        half=args.half,
        rtsp_url=args.rtsp,
        recognition_interval=args.recognition_interval
    )
    
    # Run tracking
    tracker.run_tracking(
        source=source,
        display=not args.no_display,
        recognize=not args.no_recognize,
        detect_faces=not args.no_face_detect
    )


if __name__ == '__main__':
    main()
