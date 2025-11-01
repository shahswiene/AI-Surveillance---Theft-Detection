"""
Person Tracking and Face Recognition System using imgbeddings
Tracks people using YOLO11, detects faces with Haar Cascades, and recognizes registered users
"""

import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import os
import json
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Any, Optional
import torch


class PersonTrackerImgbed:
    def __init__(self, model_path='yolo11n.pt', device=None, imgsz=640, 
                 frame_skip=0, half=False, rtsp_url=None, recognition_interval=30, display_width=None):
        """
        Initialize person tracker with face recognition using imgbeddings
        
        Args:
            model_path: Path to YOLO11 model
            device: Inference device
            imgsz: Inference image size
            frame_skip: Frames to skip
            half: Use FP16
            rtsp_url: RTSP stream URL
            recognition_interval: Run face recognition every N frames
        """
        print(f"[INFO] Loading YOLO11 model: {model_path}")
        self.model = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.frame_skip = max(0, int(frame_skip))
        self.half = bool(half)
        self.rtsp_url = rtsp_url
        self.recognition_interval = max(1, int(recognition_interval))
        self.display_width = display_width  # Resize display window for better FPS
        
        # COCO class ID for person is 0
        self.person_class_id = 0
        
        # Load Haar Cascades
        cascade_frontal = 'haarcascade_frontalface_default.xml'
        cascade_profile = 'haarcascade_profileface.xml'
        
        if not os.path.exists(cascade_frontal):
            cascade_frontal = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_profile):
            cascade_profile = cv2.data.haarcascades + 'haarcascade_profileface.xml'
        
        self.haar_cascade = cv2.CascadeClassifier(cascade_frontal)
        self.profile_cascade = cv2.CascadeClassifier(cascade_profile)
        
        if self.haar_cascade.empty() or self.profile_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascades")
        
        print("[INFO] Haar Cascades loaded")
        
        # Initialize CLIP model for face embeddings
        print("[INFO] Loading CLIP model for face recognition...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()
        print("[INFO] CLIP model loaded")
        
        # Face database
        self.face_db_path = 'face_database.json'
        self.known_embeddings = []
        self.known_names = []
        self.load_face_database()
        
        # Statistics
        self.frame_count = 0
        self.person_count = 0
        self.fps = 0
        self._last_results = None
        
        # Face box cache to reduce flicker between recognition intervals
        self._last_face_boxes = []  # list of dicts with keys: x,y,width,height,type
        self._face_cache_ttl = 30   # frames to keep cached faces when not updating
        self._face_cache_age = 0
        
        # Tracking
        self.last_recognition_names = {}
        
    def load_face_database(self):
        """Load registered faces from JSON database"""
        if os.path.exists(self.face_db_path):
            try:
                with open(self.face_db_path, 'r') as f:
                    data = json.load(f)
                    self.known_embeddings = [np.array(item['embedding'], dtype=np.float32) 
                                            for item in data]
                    self.known_names = [item['name'] for item in data]
                print(f"[INFO] Loaded {len(self.known_names)} registered users")
            except Exception as e:
                print(f"[ERROR] Failed to load face database: {e}")
                self.known_embeddings = []
                self.known_names = []
        else:
            print("[INFO] No face database found, starting fresh")
            
    def save_face_database(self):
        """Save face database to JSON file"""
        data = []
        for embedding, name in zip(self.known_embeddings, self.known_names):
            data.append({
                'name': name,
                'embedding': embedding.tolist()
            })
        
        with open(self.face_db_path, 'w') as f:
            json.dump(data, f)
        print(f"[INFO] Saved face database with {len(data)} users")
        
    def normalize_embedding(self, vector: np.ndarray) -> Optional[np.ndarray]:
        """Normalize embedding vector"""
        if vector is None:
            return None
        norm = np.linalg.norm(vector)
        if norm == 0:
            return None
        return vector / norm
    
    def compute_similarity(self, source: np.ndarray, target: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        if source is None or target is None:
            return 0.0
        
        normalized_source = self.normalize_embedding(source)
        normalized_target = self.normalize_embedding(target)
        
        if normalized_source is None or normalized_target is None:
            return 0.0
        
        return float(np.dot(normalized_source, normalized_target))
    
    def register_user(self, frame, name):
        """
        Register a new user from current frame
        
        Args:
            frame: Current video frame
            name: Name of the user to register
        """
        print(f"[INFO] Attempting to register user: {name}")
        
        if frame is None or frame.size == 0:
            print("[ERROR] Invalid frame")
            return False
        
        # Detect faces
        faces = self.detect_faces_haar(frame, [(0, 0, frame.shape[1], frame.shape[0])])
        
        if len(faces) == 0:
            print("[WARNING] No face detected in frame")
            return False
        
        if len(faces) > 1:
            print("[WARNING] Multiple faces detected, using the largest one")
        
        # Get largest face
        largest_face = max(faces, key=lambda f: f['width'] * f['height'])
        x, y, w, h = largest_face['x'], largest_face['y'], largest_face['width'], largest_face['height']
        
        # Crop face
        face_crop = frame[y:y+h, x:x+w]
        if face_crop.size == 0:
            print("[ERROR] Failed to crop face")
            return False
        
        # Convert to RGB and PIL Image
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)
        
        # Extract embedding using CLIP
        try:
            with torch.no_grad():
                inputs = self.clip_processor(images=pil_image, return_tensors="pt")
                image_features = self.clip_model.get_image_features(**inputs)
                embedding = image_features.squeeze().cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"[ERROR] Failed to extract embedding: {e}")
            return False
        
        # Add to database
        self.known_embeddings.append(embedding)
        self.known_names.append(name)
        self.save_face_database()
        
        print(f"[SUCCESS] User '{name}' registered successfully!")
        return True
    
    def recognize_face(self, embedding: np.ndarray, threshold: float = 0.82) -> Optional[str]:
        """
        Recognize a face from its embedding
        
        Args:
            embedding: Face embedding vector
            threshold: Similarity threshold (default: 0.82)
            
        Returns:
            Name of recognized person or None
        """
        if embedding is None or len(self.known_embeddings) == 0:
            return None
        
        best_match = None
        best_similarity = threshold
        
        for known_embedding, known_name in zip(self.known_embeddings, self.known_names):
            similarity = self.compute_similarity(embedding, known_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = known_name
        
        return best_match
    
    def detect_faces_haar(self, frame, person_boxes):
        """
        Detect faces using Haar Cascades within person bounding boxes
        
        Args:
            frame: Input frame
            person_boxes: List of person bounding boxes
            
        Returns:
            List of face detections with bbox
        """
        faces = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        
        for (px1, py1, px2, py2) in person_boxes:
            # Ensure valid coordinates
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(width, px2), min(height, py2)
            
            if px2 <= px1 or py2 <= py1:
                continue
            
            # Extract person ROI
            person_gray = gray[py1:py2, px1:px2]
            
            if person_gray.size == 0 or min(person_gray.shape[:2]) < 40:
                continue
            
            # Detect frontal faces
            try:
                frontal_faces = self.haar_cascade.detectMultiScale(
                    person_gray, scaleFactor=1.1, minNeighbors=5,
                    minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
                )
                for (x, y, w, h) in frontal_faces:
                    faces.append({
                        'x': px1 + x, 'y': py1 + y,
                        'width': w, 'height': h, 'type': 'frontal'
                    })
            except:
                pass
            
            # Detect profile faces
            try:
                profile_faces = self.profile_cascade.detectMultiScale(
                    person_gray, scaleFactor=1.1, minNeighbors=4,
                    minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
                )
                for (x, y, w, h) in profile_faces:
                    faces.append({
                        'x': px1 + x, 'y': py1 + y,
                        'width': w, 'height': h, 'type': 'profile'
                    })
            except:
                pass
        
        return faces
    
    def recognize_faces(self, frame, person_boxes):
        """
        Recognize faces in detected person bounding boxes
        
        Args:
            frame: Current video frame
            person_boxes: List of person bounding boxes
            
        Returns:
            Dictionary mapping box index to recognized name
        """
        if len(self.known_embeddings) == 0:
            return {}
        
        recognized = {}
        
        for idx, (x1, y1, x2, y2) in enumerate(person_boxes):
            # Extract person ROI
            person_roi = frame[y1:y2, x1:x2]
            
            if person_roi.size == 0:
                continue
            
            # Detect faces in this person's box
            faces = self.detect_faces_haar(frame, [(x1, y1, x2, y2)])
            
            if len(faces) == 0:
                continue
            
            # Use largest face
            largest_face = max(faces, key=lambda f: f['width'] * f['height'])
            fx, fy, fw, fh = largest_face['x'], largest_face['y'], largest_face['width'], largest_face['height']
            
            # Crop face
            face_crop = frame[fy:fy+fh, fx:fx+fw]
            if face_crop.size == 0:
                continue
            
            # Convert to RGB and PIL Image
            try:
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(face_rgb)
                
                # Extract embedding using CLIP
                with torch.no_grad():
                    inputs = self.clip_processor(images=pil_image, return_tensors="pt")
                    image_features = self.clip_model.get_image_features(**inputs)
                    embedding = image_features.squeeze().cpu().numpy().astype(np.float32)
                
                # Recognize
                name = self.recognize_face(embedding, threshold=0.82)
                if name:
                    recognized[idx] = name
                else:
                    recognized[idx] = "Unknown"
            except Exception as e:
                print(f"[WARNING] Recognition failed: {e}")
                continue
        
        return recognized
    
    def connect_rtsp(self, url=None):
        """Connect to RTSP stream"""
        target_url = url if url else self.rtsp_url
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
        """Draw person detections and face recognition results"""
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
        
        # Detect faces with Haar Cascades (only if enabled). To avoid flicker, cache results
        faces = []
        if detect_faces and run_recognition:
            faces = self.detect_faces_haar(frame, person_boxes)
            # Update cache
            self._last_face_boxes = faces
            self._face_cache_age = self._face_cache_ttl
        elif detect_faces and not run_recognition and self._face_cache_age > 0:
            # Reuse cached faces to avoid flicker between recognition intervals
            faces = self._last_face_boxes
            self._face_cache_age -= 1
        
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
            
            # Draw label background
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (x1, y1 - label_height - 10), 
                         (x1 + label_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw face boxes
        for face in faces:
            fx, fy, fw, fh = face['x'], face['y'], face['width'], face['height']
            face_color = (255, 0, 0) if face['type'] == 'frontal' else (0, 165, 255)
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), face_color, 1)
        
        return frame, person_count
    
    def add_info_overlay(self, frame, person_count):
        """Add information overlay to frame"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info_text = [
            f"Persons Detected: {person_count}",
            f"Registered Users: {len(self.known_names)}",
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
        """Run person tracking and face recognition"""
        src = self.rtsp_url if self.rtsp_url else source
        
        if isinstance(src, str) and src.startswith('rtsp://'):
            cap = self.connect_rtsp(url=src)
        else:
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {src}")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"[INFO] Video properties: {frame_width}x{frame_height} @ {fps_input} FPS")
        print("[INFO] Starting tracking... Press 'q' to quit, 'r' to register user")
        
        prev_time = time.time()
        registration_mode = False
        original_frame = None
        
        try:
            while True:
                ret, frame = cap.read()
                
                if ret:
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
                    # Resize display for better FPS if display_width is set
                    display_frame = annotated_frame
                    if self.display_width:
                        height, width = annotated_frame.shape[:2]
                        display_height = int(height * (self.display_width / width))
                        display_frame = cv2.resize(annotated_frame, (self.display_width, display_height))
                    
                    cv2.imshow('Person Tracking & Recognition', display_frame)
                    
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
            print(f"  Registered Users: {len(self.known_names)}")
            print(f"  Average FPS: {self.fps:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Person Tracking with Face Recognition (imgbeddings)')
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
    parser.add_argument('--display-width', type=int, default=None,
                        help='Display window width in pixels for faster rendering (e.g., 960, 640). Default: original size')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a number
    source = args.source
    if source.isdigit():
        source = int(source)
    
    # Initialize tracker
    tracker = PersonTrackerImgbed(
        model_path=args.model,
        device=args.device,
        imgsz=args.imgsz,
        frame_skip=args.frame_skip,
        half=args.half,
        rtsp_url=args.rtsp,
        recognition_interval=args.recognition_interval,
        display_width=args.display_width
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
