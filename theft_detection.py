"""
Theft Detection System
Combines person tracking, face recognition, and bag detection to alert when unknown persons take bags
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
from typing import List, Dict, Any, Optional, Tuple
import torch


class TheftDetector:
    def __init__(self, model_path='yolo11n.pt', device=None, imgsz=640, 
                 frame_skip=0, half=False, rtsp_url=None, recognition_interval=30,
                 proximity_threshold=100, display_width=None):
        """
        Initialize theft detection system
        
        Args:
            model_path: Path to YOLO11 model
            device: Inference device
            imgsz: Inference image size
            frame_skip: Frames to skip
            half: Use FP16
            rtsp_url: RTSP stream URL
            recognition_interval: Run face recognition every N frames
            proximity_threshold: Distance threshold (pixels) for person-bag association
        """
        print(f"[INFO] Loading YOLO11 model: {model_path}")
        self.model = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.frame_skip = max(0, int(frame_skip))
        self.half = bool(half)
        self.rtsp_url = rtsp_url
        self.recognition_interval = max(1, int(recognition_interval))
        self.proximity_threshold = proximity_threshold
        self.display_width = display_width  # Resize display window for better FPS
        
        # COCO class IDs
        self.person_class_id = 0
        self.backpack_class_id = 24
        self.handbag_class_id = 26
        self.suitcase_class_id = 28
        
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
        self.bag_count = 0
        self.theft_alert_count = 0
        self.fps = 0
        self._last_results = None
        
        # Tracking
        self.last_recognition_names = {}
        self._last_face_boxes = []
        self._face_cache_ttl = 30
        self._face_cache_age = 0
        
        # Theft detection state
        self.theft_alerts = {}  # {person_idx: {"start_time": timestamp, "bag_id": bag_idx}}
        self.alert_cooldown = 5  # seconds before re-alerting same person
        
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
        """Register a new user from current frame"""
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
        """Recognize a face from its embedding"""
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
        """Detect faces using Haar Cascades within person bounding boxes"""
        faces = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        
        for (px1, py1, px2, py2) in person_boxes:
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(width, px2), min(height, py2)
            
            if px2 <= px1 or py2 <= py1:
                continue
            
            person_gray = gray[py1:py2, px1:px2]
            
            if person_gray.size == 0 or min(person_gray.shape[:2]) < 40:
                continue
            
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
        
        return faces
    
    def recognize_faces(self, frame, person_boxes):
        """Recognize faces in detected person bounding boxes"""
        if len(self.known_embeddings) == 0:
            return {}
        
        recognized = {}
        
        for idx, (x1, y1, x2, y2) in enumerate(person_boxes):
            person_roi = frame[y1:y2, x1:x2]
            
            if person_roi.size == 0:
                continue
            
            faces = self.detect_faces_haar(frame, [(x1, y1, x2, y2)])
            
            if len(faces) == 0:
                continue
            
            largest_face = max(faces, key=lambda f: f['width'] * f['height'])
            fx, fy, fw, fh = largest_face['x'], largest_face['y'], largest_face['width'], largest_face['height']
            
            face_crop = frame[fy:fy+fh, fx:fx+fw]
            if face_crop.size == 0:
                continue
            
            try:
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(face_rgb)
                
                with torch.no_grad():
                    inputs = self.clip_processor(images=pil_image, return_tensors="pt")
                    image_features = self.clip_model.get_image_features(**inputs)
                    embedding = image_features.squeeze().cpu().numpy().astype(np.float32)
                
                name = self.recognize_face(embedding, threshold=0.82)
                if name:
                    recognized[idx] = name
                else:
                    recognized[idx] = "Unknown"
            except Exception as e:
                continue
        
        return recognized
    
    def compute_box_center(self, box: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Compute center point of a bounding box"""
        x1, y1, x2, y2 = box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return cx, cy
    
    def compute_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union (IoU) between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def detect_person_bag_association(self, person_boxes, bag_boxes):
        """
        Detect which persons are associated with which bags
        Returns: Dict[person_idx, List[bag_idx]]
        """
        associations = {}
        
        for pidx, pbox in enumerate(person_boxes):
            associated_bags = []
            px1, py1, px2, py2 = pbox
            pcx, pcy = self.compute_box_center(pbox)
            
            for bidx, bbox in enumerate(bag_boxes):
                bx1, by1, bx2, by2 = bbox
                bcx, bcy = self.compute_box_center(bbox)
                
                # Check IoU (person holding bag)
                iou = self.compute_iou(pbox, bbox)
                if iou > 0.05:  # Even small overlap indicates holding
                    associated_bags.append(bidx)
                    continue
                
                # Check proximity (bag near person)
                distance = np.sqrt((pcx - bcx)**2 + (pcy - bcy)**2)
                if distance < self.proximity_threshold:
                    associated_bags.append(bidx)
                    continue
                
                # Check if bag is below person (likely carrying)
                if bx1 >= px1 and bx2 <= px2 and by1 >= py1 and by1 <= py2 + 100:
                    associated_bags.append(bidx)
            
            if associated_bags:
                associations[pidx] = associated_bags
        
        return associations
    
    def trigger_alert(self, person_idx, person_name, bag_count):
        """Trigger theft alert"""
        current_time = time.time()
        
        # Check cooldown
        if person_idx in self.theft_alerts:
            last_alert_time = self.theft_alerts[person_idx].get("start_time", 0)
            if current_time - last_alert_time < self.alert_cooldown:
                return False
        
        # Update alert state
        self.theft_alerts[person_idx] = {
            "start_time": current_time,
            "person_name": person_name,
            "bag_count": bag_count
        }
        
        self.theft_alert_count += 1
        
        # Note: Alert sound is played by frontend (alarm.wav)
        # Backend only logs the event
        
        # Log alert
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_msg = f"[THEFT ALERT] {timestamp} - {person_name} detected with {bag_count} bag(s)!"
        print(alert_msg)
        
        # Save to log file
        with open("theft_alerts.log", "a") as f:
            f.write(f"{alert_msg}\n")
        
        return True
    
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
    
    def draw_detections(self, frame, results, recognize=True, run_recognition=False):
        """Draw person and bag detections with theft alerts"""
        person_boxes = []
        bag_boxes = []
        
        # Extract bounding boxes
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if confidence < 0.5:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                if class_id == self.person_class_id:
                    person_boxes.append((x1, y1, x2, y2))
                elif class_id in [self.backpack_class_id, self.handbag_class_id, self.suitcase_class_id]:
                    bag_boxes.append((x1, y1, x2, y2))
        
        # Recognize faces
        recognized = {}
        if recognize and len(person_boxes) > 0 and run_recognition:
            recognized = self.recognize_faces(frame, person_boxes)
            self.last_recognition_names = recognized
        elif recognize:
            recognized = self.last_recognition_names
        
        # Detect person-bag associations
        associations = self.detect_person_bag_association(person_boxes, bag_boxes)
        
        # Check for theft (unknown person with bag)
        theft_detected = set()
        for pidx, bag_indices in associations.items():
            person_name = recognized.get(pidx, "Unknown")
            if person_name == "Unknown" and len(bag_indices) > 0:
                if self.trigger_alert(pidx, person_name, len(bag_indices)):
                    theft_detected.add(pidx)
        
        # Draw person boxes
        for idx, (x1, y1, x2, y2) in enumerate(person_boxes):
            # Determine color based on theft detection
            if idx in theft_detected:
                color = (0, 0, 255)  # Red for theft alert
                thickness_box = 3
            else:
                color = (0, 255, 0)  # Green for normal
                thickness_box = 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness_box)
            
            # Label
            label = recognized.get(idx, f"Person {idx + 1}")
            if idx in associations:
                label += f" +{len(associations[idx])} bag(s)"
            
            # Draw label background (larger text, on right side)
            font_scale = 1.2
            thickness = 3
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # Position on right side
            label_x = x2 - label_width - 5
            label_y = y1 + label_height + 10
            
            # Black background for better visibility
            cv2.rectangle(frame, (label_x - 5, label_y - label_height - 10), 
                         (label_x + label_width + 5, label_y + 5), (0, 0, 0), -1)
            # Colored border
            cv2.rectangle(frame, (label_x - 5, label_y - label_height - 10), 
                         (label_x + label_width + 5, label_y + 5), color, 2)
            cv2.putText(frame, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # Theft alert text
            if idx in theft_detected:
                cv2.putText(frame, "THEFT ALERT!", (x1, y2 + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw bag boxes
        for idx, (x1, y1, x2, y2) in enumerate(bag_boxes):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            
            # Larger text on right side with black background
            bag_label = f"Bag {idx + 1}"
            font_scale = 1.0
            thickness = 2
            (label_width, label_height), _ = cv2.getTextSize(
                bag_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            label_x = x2 - label_width - 5
            label_y = y1 + label_height + 5
            
            # Black background
            cv2.rectangle(frame, (label_x - 5, label_y - label_height - 5), 
                         (label_x + label_width + 5, label_y + 5), (0, 0, 0), -1)
            # Orange border
            cv2.rectangle(frame, (label_x - 5, label_y - label_height - 5), 
                         (label_x + label_width + 5, label_y + 5), (255, 165, 0), 2)
            cv2.putText(frame, bag_label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return frame, len(person_boxes), len(bag_boxes), len(theft_detected)
    
    def add_info_overlay(self, frame, person_count, bag_count, theft_count):
        """Add information overlay to frame"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 160), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info_text = [
            f"Persons: {person_count} | Bags: {bag_count}",
            f"Registered Users: {len(self.known_names)}",
            f"Theft Alerts: {self.theft_alert_count}",
            f"Current Alerts: {theft_count}",
            f"FPS: {self.fps:.1f} | Frame: {self.frame_count}",
            f"Time: {timestamp}"
        ]
        
        y_offset = 30
        for text in info_text:
            color = (0, 0, 255) if "Alert" in text and theft_count > 0 else (0, 255, 0)
            cv2.putText(frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            y_offset += 22
        
        return frame
    
    def run_detection(self, source=0, display=True):
        """Run theft detection system"""
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
        print("[INFO] Starting theft detection... Press 'q' to quit, 'r' to register user")
        
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
                        frame, conf=0.5, 
                        classes=[self.person_class_id, self.backpack_class_id, 
                                self.handbag_class_id, self.suitcase_class_id],
                        imgsz=self.imgsz, device=self.device,
                        half=self.half, verbose=False
                    )
                    self._last_results = results
                else:
                    results = self._last_results if self._last_results is not None else []
                
                # Run recognition this frame?
                run_recognition_now = (self.frame_count % self.recognition_interval) == 0
                
                # Draw detections and check for theft
                annotated_frame, person_count, bag_count, theft_count = self.draw_detections(
                    frame, results, recognize=True, run_recognition=run_recognition_now
                )
                
                # Calculate FPS
                current_time = time.time()
                self.fps = 1 / (current_time - prev_time)
                prev_time = current_time
                
                # Add info overlay
                annotated_frame = self.add_info_overlay(annotated_frame, person_count, bag_count, theft_count)
                
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
                    
                    cv2.imshow('Theft Detection System', display_frame)
                    
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
                    print(f"[INFO] Frame {self.frame_count} | Persons: {person_count} | Bags: {bag_count} | Alerts: {theft_count} | FPS: {self.fps:.1f}")
        
        except KeyboardInterrupt:
            print("\n[INFO] Detection interrupted by user")
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            
            print(f"\n[INFO] Detection Summary:")
            print(f"  Total Frames: {self.frame_count}")
            print(f"  Total Theft Alerts: {self.theft_alert_count}")
            print(f"  Registered Users: {len(self.known_names)}")
            print(f"  Average FPS: {self.fps:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Theft Detection System')
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
    parser.add_argument('--recognition-interval', type=int, default=30,
                        help='Run face recognition every N frames (default: 30)')
    parser.add_argument('--proximity', type=int, default=100,
                        help='Proximity threshold for person-bag association (default: 100 pixels)')
    parser.add_argument('--display-width', type=int, default=None,
                        help='Display window width in pixels for faster rendering (e.g., 960, 640). Default: original size')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a number
    source = args.source
    if source.isdigit():
        source = int(source)
    
    # Initialize detector
    detector = TheftDetector(
        model_path=args.model,
        device=args.device,
        imgsz=args.imgsz,
        frame_skip=args.frame_skip,
        half=args.half,
        rtsp_url=args.rtsp,
        recognition_interval=args.recognition_interval,
        proximity_threshold=args.proximity,
        display_width=args.display_width
    )
    
    # Run detection
    detector.run_detection(
        source=source,
        display=not args.no_display
    )


if __name__ == '__main__':
    main()
