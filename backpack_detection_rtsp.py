"""
YOLO11 Backpack Detection with RTSP Stream Support
Detects backpacks in real-time from RTSP camera streams using Ultralytics YOLO11
"""

import cv2
import argparse
from ultralytics import YOLO
import time
from datetime import datetime
import numpy as np


class BackpackDetector:
    def __init__(self, model_path='yolo11n.pt', conf_threshold=0.5, rtsp_url=None, device=None, imgsz=640, frame_skip=0, half=False):
        """
        Initialize the backpack detector
        
        Args:
            model_path: Path to YOLO11 model weights
            conf_threshold: Confidence threshold for detections
            rtsp_url: RTSP stream URL
            device: Inference device (e.g., 'cpu', '0' for first CUDA device)
            imgsz: Inference image size (int, e.g., 640)
            frame_skip: Number of frames to skip between inferences (0 = no skip)
            half: Use FP16 (half precision) if supported (GPU only)
        """
        print(f"[INFO] Loading YOLO11 model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.rtsp_url = rtsp_url
        self.device = device
        self.imgsz = imgsz
        self.frame_skip = max(0, int(frame_skip))
        self.half = bool(half)
        
        # COCO dataset class ID for backpack is 24
        self.backpack_class_id = 24
        self.class_names = self.model.names
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.fps = 0
        
        # Inference cache (reuse results on skipped frames)
        self._last_results = None
        
    def connect_rtsp(self, url=None):
        """Connect to RTSP stream"""
        target_url = url or self.rtsp_url
        if not target_url:
            raise ConnectionError(f"Failed to connect to RTSP stream: {target_url}")
        print(f"[INFO] Connecting to RTSP stream: {target_url}")
        cap = cv2.VideoCapture(target_url)
        
        # Set buffer size to reduce latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            raise ConnectionError(f"Failed to connect to RTSP stream: {target_url}")
        
        print("[INFO] Successfully connected to RTSP stream")
        return cap
    
    def draw_detections(self, frame, results):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            results: YOLO detection results
            
        Returns:
            Annotated frame with detection count
        """
        backpack_count = 0
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get class ID and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Filter for backpack class only
                if class_id == self.backpack_class_id and confidence >= self.conf_threshold:
                    backpack_count += 1
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Prepare label
                    label = f"Backpack {confidence:.2f}"
                    
                    # Calculate label size and position
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Draw label background
                    cv2.rectangle(
                        frame,
                        (x1, y1 - label_height - 10),
                        (x1 + label_width, y1),
                        (0, 255, 0),
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2
                    )
        
        return frame, backpack_count
    
    def add_info_overlay(self, frame, backpack_count):
        """Add information overlay to frame"""
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Add text information
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info_text = [
            f"Backpacks Detected: {backpack_count}",
            f"FPS: {self.fps:.1f}",
            f"Frame: {self.frame_count}",
            f"Time: {timestamp}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(
                frame,
                text,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
            y_offset += 25
        
        return frame
    
    def run_detection(self, source=0, display=True, save_video=False, output_path='output.mp4'):
        """
        Run backpack detection on video source
        
        Args:
            source: Video source (0 for webcam, RTSP URL, or video file path)
            display: Whether to display the video feed
            save_video: Whether to save the output video
            output_path: Path to save output video
        """
        # Resolve final source preference: explicit --rtsp overrides, otherwise use --source
        src = self.rtsp_url if self.rtsp_url else source
        
        # Open video source
        if isinstance(src, str) and src.startswith('rtsp://'):
            cap = self.connect_rtsp(url=src)
        else:
            cap = cv2.VideoCapture(src)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video source: {src}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"[INFO] Video properties: {frame_width}x{frame_height} @ {fps_input} FPS")
        
        # Setup video writer if saving
        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))
            print(f"[INFO] Saving output to: {output_path}")
        
        # FPS calculation
        prev_time = time.time()
        
        print("[INFO] Starting detection... Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("[WARNING] Failed to read frame. Reconnecting...")
                    if isinstance(src, str) and src.startswith('rtsp://'):
                        cap.release()
                        time.sleep(2)
                        cap = self.connect_rtsp(url=src)
                        continue
                    else:
                        break
                
                self.frame_count += 1

                # Decide whether to run inference or reuse last results
                run_infer = True
                if self.frame_skip > 0 and (self.frame_count % (self.frame_skip + 1)) != 1:
                    run_infer = False

                if run_infer:
                    results = self.model(
                        frame,
                        conf=self.conf_threshold,
                        classes=[self.backpack_class_id],
                        imgsz=self.imgsz,
                        device=self.device,
                        half=self.half,
                        verbose=False,
                    )
                    self._last_results = results
                else:
                    results = self._last_results if self._last_results is not None else []
                
                # Draw detections
                annotated_frame, backpack_count = self.draw_detections(frame, results)
                
                if backpack_count > 0:
                    self.detection_count += 1
                
                # Calculate FPS
                current_time = time.time()
                self.fps = 1 / (current_time - prev_time)
                prev_time = current_time
                
                # Add info overlay
                annotated_frame = self.add_info_overlay(annotated_frame, backpack_count)
                
                # Save frame if recording
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame
                if display:
                    cv2.imshow('YOLO11 Backpack Detection', annotated_frame)
                    
                    # Check for quit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[INFO] Quit signal received")
                        break
                
                # Print detection info every 30 frames
                if self.frame_count % 30 == 0:
                    print(f"[INFO] Frame {self.frame_count} | Backpacks: {backpack_count} | FPS: {self.fps:.1f}")
        
        except KeyboardInterrupt:
            print("\n[INFO] Detection interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            
            print(f"\n[INFO] Detection Summary:")
            print(f"  Total Frames: {self.frame_count}")
            print(f"  Frames with Detections: {self.detection_count}")
            print(f"  Average FPS: {self.fps:.1f}")


def main():
    parser = argparse.ArgumentParser(description='YOLO11 Backpack Detection with RTSP Support')
    parser.add_argument('--model', type=str, default='yolo11s.pt',
                        help='Path to YOLO11 model (default: yolo11s.pt)')
    parser.add_argument('--rtsp', type=str, default=None,
                        help='RTSP stream URL (e.g., rtsp://username:password@ip:port/stream)')
    parser.add_argument('--source', type=str, default='0',
                        help='Video source: 0 for webcam, or path to video file (default: 0)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable video display')
    parser.add_argument('--save', action='store_true',
                        help='Save output video')
    parser.add_argument('--output', type=str, default='backpack_detection_output.mp4',
                        help='Output video path (default: backpack_detection_output.mp4)')
    parser.add_argument('--device', type=str, default=None,
                        help="Inference device, e.g. 'cpu', '0' for CUDA:0")
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Inference image size (default: 640)')
    parser.add_argument('--frame-skip', type=int, default=0,
                        help='Frames to skip between inferences (default: 0)')
    parser.add_argument('--half', action='store_true',
                        help='Use half precision (FP16) if supported (GPU only)')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a number
    source = args.source
    if source.isdigit():
        source = int(source)
    
    # Initialize detector
    detector = BackpackDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        rtsp_url=args.rtsp,
        device=args.device,
        imgsz=args.imgsz,
        frame_skip=args.frame_skip,
        half=args.half,
    )
    
    # Run detection
    detector.run_detection(
        source=source,
        display=not args.no_display,
        save_video=args.save,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
