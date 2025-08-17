import cv2
import os
import glob
import numpy as np
from ultralytics import YOLO
import threading
import queue
import mediapipe as mp
import torch

from sr_model import SRModel
from models import SCRFD, ArcFace
from utils.helpers import compute_similarity

# --- Helper Functions based on super.py logic ---
def find_pose_for_face(face_bbox, pose_landmarks, frame_shape):
    """Finds the pose that corresponds to a given face bounding box."""
    if not pose_landmarks:
        return None
    
    face_x1, face_y1, face_x2, face_y2 = face_bbox
    nose = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
    
    # Convert nose landmark to pixel coordinates
    nose_px = int(nose.x * frame_shape[1]), int(nose.y * frame_shape[0])

    # Check if the nose is inside the face bounding box
    if face_x1 <= nose_px[0] <= face_x2 and face_y1 <= nose_px[1] <= face_y2:
        return pose_landmarks
    return None

def is_gun_near_hand(gun_bbox, hand_landmark, frame_shape, proximity_thresh=100):
    """Checks if a gun's center is near a hand landmark."""
    if not hand_landmark or hand_landmark.visibility < 0.5:
        return False

    gun_center_x = (gun_bbox[0] + gun_bbox[2]) / 2
    gun_center_y = (gun_bbox[1] + gun_bbox[3]) / 2

    hand_px = int(hand_landmark.x * frame_shape[1]), int(hand_landmark.y * frame_shape[0])

    distance = np.sqrt((gun_center_x - hand_px[0])**2 + (gun_center_y - hand_px[1])**2)
    return distance < proximity_thresh

class CombinedDetector:
    def __init__(self, det_weight, rec_weight, gun_weight, faces_dir, sr_model_path, similarity_thresh=0.4):
        # Initialize models
        self.face_detector = SCRFD(det_weight)
        self.face_recognizer = ArcFace(rec_weight)
        self.gun_detector = YOLO(gun_weight)
        self.sr_model = SRModel(sr_model_path)
        self.mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        
        self.similarity_thresh = similarity_thresh
        self.known_face_embs = []
        self.known_face_names = []
        self.identified_threats = set()

        # Load known faces
        self._load_known_faces(faces_dir)

    def _load_known_faces(self, faces_dir):
        if not os.path.isdir(faces_dir):
            print(f"Error: Faces directory not found at {faces_dir}")
            return

        print("Loading known faces...")
        for image_path in glob.glob(os.path.join(faces_dir, "*")):
            try:
                img = cv2.imread(image_path)
                if img is None: continue
                
                bboxes, kpss = self.face_detector.detect(img)
                if bboxes.shape[0] > 0:
                    face_kps = kpss[0]
                    emb = self.face_recognizer(img, face_kps)
                    face_name = os.path.splitext(os.path.basename(image_path))[0]
                    self.known_face_embs.append(emb)
                    self.known_face_names.append(face_name)
                    print(f"- Loaded face for {face_name}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        print(f"Finished loading {len(self.known_face_names)} known faces.")

    def detect_and_recognize(self, frame, gun_low_thresh=0.25, gun_high_thresh=0.75):
        # --- Face Recognition ---
        face_detections = []
        bboxes, kpss = self.face_detector.detect(frame)
        if bboxes.shape[0] > 0:
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i, :4]
                kps = kpss[i]
                embedding = self.face_recognizer(frame, kps)
                
                best_sim = -1
                best_name = "Unknown"
                for idx, known_emb in enumerate(self.known_face_embs):
                    sim = compute_similarity(embedding, known_emb)
                    if sim > best_sim:
                        best_sim = sim
                        if sim > self.similarity_thresh:
                            best_name = self.known_face_names[idx]
                
                label = f"{best_name} ({best_sim:.2f})"
                face_detections.append((bbox, label))

        # --- Gun Detection (with Super-Resolution logic) ---
        gun_results = self.gun_detector(frame, conf=gun_low_thresh, verbose=False)[0]
        gun_detections = []
        should_enhance = False
        for box in gun_results.boxes:
            conf = box.conf[0].item()
            xyxy = box.xyxy[0].cpu().numpy()
            
            if conf >= gun_high_thresh:
                label = f'{self.gun_detector.model.names[int(box.cls[0])]} {conf:.2f}'
                gun_detections.append((xyxy, label))
            elif gun_low_thresh <= conf < gun_high_thresh:
                should_enhance = True

        # --- Pose Estimation with MediaPipe ---
        frame.flags.writeable = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.mp_pose.process(frame_rgb)
        frame.flags.writeable = True
        all_keypoints = pose_results.pose_landmarks

        # --- Threat Association Logic (with persistence) ---
        final_face_detections = []
        # This logic uses the high-confidence gun_detections list
        for face_bbox, label in face_detections:
            person_name = label.split(' (')[0]

            if person_name != "Unknown":
                person_pose = find_pose_for_face(face_bbox, pose_results.pose_landmarks, frame.shape)
                if person_pose:
                    lwrist = person_pose.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
                    rwrist = person_pose.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]

                    for gun_bbox, _ in gun_detections:
                        if is_gun_near_hand(gun_bbox, lwrist, frame.shape) or \
                           is_gun_near_hand(gun_bbox, rwrist, frame.shape):
                            self.identified_threats.add(person_name)
                            break 
            
            is_shooter = person_name in self.identified_threats
            final_face_detections.append((face_bbox, label, is_shooter))

        return final_face_detections, gun_detections, all_keypoints, should_enhance

if __name__ == '__main__':
    # --- Configuration ---
    DET_WEIGHT = "./weights/det_10g.onnx"
    REC_WEIGHT = "./weights/w600k_r50.onnx"
    GUN_WEIGHT = "./best.pt"
    FACES_DIR = "./faces"
    SR_MODEL_WEIGHT = "./weights/realesr-general-x4v3.pth"
    GUN_LOW_CONF_THRESHOLD = 0.25
    GUN_HIGH_CONF_THRESHOLD = 0.75

    detector = CombinedDetector(
        det_weight=DET_WEIGHT,
        rec_weight=REC_WEIGHT,
        gun_weight=GUN_WEIGHT,
        faces_dir=FACES_DIR,
        sr_model_path=SR_MODEL_WEIGHT
    )

    # --- Worker function for AI processing ---
    def worker(frame_queue, result_queue, detector, gun_low_thresh, gun_high_thresh):
        enhance_next_frame = False
        sr_scale_factor = 4  # The model is 4x

        while True:
            try:
                frame = frame_queue.get()
                if frame is None:  # Sentinel check
                    break

                processed_frame = frame
                scale_factor = 1.0

                if enhance_next_frame:
                    print("-> Enhancing frame with Super-Resolution...")
                    processed_frame = detector.sr_model.upscale(processed_frame)
                    scale_factor = sr_scale_factor
                    enhance_next_frame = False  # Reset after use

                face_dets, gun_dets, keypoints, should_enhance = detector.detect_and_recognize(
                    processed_frame,
                    gun_low_thresh=gun_low_thresh,
                    gun_high_thresh=gun_high_thresh
                )

                if should_enhance:
                    enhance_next_frame = True

                # Scale results back down if we enhanced the frame
                if scale_factor > 1.0:
                    scaled_face_dets = []
                    for bbox, label, is_shooter in face_dets:
                        scaled_bbox = [c / scale_factor for c in bbox]
                        scaled_face_dets.append((scaled_bbox, label, is_shooter))
                    face_dets = scaled_face_dets

                    scaled_gun_dets = []
                    for bbox, label in gun_dets:
                        scaled_bbox = [c / scale_factor for c in bbox]
                        scaled_gun_dets.append((scaled_bbox, label))
                    gun_dets = scaled_gun_dets

                result_queue.put((face_dets, gun_dets, keypoints))
                frame_queue.task_done()
            except queue.Empty:
                continue

    # --- Webcam Processing ---
    cap = cv2.VideoCapture(0)
    print("\nStarting webcam feed. Press 'q' to exit.")

    # --- Optimization: Set a smaller processing resolution ---
    PROC_WIDTH = 640
    PROC_HEIGHT = 480
    
    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue()

    thread = threading.Thread(
        target=worker, 
        args=(frame_queue, result_queue, detector, GUN_LOW_CONF_THRESHOLD, GUN_HIGH_CONF_THRESHOLD), 
        daemon=True
    )
    thread.start()

    face_detections, gun_detections, all_keypoints = [], [], None

    while True:
        ret, frame = cap.read()
        if not ret: break

        orig_h, orig_w = frame.shape[:2]
        scale_w, scale_h = orig_w / PROC_WIDTH, orig_h / PROC_HEIGHT

        if frame_queue.empty():
            resized_frame = cv2.resize(frame, (PROC_WIDTH, PROC_HEIGHT))
            frame_queue.put(resized_frame)

        try:
            face_detections, gun_detections, all_keypoints = result_queue.get_nowait()
        except queue.Empty:
            pass

        # Draw pose keypoints (Yellow dots)
        if all_keypoints:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, all_keypoints, mp.solutions.pose.POSE_CONNECTIONS
            )

        # Draw face boxes (Green or Red)
        if face_detections:
            for bbox, label, is_shooter in face_detections:
                x1, y1, x2, y2 = [int(i) for i in bbox]
                box_color = (0, 0, 255) if is_shooter else (0, 255, 0)
                cv2.rectangle(frame, (int(x1*scale_w), int(y1*scale_h)), (int(x2*scale_w), int(y2*scale_h)), box_color, 2)
                cv2.putText(frame, label, (int(x1*scale_w), int(y1*scale_h) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # Draw gun boxes (Blue)
        if gun_detections:
            for xyxy, label in gun_detections:
                x1, y1, x2, y2 = [int(i) for i in xyxy]
                cv2.rectangle(frame, (int(x1*scale_w), int(y1*scale_h)), (int(x2*scale_w), int(y2*scale_h)), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(x1*scale_w), int(y1*scale_h) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow('Combined Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    frame_queue.put(None) # Send sentinel to stop the thread
    thread.join() # Wait for the AI thread to finish
    cap.release()
    cv2.destroyAllWindows()