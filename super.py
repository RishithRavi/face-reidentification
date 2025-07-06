import os
import cv2
import random
import warnings
import argparse
import logging
import numpy as np
import time
import json
import tkinter as tk
from tkinter import simpledialog
from datetime import datetime
import pyttsx3
import threading
import queue
import torch
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose_solution
from mediapipe.python.solutions import drawing_utils as mp_drawing
from ultralytics import YOLO
from realesrgan import RealESRGANer

from models import SCRFD, ArcFace
from utils.helpers import compute_similarity, draw_bbox_info, draw_bbox
from sr_model import get_sr_model

warnings.filterwarnings("ignore")

# Global variables for face labeling
face_labeling_mode = False
selected_face_bbox = None
selected_face_kps = None
current_frame = None
mouse_pos = (0, 0)

# TTS and metadata globals
tts_engine = None
person_metadata = {}
threat_states = {}

# Super-resolution globals
sr_check_queue = queue.Queue()
sr_results = {}
frame_counter = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Face Recognition from Webcam")
    parser.add_argument(
        "--det-weight",
        type=str,
        default="./weights/det_10g.onnx",
        help="Path to detection model"
    )
    parser.add_argument(
        "--rec-weight",
        type=str,
        default="./weights/w600k_r50.onnx",
        help="Path to recognition model"
    )
    parser.add_argument(
        "--similarity-thresh",
        type=float,
        default=0.4,
        help="Similarity threshold between faces"
    )
    parser.add_argument(
        "--confidence-thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for face detection"
    )
    parser.add_argument(
        "--faces-dir",
        type=str,
        default="./faces",
        help="Path to faces stored dir"
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera device ID (usually 0 for the default webcam)"
    )
    parser.add_argument(
        "--max-num",
        type=int,
        default=0,
        help="Maximum number of face detections from a frame"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=10,
        help="Interval (in frames) to update performance metrics"
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default="./person_metadata.json",
        help="Path to person metadata JSON file"
    )
    parser.add_argument(
        "--announcement-cooldown",
        type=int,
        default=30,
        help="Cooldown period in seconds between announcements for the same person"
    )
    parser.add_argument(
        "--gun-det-weight",
        type=str,
        default="./best.pt",
        help="Path to gun detection model weight"
    )
    parser.add_argument(
        "--pose-model-complexity",
        type=int, default=1, choices=[0, 1, 2],
        help="Pose model complexity (0=lite, 1=full, 2=heavy)."
    )
    parser.add_argument(
        "--min-pose-detection-confidence",
        type=float, default=0.5,
        help="Minimum detection confidence for MediaPipe Pose."
    )
    parser.add_argument(
        "--min-pose-tracking-confidence",
        type=float, default=0.5,
        help="Minimum tracking confidence for MediaPipe Pose."
    )
    parser.add_argument(
        "--gun-hand-association-thresh",
        type=float, default=0.5,
        help="Minimum gun detection confidence to consider for hand association."
    )
    parser.add_argument(
        "--gun-det-confidence-thresh",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for gun detection."
    )
    parser.add_argument(
        "--gun-hand-proximity-pixels",
        type=int, default=100,
        help="Maximum distance in pixels between a detected gun and a hand/wrist to associate them."
    )
    parser.add_argument(
        "--draw-pose",
        action='store_true',
        help="Draw detected pose landmarks on the output frame."
    )
    parser.add_argument('--enable-sr', action='store_true', help='Enable super-resolution double-check.')
    parser.add_argument('--sr-model-path', type=str, default='./weights/realesr-general-x4v3.pth', help='Path to the Real-ESRGAN model.')
    parser.add_argument('--sr-conf-range', type=float, nargs=2, default=[0.5, 0.85], help='Confidence range to trigger SR check.')

    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), None),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def init_tts():
    """Initialize text-to-speech engine"""
    global tts_engine
    try:
        tts_engine = pyttsx3.init()
        # Set speech rate (optional)
        tts_engine.setProperty('rate', 150)
        # Set volume (optional)
        tts_engine.setProperty('volume', 0.8)
        logging.info("Text-to-speech engine initialized")
    except Exception as e:
        logging.error(f"Failed to initialize TTS engine: {e}")
        tts_engine = None


def speak_async(text):
    """Speak text asynchronously to avoid blocking the main thread"""
    def speak():
        if tts_engine:
            try:
                tts_engine.say(text)
                tts_engine.runAndWait()
            except Exception as e:
                logging.error(f"TTS error: {e}")
    
    if tts_engine:
        thread = threading.Thread(target=speak, daemon=True)
        thread.start()



def get_landmark_coords(landmarks, landmark_id, frame_shape):
    """Convert normalized MediaPipe landmark to pixel coordinates."""
    if landmarks and landmark_id < len(landmarks.landmark):
        landmark = landmarks.landmark[landmark_id]
        if landmark.visibility < 0.5: # Or some other threshold
            return None
        x = int(landmark.x * frame_shape[1])
        y = int(landmark.y * frame_shape[0])
        return (x, y)
    return None

def is_gun_near_hand(gun_bbox, hand_coord_px, proximity_threshold_px):
    """Check if the center of a gun bounding box is near a hand coordinate."""
    if hand_coord_px is None or gun_bbox is None:
        return False
    gun_center_x = (gun_bbox[0] + gun_bbox[2]) / 2
    gun_center_y = (gun_bbox[1] + gun_bbox[3]) / 2
    distance = np.sqrt((gun_center_x - hand_coord_px[0])**2 + (gun_center_y - hand_coord_px[1])**2)
    return distance < proximity_threshold_px

def find_pose_for_face(face_bbox_coords, pose_results, frame_shape, mp_pose_module):
    """Finds pose landmarks corresponding to a given face bounding box.
       Assumes nose landmark (ID 0) should be within the face_bbox.
    """
    if not pose_results or not pose_results.pose_landmarks:
        return None

    face_x1, face_y1, face_x2, face_y2 = face_bbox_coords
    
    # Iterate through all detected poses (though typically it's one per person if static_image_mode=False)
    # For multi-pose detection, you might need a more sophisticated matching logic if pose_results contains multiple poses.
    # Here, we assume pose_results.pose_landmarks is the primary detected pose or the most relevant one.
    # If MediaPipe returns multiple pose instances in pose_results.multi_pose_landmarks, you'd iterate through that.
    
    # Using pose_results.pose_landmarks (single primary pose)
    nose_coord = get_landmark_coords(pose_results.pose_landmarks, mp_pose_module.PoseLandmark.NOSE, frame_shape)
    if nose_coord:
        if face_x1 <= nose_coord[0] <= face_x2 and face_y1 <= nose_coord[1] <= face_y2:
            return pose_results.pose_landmarks # Return the full set of landmarks for this pose
    return None

def load_person_metadata(metadata_file):
    """Load person metadata from JSON file"""
    global person_metadata
    
    # Create default metadata if file doesn't exist
    if not os.path.exists(metadata_file):
        create_default_metadata(metadata_file)
    
    try:
        with open(metadata_file, 'r') as f:
            person_metadata = json.load(f)
        logging.info(f"Loaded metadata for {len(person_metadata)} people")
    except Exception as e:
        logging.error(f"Failed to load metadata file {metadata_file}: {e}")
        person_metadata = {}


def create_default_metadata(metadata_file):
    """Create default metadata file with sample data"""
    default_data = {
        "Richie": {
            "full_name": "Richard Martinez",
            "role": "Student",
            "schedule": {
                "Monday": [
                    {"time": "09:00-10:00", "class": "Mathematics", "room": "Room 101"},
                    {"time": "10:15-11:15", "class": "Physics", "room": "Lab 203"},
                    {"time": "11:30-12:30", "class": "Chemistry", "room": "Lab 105"},
                    {"time": "13:30-14:30", "class": "English Literature", "room": "Room 302"},
                    {"time": "14:45-15:45", "class": "Computer Science", "room": "Computer Lab"}
                ],
                "Tuesday": [
                    {"time": "09:00-10:00", "class": "History", "room": "Room 201"},
                    {"time": "10:15-11:15", "class": "Biology", "room": "Lab 301"},
                    {"time": "11:30-12:30", "class": "Art", "room": "Art Studio"},
                    {"time": "13:30-14:30", "class": "Physical Education", "room": "Gymnasium"},
                    {"time": "14:45-15:45", "class": "Study Hall", "room": "Library"}
                ],
                "Wednesday": [
                    {"time": "09:00-10:00", "class": "Mathematics", "room": "Room 101"},
                    {"time": "10:15-11:15", "class": "Physics", "room": "Lab 203"},
                    {"time": "11:30-12:30", "class": "Chemistry", "room": "Lab 105"},
                    {"time": "13:30-14:30", "class": "Spanish", "room": "Room 205"},
                    {"time": "14:45-15:45", "class": "Music", "room": "Music Room"}
                ],
                "Thursday": [
                    {"time": "09:00-10:00", "class": "History", "room": "Room 201"},
                    {"time": "10:15-11:15", "class": "Biology", "room": "Lab 301"},
                    {"time": "11:30-12:30", "class": "English Literature", "room": "Room 302"},
                    {"time": "13:30-14:30", "class": "Computer Science", "room": "Computer Lab"},
                    {"time": "14:45-15:45", "class": "Study Hall", "room": "Library"}
                ],
                "Friday": [
                    {"time": "09:00-10:00", "class": "Mathematics", "room": "Room 101"},
                    {"time": "10:15-11:15", "class": "Physics", "room": "Lab 203"},
                    {"time": "11:30-12:30", "class": "Art", "room": "Art Studio"},
                    {"time": "13:30-14:30", "class": "Physical Education", "room": "Gymnasium"},
                    {"time": "14:45-15:45", "class": "Free Period", "room": "Cafeteria"}
                ]
            }
        },
        "John": {
            "full_name": "John Smith",
            "role": "Teacher",
            "schedule": {
                "Monday": [
                    {"time": "08:00-09:00", "class": "Prep Time", "room": "Office 12"},
                    {"time": "09:00-10:00", "class": "Teaching Physics", "room": "Lab 203"},
                    {"time": "10:15-11:15", "class": "Teaching Chemistry", "room": "Lab 105"},
                    {"time": "11:30-12:30", "class": "Lunch Break", "room": "Faculty Lounge"},
                    {"time": "13:30-14:30", "class": "Department Meeting", "room": "Conference Room"},
                    {"time": "14:45-15:45", "class": "Office Hours", "room": "Office 12"}
                ],
                "Tuesday": [
                    {"time": "08:00-09:00", "class": "Prep Time", "room": "Office 12"},
                    {"time": "09:00-10:00", "class": "Teaching Physics", "room": "Lab 203"},
                    {"time": "10:15-11:15", "class": "Teaching Chemistry", "room": "Lab 105"},
                    {"time": "11:30-12:30", "class": "Lunch Break", "room": "Faculty Lounge"},
                    {"time": "13:30-14:30", "class": "Grading", "room": "Office 12"},
                    {"time": "14:45-15:45", "class": "Office Hours", "room": "Office 12"}
                ],
                "Wednesday": [
                    {"time": "08:00-09:00", "class": "Prep Time", "room": "Office 12"},
                    {"time": "09:00-10:00", "class": "Teaching Physics", "room": "Lab 203"},
                    {"time": "10:15-11:15", "class": "Teaching Chemistry", "room": "Lab 105"},
                    {"time": "11:30-12:30", "class": "Lunch Break", "room": "Faculty Lounge"},
                    {"time": "13:30-14:30", "class": "Lab Maintenance", "room": "Lab 203"},
                    {"time": "14:45-15:45", "class": "Office Hours", "room": "Office 12"}
                ],
                "Thursday": [
                    {"time": "08:00-09:00", "class": "Prep Time", "room": "Office 12"},
                    {"time": "09:00-10:00", "class": "Teaching Physics", "room": "Lab 203"},
                    {"time": "10:15-11:15", "class": "Teaching Chemistry", "room": "Lab 105"},
                    {"time": "11:30-12:30", "class": "Lunch Break", "room": "Faculty Lounge"},
                    {"time": "13:30-14:30", "class": "Parent Conferences", "room": "Office 12"},
                    {"time": "14:45-15:45", "class": "Office Hours", "room": "Office 12"}
                ],
                "Friday": [
                    {"time": "08:00-09:00", "class": "Prep Time", "room": "Office 12"},
                    {"time": "09:00-10:00", "class": "Teaching Physics", "room": "Lab 203"},
                    {"time": "10:15-11:15", "class": "Teaching Chemistry", "room": "Lab 105"},
                    {"time": "11:30-12:30", "class": "Lunch Break", "room": "Faculty Lounge"},
                    {"time": "13:30-14:30", "class": "Lesson Planning", "room": "Office 12"},
                    {"time": "14:45-15:45", "class": "Weekend Prep", "room": "Office 12"}
                ]
            }
        }
    }
    
    try:
        with open(metadata_file, 'w') as f:
            json.dump(default_data, f, indent=4)
        logging.info(f"Created default metadata file: {metadata_file}")
    except Exception as e:
        logging.error(f"Failed to create default metadata file: {e}")


def get_current_class(person_name):
    """
    Get current class for a person based on current time
    """
    if person_name not in person_metadata:
        logging.warning(f"Metadata not found for {person_name}")
        return None

    person_data = person_metadata[person_name]
    if "schedule" not in person_data:
        logging.warning(f"Schedule not found for {person_name}")
        return None

    schedule_data = person_data["schedule"]
    now = datetime.now()
    current_time = now.time()

    daily_schedule_entries = []
    if isinstance(schedule_data, dict):
        current_day = now.strftime("%A")
        if current_day not in schedule_data:
            logging.info(f"No schedule for {person_name} on {current_day}")
            return None
        daily_schedule_entries = schedule_data[current_day]
    elif isinstance(schedule_data, list):
        daily_schedule_entries = schedule_data
    else:
        logging.error(f"Unknown schedule format for {person_name}: {type(schedule_data)}. Expected dict or list.")
        return None

    for class_info in daily_schedule_entries:
        time_range = class_info.get("time")
        if not time_range:
            continue

        try:
            start_time_str, end_time_str = time_range.split("-")
            start_time = datetime.strptime(start_time_str, "%H:%M").time()
            end_time = datetime.strptime(end_time_str, "%H:%M").time()

            if start_time <= current_time < end_time:
                return class_info
        except ValueError as e:
            logging.warning(f"Could not parse time range '{time_range}' for {person_name}: {e}")
            continue

    return None


def check_bbox_intersection(person_bbox, gun_bbox, overlap_thresh=0.1):
    """
    Check if a person's bounding box intersects with a gun's bounding box.

    Args:
        person_bbox (list): Bounding box of the person [x1, y1, x2, y2].
        gun_bbox (list): Bounding box of the gun [x1, y1, x2, y2].
        overlap_thresh (float): The threshold for intersection over union (IoU).

    Returns:
        bool: True if the bounding boxes intersect significantly, False otherwise.
    """
    px1, py1, px2, py2 = person_bbox
    gx1, gy1, gx2, gy2 = gun_bbox

    # Calculate intersection area
    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    # Calculate person's bounding box area
    person_area = (px2 - px1) * (py2 - py1)
    if person_area == 0:
        return False

    # Check if intersection is significant
    overlap = inter_area / person_area
    return overlap > overlap_thresh


def announce_person(person_name):
    """Announce person's current class via TTS"""
    global announced_persons
    
    # Check if we've recently announced this person
    current_time = time.time()
    person_key = f"{person_name}_{int(current_time // 30)}"  # 30-second buckets
    
    if person_key in announced_persons:
        return
    
    # Clean old announcements (older than 2 minutes)
    announced_persons = {key for key in announced_persons 
                        if int(current_time // 30) - int(key.split('_')[1]) < 4}
    
    announced_persons.add(person_key)
    
    current_class = get_current_class(person_name)
    
    if current_class:
        announcement = f"Hello {person_name}! You should be in {current_class['class']} right now, in {current_class['room']}"
    else:
        # Check if person exists in metadata
        if person_name in person_metadata:
            role = person_metadata[person_name].get("role", "person")
            announcement = f"Hello {person_name}! No scheduled class at this time. You are registered as a {role}."
        else:
            announcement = f"Hello {person_name}! Welcome!"
    
    logging.info(f"Announcing: {announcement}")
    speak_async(announcement)


def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for face selection"""
    global mouse_pos, selected_face_bbox, selected_face_kps, face_labeling_mode
    
    mouse_pos = (x, y)
    
    if event == cv2.EVENT_LBUTTONDOWN and face_labeling_mode:
        detector, recognizer, bboxes, kpss, targets, colors, params = param
        
        # Check if click is within any face bounding box
        for bbox, kps in zip(bboxes, kpss):
            x1, y1, x2, y2 = bbox[:4].astype(np.int32)
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_face_bbox = bbox
                selected_face_kps = kps
                logging.info(f"Face selected at ({x}, {y})")
                break


def get_face_name():
    """Get face name from user using tkinter dialog"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    name = simpledialog.askstring("Face Labeling", "Enter person's name:")
    root.destroy()
    return name


def save_face_crop(frame, bbox, kps, name, faces_dir):
    """Save cropped face image to faces directory"""
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)
    
    x1, y1, x2, y2 = bbox[:4].astype(np.int32)
    
    # Add some padding around the face
    padding = 20
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(frame.shape[1], x2 + padding)
    y2 = min(frame.shape[0], y2 + padding)
    
    face_crop = frame[y1:y2, x1:x2]
    
    # Generate filename
    timestamp = int(time.time())
    filename = f"{name}_{timestamp}.jpg"
    filepath = os.path.join(faces_dir, filename)
    
    # Save the cropped face
    success = cv2.imwrite(filepath, face_crop)
    if success:
        logging.info(f"Saved face crop: {filepath}")
        return filepath
    else:
        logging.error(f"Failed to save face crop: {filepath}")
        return None


def build_targets(detector, recognizer, params: argparse.Namespace):
    """
    Build targets using face detection and recognition.

    Args:
        detector (SCRFD): Face detector model.
        recognizer (ArcFace): Face recognizer model.
        params (argparse.Namespace): Command line arguments.

    Returns:
        List[Tuple[np.ndarray, str]]: A list of tuples containing feature vectors and corresponding image names.
    """
    targets = []
    logging.info(f"Loading face targets from {params.faces_dir}")
    
    if not os.path.exists(params.faces_dir):
        os.makedirs(params.faces_dir)
        logging.info(f"Created faces directory: {params.faces_dir}")
    
    for filename in os.listdir(params.faces_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        name = os.path.splitext(filename)[0].split('_')[0]  # Remove timestamp if present
        image_path = os.path.join(params.faces_dir, filename)

        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Could not read image {image_path}. Skipping...")
            continue
            
        bboxes, kpss = detector.detect(image, max_num=1)

        if len(kpss) == 0:
            logging.warning(f"No face detected in {image_path}. Skipping...")
            continue

        embedding = recognizer(image, kpss[0])
        targets.append((embedding, name))
        logging.info(f"Added target: {name}")

    logging.info(f"Loaded {len(targets)} face targets")
    return targets


def sr_worker(q, results_dict, gun_detector, sr_model, high_thresh):
    while True:
        frame_id, frame_crop, original_bbox = q.get()
        enhanced_crop, _ = sr_model.enhance(frame_crop)
        new_results = gun_detector(enhanced_crop, verbose=False)
        new_conf = 0
        if new_results and new_results[0].boxes:
            new_conf = new_results[0].boxes.data[0][4].item()
        status = 'confirmed' if new_conf >= high_thresh else 'discarded'
        if frame_id not in results_dict:
            results_dict[frame_id] = []
        results_dict[frame_id].append((original_bbox, new_conf, status))
        q.task_done()


def frame_processor(
    frame,
    detector,
    recognizer,
    gun_detector,
    pose_model,
    targets,
    colors,
    params,
    frame_id,
    sr_check_queue,
    sr_results
):
    """
    Process a video frame for face detection, recognition, and pose-based gun association.

    Args:
        frame (np.ndarray): The video frame.
        detector (SCRFD): Face detector model.
        recognizer (ArcFace): Face recognizer model.
        gun_detector (YOLO): Gun detection model.
        pose_model (mediapipe.solutions.pose.Pose): MediaPipe Pose model instance.
        targets (List[Tuple[np.ndarray, str]]): List of target feature vectors and names.
        colors (dict): Dictionary of colors for drawing bounding boxes.
        params (argparse.Namespace): Command line arguments.

    Returns:
        Tuple[np.ndarray, int, float, np.ndarray, np.ndarray]: The processed video frame, number of faces, proc time, bboxes, kpss.
    """
    global face_labeling_mode, selected_face_bbox, selected_face_kps, threat_states
    
    start_time_proc = time.time()

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False # Performance optimization
    pose_results = pose_model.process(rgb_frame)
    rgb_frame.flags.writeable = True # Revert for OpenCV drawing
    # Frame is already BGR, so convert back if needed for other ops, or draw on original 'frame'

    if params.draw_pose and pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose_solution.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
    
    bboxes_det, kpss_det = detector.detect(frame, params.max_num)
    num_faces = len(bboxes_det)
    
    gun_results_frame = None
    if gun_detector:
        gun_results_frame = gun_detector(frame, verbose=False)

    for bbox_data, kps_data in zip(bboxes_det, kpss_det):
        face_bbox_coords = bbox_data[:4].astype(int)
        embedding = recognizer(frame, kps_data)

        max_similarity = 0.0
        best_match_name = "Unknown"
        for target_embedding, known_name in targets:
            similarity = compute_similarity(target_embedding, embedding)
            if similarity > max_similarity and similarity > params.similarity_thresh:
                max_similarity = similarity
                best_match_name = known_name

        if best_match_name != "Unknown" and best_match_name not in threat_states:
            threat_states[best_match_name] = {
                'is_threat': False,
                'last_gun_seen_time': 0,
                'weapon_in_view': False,
                'sitrep_announced': False
            }

        # New: Pose-based gun association
        person_specific_has_gun = False
        person_pose_landmarks = find_pose_for_face(face_bbox_coords, pose_results, frame.shape, mp_pose_solution)

        if person_pose_landmarks and gun_detector and gun_results_frame and gun_results_frame[0].boxes:
            left_wrist_coord = get_landmark_coords(person_pose_landmarks, mp_pose_solution.PoseLandmark.LEFT_WRIST, frame.shape)
            right_wrist_coord = get_landmark_coords(person_pose_landmarks, mp_pose_solution.PoseLandmark.RIGHT_WRIST, frame.shape)
            
            # Optional: Draw wrist circles for debugging
            # if left_wrist_coord: cv2.circle(frame, left_wrist_coord, 10, (0,255,255), -1)
            # if right_wrist_coord: cv2.circle(frame, right_wrist_coord, 10, (255,0,255), -1)

            for gun_box_data in gun_results_frame[0].boxes.data:
                gun_confidence = gun_box_data[4].item()
                # Draw all detected guns above general threshold, but associate only if above association_thresh
                if gun_confidence >= params.gun_det_confidence_thresh: # General detection threshold from params
                    detected_gun_bbox = gun_box_data[:4].cpu().numpy().astype(int)
                    low_thresh, high_thresh = params.sr_conf_range

                    if params.enable_sr and low_thresh <= gun_confidence < high_thresh:
                        x1, y1, x2, y2 = detected_gun_bbox
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            sr_check_queue.put((frame_id, crop.copy(), detected_gun_bbox))
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(frame, f"Verifying... ({gun_confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    elif gun_confidence >= high_thresh:
                        cv2.rectangle(frame, (detected_gun_bbox[0], detected_gun_bbox[1]), (detected_gun_bbox[2], detected_gun_bbox[3]), (0, 0, 255), 2)
                        cv2.putText(frame, f"Firearm ({gun_confidence:.2f})", (detected_gun_bbox[0], detected_gun_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        # Check for association only if gun confidence is high enough for association
                        if gun_confidence >= params.gun_hand_association_thresh:
                            if (is_gun_near_hand(detected_gun_bbox, left_wrist_coord, params.gun_hand_proximity_pixels) or 
                                is_gun_near_hand(detected_gun_bbox, right_wrist_coord, params.gun_hand_proximity_pixels)):
                                person_specific_has_gun = True
                                # Optionally, highlight the associated gun or hand
                                cv2.putText(frame, f"ARMED: {best_match_name}", (detected_gun_bbox[0], detected_gun_bbox[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                                break # Associated this person with a gun
        
        if best_match_name != "Unknown":
            current_time = time.time()
            state = threat_states[best_match_name]

            if person_specific_has_gun:
                # --- WEAPON IS VISIBLE ---
                if state['is_threat'] and not state['weapon_in_view']:
                    # This is the TRANSITION from NOT IN VIEW -> IN VIEW
                    speak_async("Threat's weapon is back in view.")
                
                # Update state for weapon being visible
                state['is_threat'] = True
                state['weapon_in_view'] = True
                state['last_gun_seen_time'] = current_time

                # Announce the initial threat details if not already done
                if not state['sitrep_announced']:
                    class_info = get_current_class(best_match_name)
                    announcement = f"Threat identified as {best_match_name}."
                    if class_info:
                        announcement += f" Current class would be {class_info['class']} in {class_info['room']}."
                    speak_async(announcement)
                    state['sitrep_announced'] = True
            
            else: # person_has_gun is False
                # --- WEAPON IS NOT VISIBLE ---
                # Check if the person is a known threat and the weapon was previously in view
                if state['is_threat'] and state['weapon_in_view']:
                    # This is the TRANSITION from IN VIEW -> NOT IN VIEW
                    # Check if enough time has passed since it was last seen
                    if current_time - state['last_gun_seen_time'] > 5.0:
                        speak_async("Threat's weapon not in view.")
                        # Update the state to reflect the weapon is no longer in view
                        state['weapon_in_view'] = False
        
        if face_labeling_mode:
            if selected_face_bbox is not None and np.array_equal(bbox_data[:4], selected_face_bbox[:4]):
                color = (0, 255, 255)
                draw_bbox_info(frame, face_bbox_coords, similarity=max_similarity, name=f"SELECTED: {best_match_name}", color=color)
            else:
                color = (255, 255, 0)
                draw_bbox_info(frame, face_bbox_coords, similarity=max_similarity, name=f"CLICK: {best_match_name}", color=color)
        else: # Recognition mode
            display_name_on_box = best_match_name
            box_color = colors.get(best_match_name, (0, 255, 0))

            if best_match_name != "Unknown":
                if threat_states[best_match_name]['is_threat']:
                    box_color = (0, 0, 255)
                
                current_class_info = get_current_class(best_match_name)
                if current_class_info:
                    display_name_on_box += f" | {current_class_info['class']}"
                draw_bbox_info(frame, face_bbox_coords, similarity=max_similarity, name=display_name_on_box, color=box_color)
            else: # Unknown person
                box_color = (0, 0, 255) if person_specific_has_gun else colors.get(best_match_name, (0, 255, 0))
                draw_bbox_info(frame, face_bbox_coords, similarity=0, name="Unknown", color=box_color)

    if frame_id in sr_results:
        for bbox, new_conf, status in sr_results[frame_id]:
            if status == 'confirmed':
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"Confirmed ({new_conf:.2f})", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if frame_id > 100:
            del sr_results[frame_id - 100]

    process_time_val = time.time() - start_time_proc
    return frame, num_faces, process_time_val, bboxes_det, kpss_det


def main():
    global face_labeling_mode, selected_face_bbox, selected_face_kps, current_frame, frame_counter
    
    params = parse_args()
    setup_logging(params.log_level)

    # Initialize TTS and load metadata
    init_tts()
    load_person_metadata(params.metadata_file)

    logging.info("Initializing models...")
    detector = SCRFD(params.det_weight, input_size=(640, 640), conf_thres=params.confidence_thresh)
    recognizer = ArcFace(params.rec_weight)

    logging.info("Initializing MediaPipe Pose model...")
    # mp_pose_solution = mp.solutions.pose # Defined globally now for landmark access
    pose_model = mp_pose_solution.Pose(
        static_image_mode=False, # Process video stream
        model_complexity=params.pose_model_complexity,
        min_detection_confidence=params.min_pose_detection_confidence,
        min_tracking_confidence=params.min_pose_tracking_confidence
    )
    # Initialize Gun Detector (YOLO)
    logging.info("Initializing Gun Detector (YOLO)...")
    try:
        gun_detector = YOLO(params.gun_det_weight)  # Load the YOLO model for gun detection
        logging.info("Gun Detector initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing Gun Detector: {e}")
        gun_detector = None

    sr_model = None
    if params.enable_sr:
        logging.info("Initializing Super-Resolution model...")
        try:
            model = get_sr_model()
            sr_model = RealESRGANer(
                scale=4, 
                model_path=params.sr_model_path, 
                model=model, 
                tile=0, 
                tile_pad=10, 
                pre_pad=0, 
                half=True if torch.cuda.is_available() else False
            )
            threading.Thread(
                target=sr_worker, 
                args=(sr_check_queue, sr_results, gun_detector, sr_model, params.sr_conf_range[1]), 
                daemon=True
            ).start()
            logging.info("Super-Resolution model initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing Super-Resolution model: {e}")
            sr_model = None

    logging.info("Models initialized successfully")

    targets = build_targets(detector, recognizer, params)
    colors = {name: (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)) for _, name in targets}

    logging.info(f"Opening webcam (device ID: {params.camera_id}) using DSHOW backend...")
    cap = cv2.VideoCapture(params.camera_id, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        logging.error(f"Could not open webcam with ID {params.camera_id}")
        raise Exception(f"Could not open webcam with ID {params.camera_id}")

    # Try to set a higher resolution for the webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logging.info(f"Webcam opened: {width}x{height} at {fps} FPS")
    logging.info("Press 'q' to quit, 'p' to enter face labeling mode")

    window_name = "Real-time Face Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Set mouse callback
    cv2.setMouseCallback(window_name, mouse_callback, 
                        (detector, recognizer, [], [], targets, colors, params))
    
    # Performance tracking variables
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    processing_times = []
    face_counts = []
    
    try:
        while True:
            frame_counter += 1
            # print("DEBUG: Attempting cap.read()", flush=True) # Commented out for now
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.error("Failed to grab frame from webcam after cap.read()")
                # print("DEBUG: cap.read() failed or returned empty frame.", flush=True) # Commented out for now
                break

            current_frame = frame.copy()
            processed_frame, num_faces, process_time, bboxes, kpss = frame_processor(
                frame, detector, recognizer, gun_detector, pose_model, targets, colors, params, frame_counter, sr_check_queue, sr_results
            )
            
            # Update mouse callback parameters with current detection results
            # Note: bboxes and kpss for mouse_callback should come from frame_processor
            cv2.setMouseCallback(window_name, mouse_callback, 
                               (detector, recognizer, bboxes, kpss, targets, colors, params))
            
            # Track performance metrics
            processing_times.append(process_time)
            face_counts.append(num_faces)
            
            # Update performance metrics
            frame_count += 1
            if frame_count % params.update_interval == 0:
                elapsed = time.time() - start_time
                fps_display = params.update_interval / elapsed if elapsed > 0 else 0 # Avoid division by zero
                
                # Reset timing
                start_time = time.time()
                
                # Calculate average processing time and face count
                if processing_times:
                    avg_process_time = sum(processing_times) / len(processing_times)
                    avg_faces = sum(face_counts) / len(face_counts)
                    
                    # Reset lists to avoid memory growth
                    processing_times = []
                    face_counts = []
            
            # Add performance overlay
            cv2.putText(
                processed_frame,
                f"FPS: {fps_display:.1f} | Faces: {num_faces} | Process: {process_time*1000:.1f}ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            
            # Add mode indicator and instructions
            mode_text = "LABELING MODE - Click on faces to label" if face_labeling_mode else "RECOGNITION MODE"
            mode_color = (0, 255, 255) if face_labeling_mode else (0, 255, 0)
            cv2.putText(
                processed_frame,
                mode_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                mode_color,
                2,
                cv2.LINE_AA
            )
            
            # Add instructions
            instructions = [
                "Press 'q' to quit, 'p' to toggle labeling mode",
                "Press 'Enter' to confirm face label" if selected_face_bbox is not None else ""
            ]
            
            for i, instruction in enumerate(instructions):
                if instruction:
                    cv2.putText(
                        processed_frame, 
                        instruction, 
                        (10, height - 40 + i * 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255), 
                        1, 
                        cv2.LINE_AA
                    )
            
            cv2.imshow(window_name, processed_frame)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q"):
                break
            elif key == ord("p") or key == ord("P"):
                face_labeling_mode = not face_labeling_mode
                selected_face_bbox = None
                selected_face_kps = None
                logging.info(f"Face labeling mode: {'ON' if face_labeling_mode else 'OFF'}")
            elif key == 13 and selected_face_bbox is not None:  # Enter key
                # Get name from user
                name = get_face_name()
                if name and name.strip():
                    name = name.strip()
                    # Save the face crop
                    saved_path = save_face_crop(current_frame, selected_face_bbox, 
                                              selected_face_kps, name, params.faces_dir)
                    
                    if saved_path:
                        # Add to current recognition targets
                        embedding = recognizer(current_frame, selected_face_kps)
                        targets.append((embedding, name))
                        colors[name] = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
                        logging.info(f"Added new face target: {name}")
                    
                    # Reset selection
                    selected_face_bbox = None
                    selected_face_kps = None
                    face_labeling_mode = False
                    logging.info("Face labeling completed")
                else:
                    logging.info("Face labeling cancelled - no name provided")
                
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        logging.info("Releasing resources...")
        if 'pose_model' in locals() and pose_model is not None:
            pose_model.close()
            logging.info("MediaPipe Pose model released.")
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Done")


if __name__ == "__main__":
    main()