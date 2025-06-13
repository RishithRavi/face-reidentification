import os
import cv2
import random
import warnings
import argparse
import logging
import numpy as np
import time
import tkinter as tk
from tkinter import simpledialog

from models import SCRFD, ArcFace
from utils.helpers import compute_similarity, draw_bbox_info, draw_bbox

warnings.filterwarnings("ignore")

# Global variables for face labeling
face_labeling_mode = False
selected_face_bbox = None
selected_face_kps = None
current_frame = None
mouse_pos = (0, 0)


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

    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), None),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


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


def frame_processor(
    frame,
    detector,
    recognizer,
    targets,
    colors,
    params
):
    """
    Process a video frame for face detection and recognition.

    Args:
        frame (np.ndarray): The video frame.
        detector (SCRFD): Face detector model.
        recognizer (ArcFace): Face recognizer model.
        targets (List[Tuple[np.ndarray, str]]): List of target feature vectors and names.
        colors (dict): Dictionary of colors for drawing bounding boxes.
        params (argparse.Namespace): Command line arguments.

    Returns:
        Tuple[np.ndarray, int, np.ndarray, np.ndarray]: The processed video frame, number of faces detected, bboxes, and keypoints.
    """
    global face_labeling_mode, selected_face_bbox, selected_face_kps
    
    # Start timing
    start_time = time.time()
    
    bboxes, kpss = detector.detect(frame, params.max_num)
    num_faces = len(bboxes)
    
    for bbox, kps in zip(bboxes, kpss):
        *bbox_coords, conf_score = bbox.astype(np.int32)
        embedding = recognizer(frame, kps)

        max_similarity = 0
        best_match_name = "Unknown"
        for target, name in targets:
            similarity = compute_similarity(target, embedding)
            if similarity > max_similarity and similarity > params.similarity_thresh:
                max_similarity = similarity
                best_match_name = name

        # Determine color and draw bbox
        if face_labeling_mode:
            # Highlight faces differently in labeling mode
            if selected_face_bbox is not None and np.array_equal(bbox[:4], selected_face_bbox[:4]):
                # Selected face - bright yellow
                color = (0, 255, 255)
                draw_bbox_info(frame, bbox_coords, similarity=max_similarity, name=f"SELECTED: {best_match_name}", color=color)
            else:
                # Clickable faces - cyan
                color = (255, 255, 0)
                draw_bbox_info(frame, bbox_coords, similarity=max_similarity, name=f"CLICK: {best_match_name}", color=color)
        else:
            # Normal recognition mode
            if best_match_name != "Unknown":
                color = colors.get(best_match_name, (0, 255, 0))
                draw_bbox_info(frame, bbox_coords, similarity=max_similarity, name=best_match_name, color=color)
            else:
                draw_bbox(frame, bbox_coords, (255, 0, 0))
    
    # End timing
    process_time = time.time() - start_time
    
    return frame, num_faces, process_time, bboxes, kpss


def main():
    global face_labeling_mode, selected_face_bbox, selected_face_kps, current_frame
    
    params = parse_args()
    setup_logging(params.log_level)

    logging.info("Initializing models...")
    detector = SCRFD(params.det_weight, input_size=(640, 640), conf_thres=params.confidence_thresh)
    recognizer = ArcFace(params.rec_weight)
    logging.info("Models initialized successfully")

    targets = build_targets(detector, recognizer, params)
    colors = {name: (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)) for _, name in targets}

    logging.info(f"Opening webcam (device ID: {params.camera_id})...")
    cap = cv2.VideoCapture(params.camera_id)
    
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
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to grab frame from webcam")
                break

            current_frame = frame.copy()
            processed_frame, num_faces, process_time, bboxes, kpss = frame_processor(
                frame, detector, recognizer, targets, colors, params
            )
            
            # Update mouse callback parameters with current detection results
            cv2.setMouseCallback(window_name, mouse_callback, 
                               (detector, recognizer, bboxes, kpss, targets, colors, params))
            
            # Track performance metrics
            processing_times.append(process_time)
            face_counts.append(num_faces)
            
            # Update performance metrics
            frame_count += 1
            if frame_count % params.update_interval == 0:
                elapsed = time.time() - start_time
                fps_display = params.update_interval / elapsed
                
                # Reset timing
                start_time = time.time()
                
                # Calculate average processing time and face count
                if processing_times:
                    avg_process_time = sum(processing_times) / len(processing_times)
                    avg_faces = sum(face_counts) / len(face_counts)
                    logging.info(f"FPS: {fps_display:.1f}, Avg process time: {avg_process_time*1000:.1f}ms, Avg faces: {avg_faces:.1f}")
                    
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
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Done")


if __name__ == "__main__":
    main()