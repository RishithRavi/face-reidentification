import os
import cv2
import random
import warnings
import argparse
import logging
import numpy as np
import time

from models import SCRFD, ArcFace
from utils.helpers import compute_similarity, draw_bbox_info, draw_bbox

warnings.filterwarnings("ignore")


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
    
    for filename in os.listdir(params.faces_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        name = os.path.splitext(filename)[0]
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
        Tuple[np.ndarray, int]: The processed video frame and number of faces detected.
    """
    # Start timing
    start_time = time.time()
    
    bboxes, kpss = detector.detect(frame, params.max_num)
    num_faces = len(bboxes)
    
    for bbox, kps in zip(bboxes, kpss):
        *bbox, conf_score = bbox.astype(np.int32)
        embedding = recognizer(frame, kps)

        max_similarity = 0
        best_match_name = "Unknown"
        for target, name in targets:
            similarity = compute_similarity(target, embedding)
            if similarity > max_similarity and similarity > params.similarity_thresh:
                max_similarity = similarity
                best_match_name = name

        if best_match_name != "Unknown":
            color = colors[best_match_name]
            draw_bbox_info(frame, bbox, similarity=max_similarity, name=best_match_name, color=color)
        else:
            draw_bbox(frame, bbox, (255, 0, 0))
    
    # End timing
    process_time = time.time() - start_time
    
    return frame, num_faces, process_time


def main():
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
    logging.info("Press 'q' to quit")

    window_name = "Real-time Face Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
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

            processed_frame, num_faces, process_time = frame_processor(frame, detector, recognizer, targets, colors, params)
            
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
            
            # Add a simple UI with instructions
            cv2.putText(
                processed_frame, 
                "Press 'q' to quit", 
                (10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1, 
                cv2.LINE_AA
            )
            
            cv2.imshow(window_name, processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        logging.info("Releasing resources...")
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Done")


if __name__ == "__main__":
    main()
