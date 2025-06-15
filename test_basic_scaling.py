import os
import cv2
import time
import random
import logging
import argparse
import numpy as np
from typing import List, Tuple

from models import SCRFD, ArcFace
from utils.helpers import compute_similarity, draw_bbox_info, draw_bbox

def parse_args():
    parser = argparse.ArgumentParser(description="Test Basic Face Recognition Scaling")
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
        "--synthetic-faces",
        type=int,
        default=1600,
        help="Number of synthetic faces to generate"
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
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), None),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_real_faces(detector, recognizer, params) -> List[Tuple[np.ndarray, str]]:
    """Load real face embeddings from the faces directory."""
    real_faces = []
    face_dir = params.faces_dir
    
    logging.info(f"Loading real faces from {face_dir}")
    
    for filename in os.listdir(face_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        name = os.path.splitext(filename)[0]
        image_path = os.path.join(face_dir, filename)

        try:
            image = cv2.imread(image_path)
            if image is None:
                logging.warning(f"Could not read image {image_path}. Skipping...")
                continue
                
            bboxes, kpss = detector.detect(image, max_num=1)
            
            if len(kpss) == 0:
                logging.warning(f"No face detected in {image_path}. Skipping...")
                continue
            
            embedding = recognizer(image, kpss[0])
            real_faces.append((embedding, name))
            logging.info(f"Added real face: {name}")
            
        except Exception as e:
            logging.error(f"Error processing {image_path}: {str(e)}")
    
    logging.info(f"Loaded {len(real_faces)} real faces")
    return real_faces


def generate_synthetic_faces(real_faces, num_synthetic=1600) -> List[Tuple[np.ndarray, str]]:
    """Generate synthetic face embeddings based on real faces."""
    if not real_faces:
        logging.error("No real faces available to generate synthetic faces")
        return []
    
    logging.info(f"Generating {num_synthetic} synthetic face embeddings...")
    start_time = time.time()
    
    synthetic_faces = []
    embedding_dim = real_faces[0][0].shape[0]
    
    # Calculate mean and std of real face embeddings
    real_embeddings = np.array([face[0] for face in real_faces])
    mean_embedding = np.mean(real_embeddings, axis=0)
    std_embedding = np.std(real_embeddings, axis=0)
    
    # Generate synthetic embeddings with similar statistical properties
    for i in range(num_synthetic):
        # Generate a random embedding with similar properties to real faces
        noise = np.random.normal(0, 1, embedding_dim)
        synthetic_embedding = mean_embedding + noise * std_embedding * 0.1
        
        # Normalize the embedding (important for cosine similarity)
        norm = np.linalg.norm(synthetic_embedding)
        synthetic_embedding = synthetic_embedding / norm
        
        # Generate a synthetic name
        synthetic_name = f"synthetic_person_{i+1}"
        
        synthetic_faces.append((synthetic_embedding, synthetic_name))
        
        if (i+1) % 100 == 0:
            logging.info(f"Generated {i+1} synthetic faces...")
    
    elapsed = time.time() - start_time
    logging.info(f"Generated {len(synthetic_faces)} synthetic faces in {elapsed:.2f} seconds")
    
    return synthetic_faces


def frame_processor(
    frame,
    detector,
    recognizer,
    targets,
    colors,
    params
):
    """
    Process a video frame for face detection and recognition using the basic (non-optimized) approach.
    This is similar to the original implementation in live.py.
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
        
        # This is the non-optimized part - linear search through all targets
        for target, name in targets:
            similarity = compute_similarity(target, embedding)
            if similarity > max_similarity and similarity > params.similarity_thresh:
                max_similarity = similarity
                best_match_name = name

        if best_match_name != "Unknown":
            color = colors.get(best_match_name, (0, 255, 0))  # Default to green if color not found
            draw_bbox_info(frame, bbox, similarity=max_similarity, name=best_match_name, color=color)
        else:
            draw_bbox(frame, bbox, (255, 0, 0))
    
    # End timing
    process_time = time.time() - start_time
    
    return frame, num_faces, process_time


def main():
    params = parse_args()
    setup_logging(params.log_level)
    
    # Initialize models
    logging.info("Initializing models...")
    detector = SCRFD(params.det_weight, input_size=(640, 640), conf_thres=params.confidence_thresh)
    recognizer = ArcFace(params.rec_weight)
    logging.info("Models initialized successfully")
    
    # Load real faces
    real_faces = load_real_faces(detector, recognizer, params)
    
    # Generate synthetic faces
    synthetic_faces = generate_synthetic_faces(real_faces, params.synthetic_faces)
    
    # Combine real and synthetic faces
    all_faces = real_faces + synthetic_faces
    logging.info(f"Total faces in database: {len(all_faces)}")
    
    # Generate colors for visualization
    unique_names = set([name for _, name in real_faces])  # Only use real face names for colors
    colors = {name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
              for name in unique_names}
    
    # Add some synthetic colors too (for testing)
    for i in range(20):  # Add colors for first 20 synthetic faces
        synthetic_name = f"synthetic_person_{i+1}"
        colors[synthetic_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    # Open webcam
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
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    processing_times = []
    face_counts = []
    
    window_name = f"Basic Implementation: {len(all_faces)} faces in database"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to grab frame from webcam")
                break
            
            # Process frame and measure time
            processed_frame, num_faces, process_time = frame_processor(frame, detector, recognizer, all_faces, colors, params)
            
            processing_times.append(process_time)
            face_counts.append(num_faces)
            
            # Update performance metrics
            frame_count += 1
            if frame_count % 10 == 0:  # Update every 10 frames
                elapsed = time.time() - start_time
                fps_display = 10 / elapsed
                
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
                f"FPS: {fps_display:.1f} | Faces: {num_faces} | DB Size: {len(all_faces)} | Process: {process_time*1000:.1f}ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            
            # Add instructions
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
            
            # Display the frame
            cv2.imshow(window_name, processed_frame)
            
            # Check for key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
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
