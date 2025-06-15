import os
import cv2
import time
import random
import warnings
import argparse
import logging
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

from models import SCRFD, ArcFace
from utils.helpers import compute_similarity, draw_bbox_info, draw_bbox

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Scalable Real-time Face Recognition from Webcam")
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
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for face recognition"
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


class FaceDatabase:
    """Optimized face database for efficient similarity search with large numbers of faces."""
    
    def __init__(self, similarity_threshold: float = 0.4):
        self.embeddings = []  # List of face embeddings
        self.names = []       # Corresponding names
        self.similarity_threshold = similarity_threshold
        self.embedding_dim = None
        
    def add_face(self, embedding: np.ndarray, name: str):
        """Add a face embedding and name to the database."""
        if self.embedding_dim is None:
            self.embedding_dim = embedding.shape[0]
        elif embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[0]}")
        
        self.embeddings.append(embedding)
        self.names.append(name)
    
    def finalize(self):
        """Convert lists to numpy arrays for faster processing."""
        if not self.embeddings:
            return
        
        self.embeddings = np.array(self.embeddings, dtype=np.float32)
        # Normalize embeddings for faster cosine similarity computation
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / norms
    
    def batch_search(self, query_embeddings: np.ndarray) -> List[Tuple[str, float]]:
        """
        Find the best matches for a batch of query embeddings.
        
        Args:
            query_embeddings: Batch of query embeddings, shape (n_queries, embedding_dim)
            
        Returns:
            List of (name, similarity) tuples for each query
        """
        if len(self.embeddings) == 0:
            return [("Unknown", 0.0)] * len(query_embeddings)
        
        # Normalize query embeddings
        query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        normalized_queries = query_embeddings / query_norms
        
        # Compute similarities for all queries against all database embeddings
        # This is a matrix multiplication that gives us a matrix of shape (n_queries, n_database)
        similarities = np.dot(normalized_queries, self.embeddings.T)
        
        # Find the best match for each query
        best_match_indices = np.argmax(similarities, axis=1)
        best_match_similarities = np.max(similarities, axis=1)
        
        # Create result list
        results = []
        for idx, sim in zip(best_match_indices, best_match_similarities):
            if sim > self.similarity_threshold:
                results.append((self.names[idx], sim))
            else:
                results.append(("Unknown", sim))
                
        return results
    
    def search(self, query_embedding: np.ndarray) -> Tuple[str, float]:
        """
        Find the best match for a single query embedding.
        
        Args:
            query_embedding: Query embedding vector
            
        Returns:
            Tuple of (name, similarity)
        """
        results = self.batch_search(query_embedding.reshape(1, -1))
        return results[0]
    
    def __len__(self):
        return len(self.names)


def build_face_database(detector, recognizer, params: argparse.Namespace) -> FaceDatabase:
    """
    Build an optimized face database from images in the faces directory.
    
    Args:
        detector: Face detector model
        recognizer: Face recognizer model
        params: Command line arguments
        
    Returns:
        FaceDatabase object containing face embeddings and names
    """
    database = FaceDatabase(similarity_threshold=params.similarity_thresh)
    face_dir = params.faces_dir
    
    logging.info(f"Building face database from {face_dir}")
    start_time = time.time()
    
    # Get list of image files
    image_files = [f for f in os.listdir(face_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        logging.warning(f"No image files found in {face_dir}")
        return database
    
    logging.info(f"Found {len(image_files)} potential face images")
    
    # Process each image
    for filename in image_files:
        name = os.path.splitext(filename)[0]
        image_path = os.path.join(face_dir, filename)
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                logging.warning(f"Could not read image {image_path}. Skipping...")
                continue
                
            # Detect face
            bboxes, kpss = detector.detect(image, max_num=1)
            
            if len(kpss) == 0:
                logging.warning(f"No face detected in {image_path}. Skipping...")
                continue
            
            # Get face embedding
            embedding = recognizer(image, kpss[0])
            
            # Add to database
            database.add_face(embedding, name)
            logging.info(f"Added face: {name}")
            
        except Exception as e:
            logging.error(f"Error processing {image_path}: {str(e)}")
    
    # Finalize database for efficient search
    database.finalize()
    
    elapsed = time.time() - start_time
    logging.info(f"Face database built with {len(database)} faces in {elapsed:.2f} seconds")
    
    return database


def process_frame(
    frame: np.ndarray,
    detector: SCRFD,
    recognizer: ArcFace,
    face_db: FaceDatabase,
    colors: Dict[str, Tuple[int, int, int]],
    params: argparse.Namespace
) -> Tuple[np.ndarray, int, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Process a video frame for face detection and recognition.
    
    Args:
        frame: The video frame
        detector: Face detector model
        recognizer: Face recognizer model
        face_db: Face database for matching
        colors: Dictionary of colors for drawing bounding boxes
        params: Command line arguments
        
    Returns:
        Tuple of (processed_frame, num_faces_detected, list of (bbox, kps) for detected faces)
    """
    # Detect faces
    bboxes, kpss = detector.detect(frame, params.max_num)
    
    if len(bboxes) == 0:
        return frame, 0, []
    
    # Extract face embeddings for all detected faces
    embeddings = []
    for kps in kpss:
        embedding = recognizer(frame, kps)
        embeddings.append(embedding)
    
    # Convert to numpy array for batch processing
    if embeddings:
        embeddings_array = np.array(embeddings)
        
        # Find matches for all embeddings at once
        matches = face_db.batch_search(embeddings_array)
        
        # Draw results
        for (bbox, match_info) in zip(bboxes, matches):
            name, similarity = match_info
            *bbox_coords, conf_score = bbox.astype(np.int32)
            
            if name != "Unknown":
                color = colors.get(name, (0, 255, 0))  # Default to green if color not found
                draw_bbox_info(frame, bbox_coords, similarity=similarity, name=name, color=color)
            else:
                draw_bbox(frame, bbox_coords, (255, 0, 0))  # Red for unknown faces
    
    return frame, len(bboxes), list(zip(bboxes, kpss))


def main():
    params = parse_args()
    setup_logging(params.log_level)
    
    # Initialize models
    logging.info("Initializing models...")
    detector = SCRFD(params.det_weight, input_size=(640, 640), conf_thres=params.confidence_thresh)
    recognizer = ArcFace(params.rec_weight)
    logging.info("Models initialized successfully")
    
    # Build face database
    face_db = build_face_database(detector, recognizer, params)
    if len(face_db) == 0:
        logging.warning("No faces loaded into database. Recognition will not work.")
    
    # Generate colors for visualization
    unique_names = set(face_db.names)
    colors = {name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
              for name in unique_names}
    
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
    
    window_name = f"Scalable Face Recognition ({len(face_db)} faces in database)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to grab frame from webcam")
                break
            
            # Process frame and measure time
            process_start = time.time()
            processed_frame, num_faces, _ = process_frame(frame, detector, recognizer, face_db, colors, params)
            process_end = time.time()
            
            process_time = process_end - process_start
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
                    
                    # Reset lists to avoid memory growth
                    processing_times = []
                    face_counts = []
            
            # Add performance overlay
            cv2.putText(
                processed_frame,
                f"FPS: {fps_display:.1f} | Faces: {num_faces} | DB Size: {len(face_db)}",
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
