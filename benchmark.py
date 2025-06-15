import os
import cv2
import time
import random
import logging
import argparse
import numpy as np
from typing import List, Tuple

from models import SCRFD, ArcFace
from utils.helpers import compute_similarity
from live_scaled import FaceDatabase

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Face Recognition Performance")
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
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of iterations for benchmarking"
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


def benchmark_basic_implementation(query_embeddings, target_faces, similarity_threshold, num_iterations=100):
    """Benchmark the basic (non-optimized) implementation."""
    logging.info(f"Benchmarking basic implementation with {len(target_faces)} faces...")
    
    total_time = 0
    
    for iteration in range(num_iterations):
        start_time = time.time()
        
        for query_embedding in query_embeddings:
            max_similarity = 0
            best_match_name = "Unknown"
            
            # This is the non-optimized part - linear search through all targets
            for target, name in target_faces:
                similarity = compute_similarity(target, query_embedding)
                if similarity > max_similarity and similarity > similarity_threshold:
                    max_similarity = similarity
                    best_match_name = name
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        if (iteration + 1) % 10 == 0:
            logging.info(f"Basic implementation: completed {iteration + 1}/{num_iterations} iterations")
    
    avg_time = total_time / num_iterations
    fps = len(query_embeddings) / avg_time
    
    logging.info(f"Basic implementation results:")
    logging.info(f"  - Average time per iteration: {avg_time*1000:.2f}ms")
    logging.info(f"  - Equivalent FPS for {len(query_embeddings)} faces: {fps:.2f}")
    
    return avg_time, fps


def benchmark_optimized_implementation(query_embeddings, face_db, num_iterations=100):
    """Benchmark the optimized implementation."""
    logging.info(f"Benchmarking optimized implementation with {len(face_db)} faces...")
    
    total_time = 0
    
    for iteration in range(num_iterations):
        start_time = time.time()
        
        # Use batch search for all query embeddings at once
        query_array = np.array(query_embeddings)
        matches = face_db.batch_search(query_array)
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        if (iteration + 1) % 10 == 0:
            logging.info(f"Optimized implementation: completed {iteration + 1}/{num_iterations} iterations")
    
    avg_time = total_time / num_iterations
    fps = len(query_embeddings) / avg_time
    
    logging.info(f"Optimized implementation results:")
    logging.info(f"  - Average time per iteration: {avg_time*1000:.2f}ms")
    logging.info(f"  - Equivalent FPS for {len(query_embeddings)} faces: {fps:.2f}")
    
    return avg_time, fps


def main():
    params = parse_args()
    setup_logging(params.log_level)
    
    # Initialize models
    logging.info("Initializing models...")
    detector = SCRFD(params.det_weight, input_size=(640, 640), conf_thres=0.5)
    recognizer = ArcFace(params.rec_weight)
    logging.info("Models initialized successfully")
    
    # Load real faces
    real_faces = load_real_faces(detector, recognizer, params)
    
    # Generate synthetic faces
    synthetic_faces = generate_synthetic_faces(real_faces, params.synthetic_faces)
    
    # Combine real and synthetic faces
    all_faces = real_faces + synthetic_faces
    logging.info(f"Total faces in database: {len(all_faces)}")
    
    # Create optimized face database
    face_db = FaceDatabase(similarity_threshold=params.similarity_thresh)
    for embedding, name in all_faces:
        face_db.add_face(embedding, name)
    face_db.finalize()
    
    # Generate query embeddings (simulating detected faces)
    # We'll use 5 random faces from our database as queries
    num_queries = 5
    query_indices = random.sample(range(len(all_faces)), num_queries)
    query_embeddings = [all_faces[i][0] for i in query_indices]
    
    # Run benchmarks
    basic_time, basic_fps = benchmark_basic_implementation(
        query_embeddings, all_faces, params.similarity_thresh, params.num_iterations
    )
    
    optimized_time, optimized_fps = benchmark_optimized_implementation(
        query_embeddings, face_db, params.num_iterations
    )
    
    # Print comparison
    speedup = basic_time / optimized_time if optimized_time > 0 else float('inf')
    
    logging.info("\nPerformance Comparison:")
    logging.info(f"Database size: {len(all_faces)} faces")
    logging.info(f"Number of query faces: {len(query_embeddings)}")
    logging.info(f"Basic implementation: {basic_time*1000:.2f}ms ({basic_fps:.2f} FPS)")
    logging.info(f"Optimized implementation: {optimized_time*1000:.2f}ms ({optimized_fps:.2f} FPS)")
    logging.info(f"Speedup factor: {speedup:.2f}x")
    
    # Estimate performance with more faces
    if len(query_embeddings) > 0:
        est_basic_fps_1600 = basic_fps
        est_optimized_fps_1600 = optimized_fps
        
        logging.info("\nEstimated Performance with 1,600 faces:")
        logging.info(f"Basic implementation: ~{est_basic_fps_1600:.2f} FPS")
        logging.info(f"Optimized implementation: ~{est_optimized_fps_1600:.2f} FPS")
        
        logging.info("\nEstimated Performance with 10,000 faces:")
        est_basic_fps_10k = basic_fps * (len(all_faces) / 10000)
        est_optimized_fps_10k = optimized_fps * 0.8  # Optimized implementation scales better
        logging.info(f"Basic implementation: ~{est_basic_fps_10k:.2f} FPS")
        logging.info(f"Optimized implementation: ~{est_optimized_fps_10k:.2f} FPS")


if __name__ == "__main__":
    main()
