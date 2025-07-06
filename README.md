# Real-Time Shooter Identification System

<h5 align="center">An advanced, real-time threat detection system that identifies individuals holding firearms by intelligently combining face recognition, gun detection, and pose estimation.</h5>

<!-- <video controls autoplay loop src="https://github.com/yakhyo/face-reidentification/assets/28424328/441880b0-1e43-4c28-9f63-b32bc9b6e6b4" muted="false" width="100%"></video> -->

This project provides a robust solution for identifying potential threats in real-time video streams. It uses a multi-layered AI approach to detect faces, firearms, and human poses, then applies a sophisticated logic to determine if a recognized person is holding a gun. Once a threat is identified, the system persistently marks the individual for the remainder of the session.

## Core Features

- **Multi-Modal AI Detection**: Simultaneously runs four different AI models for:
  - **Face Detection**: High-accuracy SCRFD for locating faces.
  - **Face Recognition**: ArcFace for identifying known individuals.
  - **Gun Detection**: YOLOv8 for spotting firearms.
  - **Pose Estimation**: MediaPipe Pose for tracking body keypoints.
- **Advanced Threat Association**: Implements a reliable **Face → Pose → Hand → Gun** logic. Instead of just guessing, the system confirms that a detected gun is in close proximity to the hands of a specific, identified person.
- **Persistent Threat Memory**: Once a person is identified as a shooter, their status is remembered. Their bounding box remains red for the rest of the session, ensuring continuous awareness.
- **Real-Time Performance**: Utilizes multithreading to run AI inference in a background process, ensuring a smooth, responsive UI on the main video feed.

## Technology Stack

- **OpenCV**: For video capture and rendering.
- **SCRFD & ArcFace (ONNX)**: For state-of-the-art face detection and recognition.
- **YOLOv8 (Ultralytics)**: For real-time object (gun) detection.
- **MediaPipe**: For accurate and fast human pose estimation.
- **NumPy**: For efficient numerical operations.

## Setup and Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/RishithRavi/face-reidentification.git
    cd face-reidentification
    ```

2.  **Install Dependencies**
    A `requirements.txt` file is provided for easy setup. 
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Model Weights**
    You will need the necessary model weights for the system to function. You can use the provided script to download them.
    ```bash
    ./download.sh
    ```

4.  **Add Known Faces**
    Place images of people you want the system to recognize into the `faces` directory. The filename (without extension) will be used as the person's name.
    ```
    faces/
        ├── Richie.jpg
        ├── John.png
    ```

## Usage

To run the main application, execute the `base.py` script. The system will start using your primary webcam (source 0) by default.

```bash
python base.py
```

While the application is running:
- The system will draw boxes around detected faces, guns, and pose keypoints.
- A **green** box indicates a recognized, non-threatening person.
- A **red** box indicates a person who has been identified as a shooter.
- Press **'q'** to quit the application at any time.

## How It Works

The system operates in a continuous loop, performing the following steps on each frame:

1.  **Capture & Preprocess**: A frame is captured from the webcam and resized for faster processing.
2.  **Parallel Inference**: The frame is sent to a background thread that runs all AI models:
    -   `SCRFD` detects all faces.
    -   `ArcFace` generates embeddings for each detected face and compares them to the known faces.
    -   `YOLOv8` detects any guns in the frame.
    -   `MediaPipe Pose` detects the keypoints for any person in the frame.
3.  **Threat Association**: The main thread uses the results to apply the core logic:
    a. For each recognized **face**, it finds the corresponding **pose** by matching the nose keypoint to the face's bounding box.
    b. It then gets the coordinates of that specific pose's **hands** (wrists).
    c. Finally, it checks if any detected **gun** is within a close proximity threshold to those hands.
4.  **Update & Render**: If a person is confirmed to be holding a gun, their name is added to a persistent `identified_threats` set. The system then draws the appropriate colored bounding boxes and labels on the original, full-resolution frame for display.

## Reference

1. https://github.com/deepinsight/insightface/tree/master/detection/scrfd
2. https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
