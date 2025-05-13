# Face Re-Identification with SCRFD and ArcFace


<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest updates.</h5>
-->

<video controls autoplay loop src="https://github.com/yakhyo/face-reidentification/assets/28424328/441880b0-1e43-4c28-9f63-b32bc9b6e6b4" muted="false" width="100%"></video>

This repository implements face re-identification using SCRFD for face detection and ArcFace for face recognition. It supports inference from webcam or video sources.


- [x] Smaller versions of SCFRD face detection model has been added
- [x] **Face Detection**: Utilizes [Sample and Computation Redistribution for Efficient Face Detection](https://arxiv.org/abs/2105.04714) (SCRFD) for efficient and accurate face detection. (Updated on: 2024.07.29)
  - Added models: SCRFD 500M (2.41 MB), SCRFD 2.5G (3.14 MB)
- [x] **Face Recognition**: Employs [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698) for robust face recognition. (Updated on: 2024.07.29)
  - Added models: ArcFace MobileFace (12.99 MB)
- [x] **Real-Time Inference**: Supports both webcam and video file input for real-time processing.


Put target faces into `faces` folder

```
faces/
    ├── name1.jpg
    ├── name2.jpg
```

Those file names will be displayed while real-time inference.

## Usage

```bash
python main.py --source assets/in_video.mp4
```

`main.py` arguments:

```
usage: main.py [-h] [--det-weight DET_WEIGHT] [--rec-weight REC_WEIGHT] [--similarity-thresh SIMILARITY_THRESH] [--confidence-thresh CONFIDENCE_THRESH]
               [--faces-dir FACES_DIR] [--source SOURCE] [--max-num MAX_NUM] [--log-level LOG_LEVEL]

Face Detection-and-Recognition

options:
  -h, --help            show this help message and exit
  --det-weight DET_WEIGHT
                        Path to detection model
  --rec-weight REC_WEIGHT
                        Path to recognition model
  --similarity-thresh SIMILARITY_THRESH
                        Similarity threshold between faces
  --confidence-thresh CONFIDENCE_THRESH
                        Confidence threshold for face detection
  --faces-dir FACES_DIR
                        Path to faces stored dir
  --source SOURCE       Video file or video camera source. i.e 0 - webcam
  --max-num MAX_NUM     Maximum number of face detections from a frame
  --log-level LOG_LEVEL
                        Logging level
```

## Reference

1. https://github.com/deepinsight/insightface/tree/master/detection/scrfd
2. https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
