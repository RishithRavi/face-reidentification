import cv2
import os
import glob
import numpy as np
from ultralytics import YOLO
import threading
import queue
import mediapipe as mp

from models import SCRFD, ArcFace
from utils.helpers import compute_similarity

# --- OpenCV DNN Super-Resolution Setup ---
from cv2 import dnn_superres
import os
sr = dnn_superres.DnnSuperResImpl_create()
# Ensure you download EDSR_x4.pb from the OpenCV contrib repo and place it in the project root:
# https://github.com/opencv/opencv_contrib/blob/master/modules/dnn_superres/samples/EDSR_x4.pb
MODEL_PATH = "EDSR_x4.pb"
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Super-res model not found at {MODEL_PATH}. Download it from https://github.com/opencv/opencv_contrib/blob/master/modules/dnn_superres/samples/EDSR_x4.pb")
sr.readModel(MODEL_PATH)
sr.setModel("edsr", 4)("edsr", 4)          # model name and scale factor

# --- Helper Functions based on super.py logic ---
def find_pose_for_face(face_bbox, pose_landmarks, frame_shape):
    if not pose_landmarks:
        return None
    face_x1, face_y1, face_x2, face_y2 = face_bbox
    nose = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
    nose_px = int(nose.x * frame_shape[1]), int(nose.y * frame_shape[0])
    if face_x1 <= nose_px[0] <= face_x2 and face_y1 <= nose_px[1] <= face_y2:
        return pose_landmarks
    return None


def is_gun_near_hand(gun_bbox, hand_landmark, frame_shape, proximity_thresh=100):
    if not hand_landmark or hand_landmark.visibility < 0.5:
        return False
    gun_center_x = (gun_bbox[0] + gun_bbox[2]) / 2
    gun_center_y = (gun_bbox[1] + gun_bbox[3]) / 2
    hand_px = int(hand_landmark.x * frame_shape[1]), int(hand_landmark.y * frame_shape[0])
    distance = np.sqrt((gun_center_x - hand_px[0])**2 + (gun_center_y - hand_px[1])**2)
    return distance < proximity_thresh

class CombinedDetector:
    def __init__(self, det_weight, rec_weight, gun_weight, faces_dir, similarity_thresh=0.4):
        self.face_detector = SCRFD(det_weight)
        self.face_recognizer = ArcFace(rec_weight)
        self.gun_detector = YOLO(gun_weight)
        self.mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.similarity_thresh = similarity_thresh
        self.known_face_embs = []
        self.known_face_names = []
        self.identified_threats = set()
        self._load_known_faces(faces_dir)

    def _load_known_faces(self, faces_dir):
        if not os.path.isdir(faces_dir):
            print(f"Error: Faces directory not found at {faces_dir}")
            return
        print("Loading known faces...")
        for image_path in glob.glob(os.path.join(faces_dir, "*")):
            img = cv2.imread(image_path)
            if img is None: continue
            bboxes, kpss = self.face_detector.detect(img)
            if bboxes.shape[0] > 0:
                emb = self.face_recognizer(img, kpss[0])
                name = os.path.splitext(os.path.basename(image_path))[0]
                self.known_face_embs.append(emb)
                self.known_face_names.append(name)
                print(f"- Loaded face for {name}")
        print(f"Finished loading {len(self.known_face_names)} known faces.")

    def detect_and_recognize(self, frame, gun_conf_thresh=0.5):
        # Face Recognition
        face_detections = []
        bboxes, kpss = self.face_detector.detect(frame)
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, :4]
            emb = self.face_recognizer(frame, kpss[i])
            best_sim, best_name = -1, "Unknown"
            for idx, known in enumerate(self.known_face_embs):
                sim = compute_similarity(emb, known)
                if sim > best_sim:
                    best_sim = sim
                    if sim > self.similarity_thresh:
                        best_name = self.known_face_names[idx]
            face_detections.append((bbox, f"{best_name} ({best_sim:.2f})"))

        # Gun Detection
        gun_results = self.gun_detector(frame, conf=gun_conf_thresh, verbose=False)[0]
        gun_detections = []
        for box in gun_results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            name = self.gun_detector.model.names[cls_id]
            conf = float(box.conf[0])
            gun_detections.append((xyxy, name, conf))

        # Pose
        frame.flags.writeable = False
        pose = self.mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks
        frame.flags.writeable = True

        # Associate threats
        final_faces = []
        for bbox, label in face_detections:
            name = label.split(' (')[0]
            if name != "Unknown":
                pose_found = find_pose_for_face(bbox, pose, frame.shape)
                if pose_found:
                    lw = pose_found.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
                    rw = pose_found.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
                    for gbb, _, _ in gun_detections:
                        if is_gun_near_hand(gbb, lw, frame.shape) or is_gun_near_hand(gbb, rw, frame.shape):
                            self.identified_threats.add(name)
                            break
            final_faces.append((bbox, label, name in self.identified_threats))

        return final_faces, gun_detections, pose

if __name__ == '__main__':
    DET_WEIGHT, REC_WEIGHT, GUN_WEIGHT = "./weights/det_10g.onnx", "./weights/w600k_r50.onnx", "./best.pt"
    FACES_DIR = "./faces"
    LOW, HIGH = 0.5, 0.75

    detector = CombinedDetector(DET_WEIGHT, REC_WEIGHT, GUN_WEIGHT, FACES_DIR)

    def worker(in_q, out_q, det):
        while True:
            frame = in_q.get()
            if frame is None: break
            fds, gds, p = det.detect_and_recognize(frame, gun_conf_thresh=LOW)
            out_q.put((fds, gds, p))
            in_q.task_done()

    cap = cv2.VideoCapture(0)
    PROC_W, PROC_H = 640, 480
    fq, rq = queue.Queue(1), queue.Queue()
    threading.Thread(target=worker, args=(fq, rq, detector), daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret: break
        if fq.empty(): fq.put(cv2.resize(frame, (PROC_W, PROC_H)))
        try:
            faces, guns, pose = rq.get_nowait()
        except queue.Empty:
            faces, guns, pose = [], [], None

        if pose:
            mp.solutions.drawing_utils.draw_landmarks(frame, pose, mp.solutions.pose.POSE_CONNECTIONS)

        # Gun boxes + SR
        for xyxy, name, conf in guns:
            x1,y1,x2,y2 = map(int, xyxy)
            lbl, clr = f"{name} {conf:.2f}", (255,0,0)
            if name.lower() == 'weapon' and LOW <= conf < HIGH:
                roi = frame[y1:y2, x1:x2]
                sr_roi = sr.upsample(roi)
                frame[y1:y2, x1:x2] = cv2.resize(sr_roi, (x2-x1, y2-y1))
                sr_res = detector.gun_detector(sr_roi, conf=LOW, verbose=False)[0]
                if sr_res.boxes:
                    new_c = float(sr_res.boxes.conf.max().item())
                    lbl = f"SR {new_c:.2f}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), clr, 2)
            cv2.putText(frame, lbl, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2)

        # Face boxes
        for bbox, lbl, threat in faces:
            x1,y1,x2,y2 = (int(i) for i in bbox)
            c = (0,0,255) if threat else (0,255,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), c, 2)
            cv2.putText(frame, lbl, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

        cv2.imshow('Combined + SR', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fq.put(None)
    cap.release()
    cv2.destroyAllWindows()
