import cv2
import os
import glob
import numpy as np
from ultralytics import YOLO
import threading
import queue
import mediapipe as mp
import json
import asyncio
import websockets
from datetime import datetime
import base64
import networkx as nx
import pyttsx3
from typing import Dict, List, Tuple, Optional

from models import SCRFD, ArcFace
from utils.helpers import compute_similarity

# --- Pathfinding Module ---
class PathfindingService:
    def __init__(self, gexf_file, floor_plan_image):
        """Initialize pathfinding service with building graph"""
        self.G = nx.read_gexf(gexf_file)
        self.floor_plan = cv2.imread(floor_plan_image)
        if self.floor_plan is None:
            raise FileNotFoundError(f"Floor plan image not found at {floor_plan_image}")
        
        # Extract node positions and floor information
        self.pos = {}
        self.floor_info = {}
        for node, data in self.G.nodes(data=True):
            self.pos[node] = (float(data['pos_x']), float(data['pos_y']))
            if 'floor' in data:
                self.floor_info[node] = data['floor']
        
        # Pre-defined important locations
        self.location_nodes = {
            "Cafeteria": "CAF1",
            "Gym": "GYM1",
            "Library": "LIB1",
            "Main Entrance": "ENT1",
            "Safe Room 1": "SAFE1",
            "Safe Room 2": "SAFE2",
            "Office": "OFF1",
            "Hallway A": "HALL1",
            "Hallway B": "HALL2",
            "Stairwell North": "STAIR1",
            "Stairwell South": "STAIR2"
        }
        
        # Officer positions (can be updated dynamically)
        self.officer_positions = {
            "Officer1": "ENT1",
            "Officer2": "HALL1",
            "Officer3": "STAIR1"
        }
        
        # Initialize TTS engine (optional, can be disabled)
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
            self.tts_enabled = True
        except:
            self.tts_enabled = False
            print("TTS not available, voice navigation disabled")
    
    def get_nearest_node(self, x: int, y: int, max_dist: float = 50) -> Optional[str]:
        """Find nearest graph node to given coordinates"""
        closest_node = None
        min_dist = float('inf')
        
        for node, (nx_pos, ny_pos) in self.pos.items():
            dist = np.sqrt((x - nx_pos)**2 + (y - ny_pos)**2)
            if dist < min_dist and dist < max_dist:
                min_dist = dist
                closest_node = node
        
        return closest_node
    
    def find_path(self, start_node: str, end_node: str) -> Dict:
        """Find shortest path between two nodes"""
        try:
            if start_node == end_node:
                return {
                    "status": "same_node",
                    "path": [start_node],
                    "distance": 0,
                    "instructions": ["Already at destination"]
                }
            
            # Find shortest path
            path_nodes = nx.shortest_path(self.G, source=start_node, target=end_node, weight='weight')
            path_length = nx.shortest_path_length(self.G, source=start_node, target=end_node, weight='weight')
            
            # Generate instructions
            instructions = []
            path_coordinates = []
            
            current_floor = self.floor_info.get(path_nodes[0], "Ground")
            instructions.append(f"Start at {path_nodes[0]} on {current_floor}")
            
            for i in range(len(path_nodes)):
                node = path_nodes[i]
                path_coordinates.append(self.pos[node])
                
                if i > 0:
                    new_floor = self.floor_info.get(node, "Ground")
                    
                    # Check if this is a stair connection
                    edge_data = self.G.get_edge_data(path_nodes[i-1], node)
                    if edge_data and 'is_stair' in edge_data and edge_data['is_stair'] == 'true':
                        instructions.append(f"Take stairs from {current_floor} to {new_floor}")
                        current_floor = new_floor
                    else:
                        instructions.append(f"Move to {node}")
            
            instructions.append(f"Arrived at {end_node}")
            
            return {
                "status": "success",
                "path": path_nodes,
                "coordinates": path_coordinates,
                "distance": path_length,
                "instructions": instructions,
                "floors_traversed": list(set([self.floor_info.get(n, "Ground") for n in path_nodes]))
            }
            
        except nx.NetworkXNoPath:
            return {
                "status": "no_path",
                "error": f"No path exists between {start_node} and {end_node}"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def find_intercept_path(self, threat_location: str, officer_id: str) -> Dict:
        """Calculate optimal intercept path for an officer to reach threat location"""
        if officer_id not in self.officer_positions:
            return {"status": "error", "error": f"Unknown officer: {officer_id}"}
        
        officer_node = self.officer_positions[officer_id]
        
        # If threat_location is a known location name, convert to node
        if threat_location in self.location_nodes:
            threat_node = self.location_nodes[threat_location]
        else:
            threat_node = threat_location
        
        path_result = self.find_path(officer_node, threat_node)
        path_result["officer"] = officer_id
        path_result["threat_location"] = threat_location
        
        return path_result
    
    def find_evacuation_routes(self, current_location: str) -> List[Dict]:
        """Find all evacuation routes from current location to safe zones"""
        evacuation_routes = []
        
        # Define safe zones
        safe_zones = ["SAFE1", "SAFE2", "ENT1"]  # Safe rooms and main entrance
        
        for safe_zone in safe_zones:
            path_result = self.find_path(current_location, safe_zone)
            if path_result["status"] == "success":
                evacuation_routes.append({
                    "destination": safe_zone,
                    "distance": path_result["distance"],
                    "path": path_result["path"],
                    "instructions": path_result["instructions"]
                })
        
        # Sort by distance (shortest first)
        evacuation_routes.sort(key=lambda x: x["distance"])
        
        return evacuation_routes
    
    def update_officer_position(self, officer_id: str, new_position: str):
        """Update an officer's current position"""
        self.officer_positions[officer_id] = new_position
        return {"status": "updated", "officer": officer_id, "position": new_position}
    
    def render_path_on_floor_plan(self, path_nodes: List[str]) -> str:
        """Render a path on the floor plan and return as base64 image"""
        img = self.floor_plan.copy()
        
        # Draw the path
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            if self.G.has_edge(u, v):
                edge_data = self.G.get_edge_data(u, v)
                
                # Different color for stairs
                color = (255, 0, 255) if 'is_stair' in edge_data and edge_data['is_stair'] == 'true' else (0, 0, 255)
                
                if 'path' in edge_data:
                    try:
                        path_points = json.loads(edge_data['path'])
                        pts = np.array([(int(x), int(y)) for x, y in path_points], dtype=np.int32)
                        cv2.polylines(img, [pts], False, color, 3)
                    except:
                        # Fallback to straight line
                        pt1 = (int(self.pos[u][0]), int(self.pos[u][1]))
                        pt2 = (int(self.pos[v][0]), int(self.pos[v][1]))
                        cv2.line(img, pt1, pt2, color, 3)
        
        # Highlight start and end nodes
        if len(path_nodes) > 0:
            start_pos = (int(self.pos[path_nodes[0]][0]), int(self.pos[path_nodes[0]][1]))
            end_pos = (int(self.pos[path_nodes[-1]][0]), int(self.pos[path_nodes[-1]][1]))
            cv2.circle(img, start_pos, 10, (0, 255, 0), -1)  # Green for start
            cv2.circle(img, end_pos, 10, (0, 0, 255), -1)    # Red for end
        
        # Encode as base64
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return img_base64

# --- Detection Helper Functions ---
def find_pose_for_face(face_bbox, pose_landmarks, frame_shape):
    """Finds the pose that corresponds to a given face bounding box."""
    if not pose_landmarks:
        return None
    
    face_x1, face_y1, face_x2, face_y2 = face_bbox
    nose = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
    
    nose_px = int(nose.x * frame_shape[1]), int(nose.y * frame_shape[0])

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

# --- Combined Detector with Pathfinding ---
class CombinedDetectorWithPathfinding:
    def __init__(self, det_weight, rec_weight, gun_weight, faces_dir, 
                 gexf_file, floor_plan_image, similarity_thresh=0.4):
        # Initialize detection models
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
        
        # Initialize pathfinding service
        self.pathfinder = PathfindingService(gexf_file, floor_plan_image)
        
        # WebSocket clients
        self.websocket_clients = set()
        
        # Current threat locations for pathfinding
        self.threat_locations = {}

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

    def detect_and_recognize(self, frame, gun_conf_thresh=0.7):
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

        # --- Gun Detection ---
        gun_results = self.gun_detector(frame, conf=gun_conf_thresh, verbose=False)[0]
        gun_detections = []
        for box in gun_results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            label = f'{self.gun_detector.model.names[int(box.cls[0])]} {box.conf[0]:.2f}'
            gun_detections.append((xyxy, label))

        # --- Pose Estimation ---
        frame.flags.writeable = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.mp_pose.process(frame_rgb)
        frame.flags.writeable = True
        all_keypoints = pose_results.pose_landmarks

        # --- Threat Association Logic ---
        final_face_detections = []
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

        return final_face_detections, gun_detections, all_keypoints

    async def register_client(self, websocket):
        """Register a new WebSocket client."""
        self.websocket_clients.add(websocket)
        print(f"New client connected. Total clients: {len(self.websocket_clients)}")

    async def unregister_client(self, websocket):
        """Unregister a WebSocket client."""
        self.websocket_clients.discard(websocket)
        print(f"Client disconnected. Total clients: {len(self.websocket_clients)}")

    async def broadcast_data(self, data):
        """Broadcast data to all connected WebSocket clients."""
        if self.websocket_clients:
            message = json.dumps(data)
            disconnected_clients = set()
            for client in self.websocket_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
            
            for client in disconnected_clients:
                await self.unregister_client(client)
    
    async def handle_pathfinding_request(self, request_data):
        """Handle pathfinding requests from the dashboard"""
        request_type = request_data.get("type") or request_data.get("request_type")
        
        if request_type == "intercept":
            # Calculate intercept path for officer to threat
            officer_id = request_data.get("officer_id")
            threat_location = request_data.get("threat_location")
            
            result = self.pathfinder.find_intercept_path(threat_location, officer_id)
            
            # Add rendered floor plan if path found
            if result["status"] == "success":
                result["floor_plan"] = self.pathfinder.render_path_on_floor_plan(result["path"])
            
            await self.broadcast_data({
                "type": "pathfinding_response",
                "request_type": "intercept",
                "data": result
            })
            
        elif request_type == "evacuation":
            # Calculate evacuation routes from a location
            current_location = request_data.get("location")
            
            if current_location in self.pathfinder.location_nodes:
                node = self.pathfinder.location_nodes[current_location]
            else:
                node = current_location
                
            routes = self.pathfinder.find_evacuation_routes(node)
            
            await self.broadcast_data({
                "type": "pathfinding_response",
                "request_type": "evacuation",
                "data": {
                    "location": current_location,
                    "routes": routes
                }
            })
            
        elif request_type == "update_officer":
            # Update officer position
            officer_id = request_data.get("officer_id")
            new_position = request_data.get("position")
            
            result = self.pathfinder.update_officer_position(officer_id, new_position)
            
            await self.broadcast_data({
                "type": "pathfinding_response",
                "request_type": "update_officer",
                "data": result
            })

def create_detection_payload(face_detections, gun_detections, frame, location="Main Campus"):
    """Create a JSON payload with detection results."""
    
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    threats = []
    persons = []
    
    for bbox, label, is_shooter in face_detections:
        person_name = label.split(' (')[0]
        confidence = float(label.split('(')[1].split(')')[0])
        
        person_data = {
            "name": person_name,
            "confidence": confidence,
            "bbox": bbox.tolist(),
            "is_threat": is_shooter,
            "location": location
        }
        
        if is_shooter:
            threats.append({
                "type": "armed_person",
                "person": person_name,
                "confidence": confidence,
                "location": location,
                "timestamp": datetime.now().isoformat()
            })
        
        persons.append(person_data)
    
    timestamp = datetime.now().isoformat()
    weapons = []
    for bbox, label in gun_detections:
        confidence = float(label.split()[-1])
        weapons.append({
            "type": "weapon",
            "weapon_type": label.split()[0],
            "confidence": confidence,
            "bbox": bbox.tolist(),
            "location": location,
            "timestamp": timestamp
        })

    # Generate threats based on detections
    if weapons:
        recognized_persons = [p for p in persons if p["name"] != "Unknown"]
        armed_person_threats = [p for p in persons if p.get("is_threat")]
        first_weapon = weapons[0]

        # Prioritize explicitly identified shooters
        if armed_person_threats:
            for person in armed_person_threats:
                # Avoid duplicate threats if already added
                if not any(t["type"] == "armed_person" and t["person"] == person["name"] for t in threats):
                    threats.append({
                        "type": "armed_person",
                        "person": person["name"],
                        "weapon_type": first_weapon["weapon_type"],
                        "confidence": person["confidence"],
                        "location": location,
                        "timestamp": timestamp
                    })
        # If a gun is detected and a known person is in frame, associate it
        elif recognized_persons:
            person = recognized_persons[0]
            threats.append({
                "type": "armed_person",
                "person": person["name"],
                "weapon_type": first_weapon["weapon_type"],
                "confidence": person["confidence"],
                "location": location,
                "timestamp": timestamp
            })
        # Otherwise, create a general weapon threat
        else:
            threats.append({
                "type": "weapon_detected",
                "weapon_type": first_weapon["weapon_type"],
                "confidence": first_weapon["confidence"],
                "location": location,
                "timestamp": timestamp
            })
    
    return {
        "type": "detection_data",
        "timestamp": timestamp,
        "location": location,
        "frame": frame_base64,
        "detections": {
            "persons": persons,
            "weapons": weapons,
            "threat_count": len(threats)
        },
        "threats": threats,
        "status": "alert" if threats else "clear"
    }

async def websocket_handler(websocket, path, detector):
    """Handle WebSocket connections and messages."""
    await detector.register_client(websocket)
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                # Handle different message types
                if data.get("type") == "pathfinding_request":
                    await detector.handle_pathfinding_request(data)
                    
                elif data.get("type") == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))
                    
            except json.JSONDecodeError:
                print(f"Invalid JSON received: {message}")
            except Exception as e:
                print(f"Error handling message: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        await detector.unregister_client(websocket)

async def process_video_stream(detector, gun_conf_thresh=0.75):
    """Process video stream and send detection results via WebSocket."""
    
    cap = cv2.VideoCapture(0)
    print("\nStarting webcam feed with WebSocket server...")
    
    PROC_WIDTH = 640
    PROC_HEIGHT = 480
    
    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue()
    
    def worker(frame_queue, result_queue, detector, gun_conf_thresh):
        while True:
            try:
                frame = frame_queue.get()
                if frame is None:
                    break
                face_detections, gun_detections, all_keypoints = detector.detect_and_recognize(
                    frame, gun_conf_thresh=gun_conf_thresh
                )
                result_queue.put((face_detections, gun_detections, all_keypoints))
                frame_queue.task_done()
            except queue.Empty:
                continue
    
    thread = threading.Thread(
        target=worker, 
        args=(frame_queue, result_queue, detector, gun_conf_thresh), 
        daemon=True
    )
    thread.start()
    
    face_detections, gun_detections, all_keypoints = [], [], None
    
    # Camera location mapping (you can modify based on your setup)
    camera_locations = ["Cafeteria", "Hallway A", "Gym", "Library"]
    current_camera_index = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            orig_h, orig_w = frame.shape[:2]
            scale_w, scale_h = orig_w / PROC_WIDTH, orig_h / PROC_HEIGHT
            
            if frame_queue.empty():
                resized_frame = cv2.resize(frame, (PROC_WIDTH, PROC_HEIGHT))
                frame_queue.put(resized_frame)
            
            try:
                face_detections, gun_detections, all_keypoints = result_queue.get_nowait()
                
                # Get current camera location
                current_location = camera_locations[current_camera_index % len(camera_locations)]
                
                # Send detection data
                detection_data = create_detection_payload(
                    face_detections, 
                    gun_detections, 
                    resized_frame,
                    location=current_location
                )
                
                # If threats detected, update threat locations for pathfinding
                if detection_data["threats"]:
                    for threat in detection_data["threats"]:
                        if threat["type"] == "armed_person":
                            detector.threat_locations[threat["person"]] = current_location
                
                await detector.broadcast_data(detection_data)
                
                # Auto-calculate intercept paths when threats detected
                if detection_data["threats"] and len(detector.pathfinder.officer_positions) > 0:
                    for officer_id in detector.pathfinder.officer_positions.keys():
                        intercept_result = detector.pathfinder.find_intercept_path(
                            current_location, officer_id
                        )
                        if intercept_result["status"] == "success":
                            intercept_result["floor_plan"] = detector.pathfinder.render_path_on_floor_plan(
                                intercept_result["path"]
                            )
                            await detector.broadcast_data({
                                "type": "auto_intercept",
                                "data": intercept_result
                            })
                
            except queue.Empty:
                pass
            
            # Draw on frame for local display
            if all_keypoints:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, all_keypoints, mp.solutions.pose.POSE_CONNECTIONS
                )
            
            if face_detections:
                for bbox, label, is_shooter in face_detections:
                    x1, y1, x2, y2 = [int(i) for i in bbox]
                    box_color = (0, 0, 255) if is_shooter else (0, 255, 0)
                    cv2.rectangle(frame, (int(x1*scale_w), int(y1*scale_h)), 
                                (int(x2*scale_w), int(y2*scale_h)), box_color, 2)
                    cv2.putText(frame, label, (int(x1*scale_w), int(y1*scale_h) - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
            if gun_detections:
                for xyxy, label in gun_detections:
                    x1, y1, x2, y2 = [int(i) for i in xyxy]
                    cv2.rectangle(frame, (int(x1*scale_w), int(y1*scale_h)), 
                                (int(x2*scale_w), int(y2*scale_h)), (255, 0, 0), 2)
                    cv2.putText(frame, label, (int(x1*scale_w), int(y1*scale_h) - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            cv2.imshow('Combined Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Switch camera location (for testing)
                current_camera_index += 1
                print(f"Switched to camera: {camera_locations[current_camera_index % len(camera_locations)]}")
            
            await asyncio.sleep(0.1)
            
    finally:
        frame_queue.put(None)
        thread.join()
        cap.release()
        cv2.destroyAllWindows()

async def main():
    # --- Configuration ---
    DET_WEIGHT = "./weights/det_10g.onnx"
    REC_WEIGHT = "./weights/w600k_r50.onnx"
    GUN_WEIGHT = "./best.pt"
    FACES_DIR = "./faces"
    
    # Pathfinding configuration
    GEXF_FILE = "./BMHS_FloorPlan_combined.gexf"  # Update path
    FLOOR_PLAN_IMAGE = "./BMHS_FloorPlan.JPG"     # Update path
    
    GUN_CONF_THRESHOLD = 0.75
    WEBSOCKET_PORT = 8765
    
    detector = CombinedDetectorWithPathfinding(
        det_weight=DET_WEIGHT,
        rec_weight=REC_WEIGHT,
        gun_weight=GUN_WEIGHT,
        faces_dir=FACES_DIR,
        gexf_file=GEXF_FILE,
        floor_plan_image=FLOOR_PLAN_IMAGE
    )
    
    # Start WebSocket server
    server = await websockets.serve(
        lambda ws, path: websocket_handler(ws, path, detector),
        "localhost",
        WEBSOCKET_PORT
    )
    
    print(f"WebSocket server started on ws://localhost:{WEBSOCKET_PORT}")
    print("Press 'c' to cycle through camera locations")
    print("Press 'q' to quit")
    
    # Process video stream
    await process_video_stream(detector, GUN_CONF_THRESHOLD)
    
    # Close server
    server.close()
    await server.wait_closed()

if __name__ == '__main__':
    asyncio.run(main())