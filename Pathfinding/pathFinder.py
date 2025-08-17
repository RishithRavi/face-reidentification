import cv2
import numpy as np
import networkx as nx
import json
import pyttsx3

def load_combined_graph(gexf_file, original_image_path):
    """Load the combined graph with multi-floor data"""
    img = cv2.imread(original_image_path)
    if img is None:
        raise FileNotFoundError(f"Original image not found at {original_image_path}")
    
    G = nx.read_gexf(gexf_file)
    
    img_copy = img.copy()
    
    # Extract node positions and floor information
    pos = {}
    floor_info = {}
    for node, data in G.nodes(data=True):
        pos[node] = (float(data['pos_x']), float(data['pos_y']))
        if 'floor' in data:
            floor_info[node] = data['floor']
    
    return G, pos, floor_info, img_copy

def draw_graph(img, G, pos, floor_info=None):
    """Draw the graph with floor-specific coloring"""
    # Draw edges first
    for u, v, data in G.edges(data=True):
        color = (0, 200, 255)  # Default orange for regular edges
        
        # Check if this is a stair connection
        if 'is_stair' in data and data['is_stair'] == 'true':
            color = (255, 0, 255)  # Purple for stairs
        
        if 'path' in data:
            try:
                path = json.loads(data['path'])
                pts = np.array([(int(x), int(y)) for x, y in path], dtype=np.int32)
                cv2.polylines(img, [pts], False, color, 2)
            except:
                pass
    
    # Then draw nodes
    for node, (x, y) in pos.items():
        # Determine node color based on floor if available
        if floor_info and node in floor_info:
            # Simple color coding based on floor name hash
            floor_hash = hash(floor_info[node]) % 5
            colors = [
                (0, 255, 0),   # Green
                (255, 0, 0),   # Blue
                (0, 0, 255),   # Red
                (255, 255, 0), # Cyan
                (0, 255, 255)  # Yellow
            ]
            color = colors[floor_hash]
        else:
            color = (0, 255, 0)  # Default green
        
        cv2.circle(img, (int(x), int(y)), 8, color, -1)
        cv2.putText(img, node, (int(x)+10, int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def find_and_draw_shortest_path(G, pos, floor_info, img, node1, node2):
    """Find and draw shortest path, handling multi-floor paths without drawing lines between stairs"""
    try:
        # Initialize TTS engine
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)  # Adjust speech rate
        tts_engine.setProperty('volume', 0.9)  # Adjust volume

        # Check if nodes are the same
        if node1 == node2:
            cv2.circle(img, (int(pos[node1][0]), int(pos[node1][1])), 12, (0, 0, 255), -1)
            cv2.putText(img, "Same node!", (int(pos[node1][0])+15, int(pos[node1][1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print("Start and end nodes are the same")
            tts_engine.say("Start and end nodes are the same")
            tts_engine.runAndWait()
            return None

        # Find shortest path
        path_nodes = nx.shortest_path(G, source=node1, target=node2, weight='weight')
        path_length = nx.shortest_path_length(G, source=node1, target=node2, weight='weight')
        
        # Create a copy of the image to draw on
        path_img = img.copy()
        
        # Draw regular path segments first (non-stair connections)
        for i in range(len(path_nodes)-1):
            u, v = path_nodes[i], path_nodes[i+1]
            if G.has_edge(u, v):
                edge_data = G.get_edge_data(u, v)
                
                # Skip stair connections
                if 'is_stair' in edge_data and edge_data['is_stair'] == 'true':
                    continue
                    
                # Draw regular path segments
                if 'path' in edge_data:
                    try:
                        path_points = json.loads(edge_data['path'])
                        if path_points:
                            pts = np.array([(int(x), int(y)) for x, y in path_points], dtype=np.int32)
                            cv2.polylines(path_img, [pts], False, (0, 0, 255), 4)
                    except:
                        pass

        # Draw stair connections as special markers (no lines)
        stair_nodes = set()
        for i in range(len(path_nodes)-1):
            u, v = path_nodes[i], path_nodes[i+1]
            if G.has_edge(u, v):
                edge_data = G.get_edge_data(u, v)
                if 'is_stair' in edge_data and edge_data['is_stair'] == 'true':
                    stair_nodes.add(u)
                    stair_nodes.add(v)
                    # Draw stair indicators at node positions
                    cv2.circle(path_img, (int(pos[u][0]), int(pos[u][1])), 12, (255, 0, 255), -1)
                    cv2.circle(path_img, (int(pos[v][0]), int(pos[v][1])), 12, (255, 0, 255), -1)
                    cv2.putText(path_img, "STAIR", (int(pos[u][0])+15, int(pos[u][1])),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Highlight start and end nodes
        cv2.circle(path_img, (int(pos[node1][0]), int(pos[node1][1])), 12, (0, 0, 255), -1)
        cv2.circle(path_img, (int(pos[node2][0]), int(pos[node2][1])), 12, (0, 0, 255), -1)
        
        # Display path length (find a visible segment to place the text)
        text_placed = False
        for i in range(len(path_nodes)-1):
            u, v = path_nodes[i], path_nodes[i+1]
            if G.has_edge(u, v):
                edge_data = G.get_edge_data(u, v)
                if 'is_stair' not in edge_data or edge_data['is_stair'] != 'true':
                    if 'path' in edge_data:
                        try:
                            path_points = json.loads(edge_data['path'])
                            if len(path_points) > 0:
                                mid_point = path_points[len(path_points)//2]
                                cv2.putText(path_img, f"{path_length:.1f} px", 
                                           (int(mid_point[0]), int(mid_point[1])),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                text_placed = True
                                break
                        except:
                            continue
        
        if not text_placed:
            # Fallback position if no suitable path segment found
            mid_x = (pos[node1][0] + pos[node2][0]) / 2
            mid_y = (pos[node1][1] + pos[node2][1]) / 2
            cv2.putText(path_img, f"{path_length:.1f} px",
                       (int(mid_x), int(mid_y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Print path details including floor transitions
        print("\n=== Path Details ===")
        current_floor = floor_info.get(path_nodes[0], "Unknown")
        start_msg = f"Start at {path_nodes[0]} on {current_floor}"
        print(start_msg)
        tts_engine.say(start_msg)
        
        for i in range(1, len(path_nodes)):
            node = path_nodes[i]
            new_floor = floor_info.get(node, "Unknown")
            if new_floor != current_floor:
                stair_msg = f"Take stairs from {path_nodes[i-1]} on {current_floor} to {node} on {new_floor}"
                print(f"  {stair_msg}")
                tts_engine.say(stair_msg)
                current_floor = new_floor
            
            move_msg = f"Move to {node} on {current_floor}"
            print(f"  {move_msg}")
            tts_engine.say(move_msg)
        
        arrival_msg = f"Arrived at destination. Total distance: {path_length:.1f} pixels"
        print(f"\n{arrival_msg}")
        print("===================")
        tts_engine.say(arrival_msg)
        tts_engine.runAndWait()  # Speak all queued messages

        # Blend the path visualization with the original image
        cv2.addWeighted(path_img, 0.7, img, 0.3, 0, img)
        
        return path_nodes
        
    except nx.NetworkXNoPath:
        error_msg = f"No path exists between {node1} and {node2}"
        print(error_msg)
        tts_engine.say(error_msg)
        tts_engine.runAndWait()
        return None
    except Exception as e:
        error_msg = f"Error finding path: {e}"
        print(error_msg)
        tts_engine.say(error_msg)
        tts_engine.runAndWait()
        return None

def main():
    gexf_file = "/Users/vibhushsivakumar/Desktop/DreamSafety/Pathfinding/BMHS_FloorPlan_combined.gexf"
    original_image_path = "/Users/vibhushsivakumar/Desktop/DreamSafety/Pathfinding/BMHS_FloorPlan.JPG"
    
    try:
        G, pos, floor_info, img = load_combined_graph(gexf_file, original_image_path)
    except Exception as e:
        print(f"Error loading files: {e}")
        return
    
    cv2.namedWindow("Multi-Floor Pathfinder", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Multi-Floor Pathfinder", 1000, 800)
    
    draw_graph(img, G, pos, floor_info)
    cv2.imshow("Multi-Floor Pathfinder", img)
    
    selected_nodes = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_nodes, img
        
        if event == cv2.EVENT_LBUTTONDOWN:
            closest_node = None
            min_dist = float('inf')
            
            for node, (nx_pos, ny_pos) in pos.items():
                dist = np.sqrt((x - nx_pos)**2 + (y - ny_pos)**2)
                if dist < min_dist and dist < 20:
                    min_dist = dist
                    closest_node = node
            
            if closest_node:
                selected_nodes.append(closest_node)
                print(f"Selected node: {closest_node} on {floor_info.get(closest_node, 'Unknown')}")
                
                # Reset display
                img = cv2.imread(original_image_path)
                draw_graph(img, G, pos, floor_info)
                
                # Highlight selected nodes
                for node in selected_nodes:
                    cv2.circle(img, (int(pos[node][0]), int(pos[node][1])), 
                              12, (255, 0, 0), -1)
                
                cv2.imshow("Multi-Floor Pathfinder", img)
                
                # Find and draw path when two nodes are selected
                if len(selected_nodes) == 2:
                    path = find_and_draw_shortest_path(G, pos, floor_info, img, 
                                                     selected_nodes[0], selected_nodes[1])
                    cv2.imshow("Multi-Floor Pathfinder", img)
                    selected_nodes = []
    
    cv2.setMouseCallback("Multi-Floor Pathfinder", mouse_callback)
    
    print("Instructions:")
    print("1. Click on two nodes to find the shortest path between them (can be on different floors)")
    print("2. The path will be shown in red (purple for stairs between floors)")
    print("3. Press 'r' to reset the view")
    print("4. Press 'q' to quit")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            img = cv2.imread(original_image_path)
            draw_graph(img, G, pos, floor_info)
            cv2.imshow("Multi-Floor Pathfinder", img)
            selected_nodes = []
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()