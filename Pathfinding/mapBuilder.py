import cv2
import numpy as np
import networkx as nx
import json
import os

class MultiFloorEditor:
    def __init__(self):
        self.floors = {}  # Dictionary to store floor data
        self.current_floor = None
        self.combined_graph = nx.Graph()
        self.stair_connections = []
        
        # Drawing state
        self.drawing = False
        self.current_path = []
        self.nodes = {}
        self.stair_selection = None
        
        # Full image
        self.full_image = None
        self.display_image = None

    def configure_floors(self, image_path, floor_names):
        """
        Configure multiple floors within a single image
        floor_names = ['Floor1', 'Floor2', ...]
        """
        self.full_image = cv2.imread(image_path)
        if self.full_image is None:
            print(f"Error: Image not found at {image_path}")
            return False
            
        self.display_image = self.full_image.copy()
        self.image_path = image_path
        
        # Initialize floor data
        for floor_name in floor_names:
            # Initialize empty floor data
            self.floors[floor_name] = {
                'graph': nx.Graph(),
                'nodes': {},
                'color': tuple(np.random.randint(0, 255, 3).tolist())
            }
            
        # Try to load combined graph if it exists
        combined_path = f"{os.path.splitext(image_path)[0]}_combined.gexf"
        if os.path.exists(combined_path):
            self.load_combined_graph(combined_path)
        else:
            # Load individual floor graphs if they exist
            for floor_name in floor_names:
                gexf_path = f"{os.path.splitext(image_path)[0]}_{floor_name}.gexf"
                if os.path.exists(gexf_path):
                    G = nx.read_gexf(gexf_path)
                    # Convert position attributes to float
                    nodes = {}
                    for node, data in G.nodes(data=True):
                        nodes[node] = (float(data['pos_x']), float(data['pos_y']))
                    self.floors[floor_name]['graph'] = G
                    self.floors[floor_name]['nodes'] = nodes
                    self.combined_graph = nx.compose(self.combined_graph, G)
                
        if not self.current_floor and floor_names:
            self.current_floor = floor_names[0]
                
        return True

    def load_combined_graph(self, gexf_path):
        """Load a combined graph file and distribute nodes to their respective floors"""
        if not os.path.exists(gexf_path):
            return False
            
        G = nx.read_gexf(gexf_path)
        
        # Clear existing data
        for floor_data in self.floors.values():
            floor_data['graph'].clear()
            floor_data['nodes'].clear()
        self.stair_connections = []
        self.combined_graph = nx.Graph()
        
        # Load nodes into their respective floors
        for node, data in G.nodes(data=True):
            if 'floor' in data:
                floor_name = data['floor']
                if floor_name in self.floors:
                    pos_x = float(data['pos_x'])
                    pos_y = float(data['pos_y'])
                    self.floors[floor_name]['nodes'][node] = (pos_x, pos_y)
                    self.floors[floor_name]['graph'].add_node(node, pos_x=pos_x, pos_y=pos_y)
        
        # Load edges (both regular and stair connections)
        for u, v, data in G.edges(data=True):
            if 'is_stair' in data and data['is_stair'] == 'true':
                # Find which floors these nodes belong to
                floor1 = None
                floor2 = None
                for floor_name, floor_data in self.floors.items():
                    if u in floor_data['nodes']:
                        floor1 = floor_name
                    if v in floor_data['nodes']:
                        floor2 = floor_name
                
                if floor1 and floor2 and floor1 != floor2:
                    self.stair_connections.append({
                        'floor1': floor1, 'node1': u,
                        'floor2': floor2, 'node2': v
                    })
            else:
                # Regular edge - add to the appropriate floor's graph
                for floor_name, floor_data in self.floors.items():
                    if u in floor_data['nodes'] and v in floor_data['nodes']:
                        floor_data['graph'].add_edge(u, v, 
                                                   weight=float(data['weight']),
                                                   path=data.get('path', '[]'))
                        break
        
        # Rebuild combined graph
        self.combined_graph = nx.compose_all([floor_data['graph'] for floor_data in self.floors.values()])
        
        # Add stair connections to combined graph
        for conn in self.stair_connections:
            self.combined_graph.add_edge(conn['node1'], conn['node2'],
                                       weight=10, path='[]', is_stair=True)
        
        return True

    def save_combined_graph(self):
        """Save all floors and connections to a single combined graph file"""
        combined_path = f"{os.path.splitext(self.image_path)[0]}_combined.gexf"
        
        # Create a new graph for saving
        G_save = nx.Graph()
        
        # Add all nodes with floor information
        for floor_name, floor_data in self.floors.items():
            for node, pos in floor_data['nodes'].items():
                G_save.add_node(node, 
                              pos_x=str(pos[0]), 
                              pos_y=str(pos[1]),
                              floor=floor_name)
        
        # Add all regular edges
        for floor_name, floor_data in self.floors.items():
            for u, v, data in floor_data['graph'].edges(data=True):
                G_save.add_edge(u, v, 
                              weight=str(data['weight']), 
                              path=data.get('path', '[]'),
                              is_stair='false')
        
        # Add stair connections
        for conn in self.stair_connections:
            G_save.add_edge(conn['node1'], conn['node2'],
                          weight='10',
                          path='[]',
                          is_stair='true')
        
        nx.write_gexf(G_save, combined_path)
        print(f"Saved combined graph to {combined_path}")
        return True


    def get_closest_node(self, x, y, floor_name=None):
        """Find the closest node to the given coordinates on specified floor (or current floor)"""
        floor = self.floors[floor_name or self.current_floor]
        if not floor['nodes']:
            return None
            
        closest_node = None
        min_distance = float('inf')
        
        for node, (node_x, node_y) in floor['nodes'].items():
            distance = np.sqrt((node_x - x)**2 + (node_y - y)**2)
            if distance < min_distance:
                min_distance = distance
                closest_node = node
        
        return closest_node if min_distance < 20 else None

    def add_stair_connection(self, floor1, node1, floor2, node2):
        """Connect two nodes on different floors as stairs"""
        if (floor1 not in self.floors or floor2 not in self.floors or
            node1 not in self.floors[floor1]['nodes'] or 
            node2 not in self.floors[floor2]['nodes']):
            return False
            
        # Add to combined graph
        self.combined_graph.add_edge(node1, node2, weight=10, path="[]", is_stair=True)
        
        self.stair_connections.append({
            'floor1': floor1, 'node1': node1,
            'floor2': floor2, 'node2': node2
        })
        return True

    def redraw_display(self):
        """Redraw the full display image with all floors"""
        self.display_image = self.full_image.copy()
        
        # Draw all nodes and edges
        for floor_name, floor_data in self.floors.items():
            color = floor_data['color']
            
            # Draw edges
            for u, v, data in floor_data['graph'].edges(data=True):
                if 'path' in data:
                    try:
                        path = json.loads(data['path'])
                        pts = np.array([(int(x), int(y)) for x, y in path], dtype=np.int32)
                        cv2.polylines(self.display_image, [pts], False, color, 2)
                    except:
                        pass
            
            # Draw nodes
            for node, (x, y) in floor_data['nodes'].items():
                node_color = color
                # Highlight if part of stair connection
                for conn in self.stair_connections:
                    if conn['node1'] == node or conn['node2'] == node:
                        node_color = (255, 0, 255)  # Purple for stair nodes
                        break
                cv2.circle(self.display_image, (int(x), int(y)), 8, node_color, -1)
                # Label current floor's nodes with their IDs
                #if floor_name == self.current_floor:
                    #cv2.putText(self.display_image, node, (int(x)+10, int(y)), 
                               #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        
        # Highlight current floor
        cv2.putText(self.display_image, f"Current Floor: {self.current_floor}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.floors[self.current_floor]['color'], 2)
        
        cv2.imshow("Multi-Floor Editor", self.display_image)

    def run_editor(self):
        """Main editor loop"""
        if not self.current_floor:
            print("No floors loaded!")
            return
            
        cv2.namedWindow("Multi-Floor Editor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Multi-Floor Editor", 1000, 800)
        
        # Draw initial state
        self.redraw_display()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if mode == "add_nodes":
                    node_id = f"{self.current_floor}_Node_{len(self.floors[self.current_floor]['nodes'])+1}"
                    self.floors[self.current_floor]['nodes'][node_id] = (x, y)
                    self.floors[self.current_floor]['graph'].add_node(node_id, pos_x=float(x), pos_y=float(y))
                    self.combined_graph.add_node(node_id, pos_x=float(x), pos_y=float(y), floor=self.current_floor)
                    self.redraw_display()
                
                elif mode == "add_edges":
                    self.drawing = True
                    self.current_path = [(x, y)]
                    self.redraw_display()
                    cv2.circle(self.display_image, (x, y), 3, (0, 0, 255), -1)
                    cv2.imshow("Multi-Floor Editor", self.display_image)
                
                elif mode == "add_stairs":
                    node = self.get_closest_node(x, y)
                    if node:
                        if not self.stair_selection:
                            self.stair_selection = (self.current_floor, node)
                            print(f"Selected {node} on {self.current_floor}. Now select destination node.")
                        else:
                            if self.current_floor != self.stair_selection[0] or node != self.stair_selection[1]:
                                self.add_stair_connection(
                                    self.stair_selection[0], self.stair_selection[1],
                                    self.current_floor, node
                                )
                                print(f"Connected {self.stair_selection[1]} on {self.stair_selection[0]} to {node} on {self.current_floor}")
                            else:
                                print("Cannot connect a node to itself on the same floor")
                            self.stair_selection = None
                            self.redraw_display()
            
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing and mode == "add_edges":
                    cv2.line(self.display_image, self.current_path[-1], (x, y), (255, 0, 0), 2)
                    self.current_path.append((x, y))
                    cv2.imshow("Multi-Floor Editor", self.display_image)
                    
            elif event == cv2.EVENT_LBUTTONUP:
                if self.drawing and mode == "add_edges":
                    self.drawing = False
                    if len(self.current_path) >= 2:
                        # Find closest nodes on current floor
                        start_node = self.get_closest_node(*self.current_path[0])
                        end_node = self.get_closest_node(*self.current_path[-1])
                        
                        if start_node and end_node:
                            # Calculate distance
                            distance = sum(np.linalg.norm(np.array(self.current_path[i]) - np.array(self.current_path[i-1]))
                                           for i in range(1, len(self.current_path)))
                            
                            # Add edge
                            path_str = json.dumps(self.current_path)
                            self.floors[self.current_floor]['graph'].add_edge(
                                start_node, end_node,
                                weight=round(distance, 1),
                                path=path_str
                            )
                            self.combined_graph.add_edge(
                                start_node, end_node,
                                weight=round(distance, 1),
                                path=path_str
                            )
                            
                            self.redraw_display()
                    self.current_path = []
        
        # Set initial mode
        mode = "add_nodes"
        cv2.setMouseCallback("Multi-Floor Editor", mouse_callback)
        
        print("Controls:")
        print("1. Press 'n' to add nodes mode")
        print("2. Press 'e' to add edges mode")
        print("3. Press 's' to add stair connections mode")
        print("4. Press '1', '2' etc. to switch floors")
        print("5. Press 'c' to save combined graph")
        print("6. Press 'q' to quit")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                mode = "add_nodes"
                self.stair_selection = None
                print("Mode: Add Nodes")
            elif key == ord('e'):
                mode = "add_edges"
                self.stair_selection = None
                print("Mode: Add Edges")
            elif key == ord('s'):
                mode = "add_stairs"
                self.stair_selection = None
                print("Mode: Add Stairs - Select first node")
            elif key >= ord('1') and key <= ord('9'):
                floor_idx = key - ord('1')
                floor_names = list(self.floors.keys())
                if floor_idx < len(floor_names):
                    self.current_floor = floor_names[floor_idx]
                    self.redraw_display()
                    print(f"Switched to {self.current_floor}")
            elif key == ord('c'):
                self.save_combined_graph()
        
        cv2.destroyAllWindows()


# Example usage:
if __name__ == "__main__":
    editor = MultiFloorEditor()
    
    # Configure floors - just provide names, no position bounds needed
    image_path = "/Users/vibhushsivakumar/Desktop/DreamSafety/Pathfinding/BMHS_FloorPlan.JPG"
    floor_names = ['Floor1', 'Floor2', 'Floor3']  # Now with 3 floors

    if editor.configure_floors(image_path, floor_names):
        editor.run_editor()
    else:
        print("Failed to load floors")