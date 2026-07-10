"""
Intelligent Autonomous Driving System with LIDAR, Traffic Rules, and Realistic Behavior
Chandigarh Autonomous Driving Simulator
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List, Tuple, Dict, Any, Optional
import math
import random
import osmnx as ox
import networkx as nx
from dataclasses import dataclass, field
import uvicorn
import time
import numpy as np
from enum import Enum

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# ENUMS AND CONSTANTS
# ============================================================

class SensorType(Enum):
    LIDAR = "lidar"
    RADAR = "radar"
    CAMERA = "camera"
    ULTRASONIC = "ultrasonic"
    GPS = "gps"
    IMU = "imu"

class TrafficLightState(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"

class VehicleType(Enum):
    SEDAN = "sedan"
    SUV = "suv"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    HATCHBACK = "hatchback"

@dataclass
class LIDARPoint:
    x: float
    y: float
    distance: float
    angle: float
    intensity: float
    object_type: str = "unknown"

@dataclass
class SensorData:
    lidar_points: List[LIDARPoint] = field(default_factory=list)
    radar_detections: List[Dict] = field(default_factory=list)
    camera_objects: List[Dict] = field(default_factory=list)
    gps_position: Tuple[float, float] = (0, 0)
    imu_data: Dict = field(default_factory=dict)
    timestamp: float = 0.0

@dataclass
class VehicleState:
    x: float
    y: float
    heading: float
    speed: float
    acceleration: float
    steering: float
    is_self_driving: bool = True
    distance_traveled: float = 0.0
    distance_remaining: float = 0.0
    eta: float = 0.0
    traveled_path: List[Tuple[float, float]] = None
    current_speed_limit: float = 0.0
    upcoming_turn_distance: float = 0.0
    upcoming_turn_sharpness: float = 0.0
    sensor_data: SensorData = field(default_factory=SensorData)
    
    def __post_init__(self):
        if self.traveled_path is None:
            self.traveled_path = []
        if not self.sensor_data:
            self.sensor_data = SensorData()

# ============================================================
# MAP LOADER WITH BLACK/WHITE THEME
# ============================================================

class MapLoader:
    def __init__(self):
        self.graph = None
        self.nodes = {}
        self.edges = []
        self.roads = []
        self.bounds = None
        self.center = None
        self.intersections = []
        self.traffic_light_positions = []
        
    def load_map(self):
        try:
            print("Loading Chandigarh road network...")
            self.graph = ox.graph_from_place(
                "Chandigarh, India",
                network_type="drive",
                simplify=True
            )
            self.graph = ox.project_graph(self.graph)
            
            for node_id, data in self.graph.nodes(data=True):
                if 'x' in data and 'y' in data:
                    self.nodes[node_id] = (data['x'], data['y'])
                    
                    # Identify intersections
                    if self.graph.degree(node_id) >= 3:
                        self.intersections.append(node_id)
            
            for u, v, data in self.graph.edges(data=True):
                geometry = data.get('geometry')
                if geometry is None:
                    if u in self.nodes and v in self.nodes:
                        geometry = [(self.nodes[u][0], self.nodes[u][1]),
                                   (self.nodes[v][0], self.nodes[v][1])]
                elif hasattr(geometry, 'coords'):
                    geometry = list(geometry.coords)
                
                if geometry and len(geometry) > 0:
                    if len(geometry) > 20:
                        simplified = self._simplify_path(geometry, 3.0)
                    else:
                        simplified = geometry
                    
                    self.edges.append({
                        'u': u, 'v': v,
                        'geometry': simplified,
                        'length': data.get('length', 0),
                        'highway': data.get('highway', ''),
                        'lanes': data.get('lanes', 1)
                    })
                    self.roads.append(simplified)
            
            # Calculate bounds
            xs = [p[0] for p in self.nodes.values()]
            ys = [p[1] for p in self.nodes.values()]
            self.bounds = (min(xs), min(ys), max(xs), max(ys))
            self.center = ((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2)
            
            # Generate traffic light positions at intersections (2 per km²)
            self._generate_traffic_lights()
            
            print(f"Loaded {len(self.nodes)} nodes, {len(self.edges)} edges")
            print(f"Found {len(self.intersections)} intersections")
            print(f"Generated {len(self.traffic_light_positions)} traffic lights")
            return True
            
        except Exception as e:
            print(f"Error loading map: {e}")
            return False
    
    def _simplify_path(self, path: List[Tuple[float, float]], tolerance: float) -> List[Tuple[float, float]]:
        if len(path) <= 2:
            return path
        
        def point_line_distance(point, line_start, line_end):
            x0, y0 = point
            x1, y1 = line_start
            x2, y2 = line_end
            
            if x1 == x2 and y1 == y2:
                return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
            
            numerator = abs((x2 - x1)*(y1 - y0) - (x1 - x0)*(y2 - y1))
            denominator = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            return numerator / denominator
        
        max_dist = 0
        max_index = 0
        for i in range(1, len(path) - 1):
            dist = point_line_distance(path[i], path[0], path[-1])
            if dist > max_dist:
                max_dist = dist
                max_index = i
        
        if max_dist > tolerance:
            left = self._simplify_path(path[:max_index + 1], tolerance)
            right = self._simplify_path(path[max_index:], tolerance)
            return left[:-1] + right
        else:
            return [path[0], path[-1]]
    
    def _generate_traffic_lights(self):
        """Generate traffic lights at intersections (2 per km²)"""
        if not self.intersections:
            return
        
        # Select 2 intersections per km² of map area
        map_width = self.bounds[2] - self.bounds[0]
        map_height = self.bounds[3] - self.bounds[1]
        area_km2 = (map_width * map_height) / 1_000_000
        num_lights = max(1, int(area_km2 * 2))
        
        # Randomly select intersections
        selected = random.sample(self.intersections, min(num_lights, len(self.intersections)))
        
        self.traffic_light_positions = []
        for node_id in selected:
            if node_id in self.nodes:
                x, y = self.nodes[node_id]
                # Get road directions from this intersection
                directions = []
                for neighbor in self.graph.neighbors(node_id):
                    if neighbor in self.nodes:
                        nx, ny = self.nodes[neighbor]
                        angle = math.atan2(ny - y, nx - x)
                        directions.append(angle)
                
                if directions:
                    # Create lights for each direction
                    for i, direction in enumerate(directions[:4]):  # Max 4 directions
                        self.traffic_light_positions.append({
                            'x': x + 15 * math.cos(direction),
                            'y': y + 15 * math.sin(direction),
                            'direction': direction,
                            'state': 'green' if i % 3 == 0 else 'yellow' if i % 3 == 1 else 'red',
                            'timer': random.uniform(0, 15)
                        })

# ============================================================
# INTELLIGENT PATH PLANNER WITH REAL-TIME UPDATES
# ============================================================

class PathPlanner:
    def __init__(self, graph, nodes):
        self.graph = graph
        self.nodes = nodes
        self.current_path = []
        self.path_cache = {}
        
    def find_nearest_node(self, x: float, y: float) -> int:
        nearest = None
        min_dist = float('inf')
        
        for node_id, (nx, ny) in self.nodes.items():
            dist = math.sqrt((nx - x)**2 + (ny - y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = node_id
        
        return nearest
    
    def plan_route(self, start: Tuple[float, float], end: Tuple[float, float]) -> List[Tuple[float, float]]:
        if self.graph is None:
            return []
        
        start_node = self.find_nearest_node(start[0], start[1])
        end_node = self.find_nearest_node(end[0], end[1])
        
        if start_node == end_node:
            return [start, end]
        
        # Check cache
        cache_key = (start_node, end_node)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        try:
            path = nx.astar_path(
                self.graph,
                start_node,
                end_node,
                heuristic=lambda u, v: self._heuristic(u, v),
                weight='length'
            )
            
            route = []
            for node_id in path:
                if node_id in self.nodes:
                    route.append(self.nodes[node_id])
            
            # Cache the path
            self.path_cache[cache_key] = route
            if len(self.path_cache) > 100:
                self.path_cache.pop(next(iter(self.path_cache)))
            
            self.current_path = route
            return route
            
        except Exception as e:
            print(f"Path planning failed: {e}")
            return []
    
    def _heuristic(self, u: int, v: int) -> float:
        if u in self.nodes and v in self.nodes:
            ux, uy = self.nodes[u]
            vx, vy = self.nodes[v]
            return math.sqrt((ux - vx)**2 + (uy - vy)**2)
        return 0

# ============================================================
# LIDAR SENSOR SYSTEM
# ============================================================

class LIDARSystem:
    def __init__(self, num_rays=360, max_range=150.0):
        self.num_rays = num_rays
        self.max_range = max_range
        self.angle_increment = (2 * math.pi) / num_rays
        self.noise_level = 0.02
        
    def scan(self, position: Tuple[float, float], heading: float, 
             obstacles: List[Tuple[float, float]]) -> List[LIDARPoint]:
        """Simulate LIDAR scan with obstacles"""
        points = []
        x, y = position
        
        for i in range(self.num_rays):
            angle = heading - math.pi/2 + i * self.angle_increment
            
            # Ray casting
            found_obstacle = False
            min_distance = self.max_range
            
            for obs_x, obs_y in obstacles:
                # Check if obstacle is in ray direction
                dx = obs_x - x
                dy = obs_y - y
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist > self.max_range:
                    continue
                
                # Calculate angle to obstacle
                obs_angle = math.atan2(dy, dx)
                angle_diff = self._normalize_angle(obs_angle - angle)
                
                # Check if obstacle is within ray cone (narrow for LIDAR)
                if abs(angle_diff) < 0.02:  # ~1 degree
                    if dist < min_distance:
                        min_distance = dist
                        found_obstacle = True
            
            # Add noise to distance
            distance = min_distance + random.gauss(0, self.noise_level)
            if distance < 0:
                distance = 0.1
            
            # Calculate point position
            px = x + distance * math.cos(angle)
            py = y + distance * math.sin(angle)
            
            intensity = 1.0 - (distance / self.max_range)
            if found_obstacle:
                intensity *= (0.7 + random.random() * 0.3)
            else:
                intensity *= 0.1
            
            points.append(LIDARPoint(
                x=px, y=py,
                distance=distance,
                angle=angle,
                intensity=intensity,
                object_type="vehicle" if found_obstacle and distance < 50 else "unknown"
            ))
        
        return points
    
    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

# ============================================================
# INTELLIGENT VEHICLE CONTROLLER
# ============================================================

class IntelligentVehicleController:
    def __init__(self):
        self.state = VehicleState(0, 0, 0, 0, 0, 0, True, 0, 0, 0, [])
        self.waypoints = []
        self.current_waypoint = 0
        self.path = []
        self.is_following_path = False
        self.total_route_distance = 0
        self.last_position = (0, 0)
        self.smooth_speed = 0
        self.start_time = time.time()
        
        # LIDAR system
        self.lidar = LIDARSystem(num_rays=180, max_range=120.0)
        
        # Advanced speed control parameters
        self.max_speed_straight = 22.2  # 80 km/h
        self.max_speed_gentle = 18.0    # 65 km/h
        self.max_speed_moderate = 13.9  # 50 km/h
        self.max_speed_sharp = 8.3      # 30 km/h
        self.max_speed_very_sharp = 5.56  # 20 km/h - capped
        
        self.max_acceleration = 6.0
        self.max_brake = 10.0
        self.max_steering = 0.7
        self.wheelbase = 2.8
        self.lookahead_distance = 40.0
        self.turn_preparation_distance = 35.0
        self.reaction_time = 0.2
        
        # Traffic rules
        self.following_distance = 15.0
        self.safe_gap = 5.0
        self.at_traffic_light = False
        self.traffic_light_state = "green"
        
    def set_route(self, route: List[Tuple[float, float]]):
        self.path = route
        self.waypoints = route
        self.current_waypoint = 0
        self.is_following_path = True
        self.last_position = (self.state.x, self.state.y)
        self.state.traveled_path = [(self.state.x, self.state.y)]
        self.start_time = time.time()
        
        self.total_route_distance = 0
        for i in range(len(route) - 1):
            dx = route[i+1][0] - route[i][0]
            dy = route[i+1][1] - route[i][1]
            dist = math.sqrt(dx*dx + dy*dy)
            self.total_route_distance += dist
        
        self.state.distance_remaining = self.total_route_distance
        self.state.distance_traveled = 0
    
    def _calculate_curvature(self, point1, point2, point3):
        x1, y1 = point1
        x2, y2 = point2
        x3, y3 = point3
        
        v1x = x1 - x2
        v1y = y1 - y2
        v2x = x3 - x2
        v2y = y3 - y2
        
        len1 = math.sqrt(v1x*v1x + v1y*v1y)
        len2 = math.sqrt(v2x*v2x + v2y*v2y)
        
        if len1 < 0.001 or len2 < 0.001:
            return 0.0
        
        v1x /= len1
        v1y /= len1
        v2x /= len2
        v2y /= len2
        
        dot = v1x * v2x + v1y * v2y
        dot = max(-1.0, min(1.0, dot))
        angle = math.acos(dot)
        
        return angle / (len1 + len2)
    
    def _find_upcoming_turn(self, current_pos):
        if not self.waypoints or self.current_waypoint >= len(self.waypoints) - 2:
            return 0.0, float('inf'), -1
        
        max_curvature = 0.0
        turn_distance = float('inf')
        accumulated_distance = 0.0
        
        idx = self.current_waypoint
        
        while idx < len(self.waypoints) - 2 and accumulated_distance < 80.0:
            if idx < len(self.waypoints) - 1:
                dx = self.waypoints[idx+1][0] - self.waypoints[idx][0]
                dy = self.waypoints[idx+1][1] - self.waypoints[idx][1]
                segment_dist = math.sqrt(dx*dx + dy*dy)
                accumulated_distance += segment_dist
            
            if idx > 0 and idx < len(self.waypoints) - 1:
                curvature = self._calculate_curvature(
                    self.waypoints[idx-1],
                    self.waypoints[idx],
                    self.waypoints[idx+1]
                )
                
                if curvature > 0.03:
                    if curvature > max_curvature:
                        max_curvature = curvature
                        turn_distance = accumulated_distance - segment_dist/2
            
            idx += 1
        
        return max_curvature, turn_distance
    
    def _get_speed_limit(self, curvature, distance_to_turn, traffic_light_state):
        base_limit = self.max_speed_straight
        
        # Adjust for curvature
        if curvature < 0.03:
            base_limit = self.max_speed_straight
        elif curvature < 0.05:
            base_limit = self.max_speed_gentle
        elif curvature < 0.1:
            base_limit = self.max_speed_moderate
        elif curvature < 0.2:
            base_limit = self.max_speed_sharp
        else:
            base_limit = self.max_speed_very_sharp
        
        # Prepare for turn
        if distance_to_turn < self.turn_preparation_distance and distance_to_turn > 0:
            progress = 1.0 - (distance_to_turn / self.turn_preparation_distance)
            slowdown_factor = max(0.3, 1.0 - progress * 0.7)
            base_limit = max(self.max_speed_very_sharp, base_limit * slowdown_factor)
        
        # Traffic light adjustment
        if traffic_light_state == "red":
            base_limit = min(base_limit, 2.0)  # Stop
        elif traffic_light_state == "yellow":
            base_limit = min(base_limit, 8.0)  # Slow down
        
        # Safety - don't exceed reasonable limits
        return min(base_limit, 25.0)
    
    def _process_sensor_data(self, dt: float):
        """Process LIDAR and other sensor data"""
        current_pos = (self.state.x, self.state.y)
        
        # Get nearby vehicles for LIDAR detection
        obstacles = []
        if hasattr(self, 'nearby_vehicles'):
            for vehicle in self.nearby_vehicles:
                if vehicle != (self.state.x, self.state.y):
                    obstacles.append((vehicle[0], vehicle[1]))
        
        # Perform LIDAR scan
        lidar_points = self.lidar.scan(
            current_pos, 
            self.state.heading,
            obstacles
        )
        
        # Update sensor data
        self.state.sensor_data.lidar_points = lidar_points
        self.state.sensor_data.timestamp = time.time()
        
        # Detect obstacles in front
        front_obstacles = []
        for point in lidar_points:
            # Check if point is in front (within 60 degree cone)
            angle_diff = self._normalize_angle(point.angle - self.state.heading)
            if abs(angle_diff) < 0.5 and point.distance < 50:
                front_obstacles.append(point)
        
        return front_obstacles
    
    def update(self, dt: float, traffic_lights: List[Dict], nearby_vehicles: List[Tuple[float, float]] = None):
        if not self.is_following_path or not self.waypoints:
            self.state.acceleration = 0
            return
        
        if self.current_waypoint >= len(self.waypoints):
            self.is_following_path = False
            self.state.acceleration = 0
            return
        
        self.nearby_vehicles = nearby_vehicles or []
        
        target = self.waypoints[self.current_waypoint]
        dx = target[0] - self.state.x
        dy = target[1] - self.state.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Check if reached waypoint
        if distance < 2.0:
            self.current_waypoint += 1
            self._update_remaining_distance()
            return
        
        # Calculate target angle
        target_angle = math.atan2(dy, dx)
        angle_diff = self._normalize_angle(target_angle - self.state.heading)
        
        # Steering
        self.state.steering = max(-self.max_steering, 
                                  min(self.max_steering, angle_diff * 2.5))
        
        # ----- SENSOR PROCESSING -----
        front_obstacles = self._process_sensor_data(dt)
        
        # Check traffic light state
        traffic_light_state = "green"
        for light in traffic_lights:
            lx, ly = light['x'], light['y']
            dist_to_light = math.sqrt((lx - self.state.x)**2 + (ly - self.state.y)**2)
            if dist_to_light < 30:
                traffic_light_state = light['state']
                self.at_traffic_light = True
                break
            else:
                self.at_traffic_light = False
        
        # ----- FIND UPCOMING TURN -----
        current_pos = (self.state.x, self.state.y)
        max_curvature, turn_distance = self._find_upcoming_turn(current_pos)
        
        # ----- SPEED CONTROL -----
        # Get speed limit based on road conditions
        speed_limit = self._get_speed_limit(max_curvature, turn_distance, traffic_light_state)
        self.state.current_speed_limit = speed_limit * 3.6
        
        # Obstacle avoidance - emergency braking
        min_obstacle_dist = float('inf')
        for obs in front_obstacles:
            if obs.distance < min_obstacle_dist:
                min_obstacle_dist = obs.distance
        
        if min_obstacle_dist < 15.0:
            # Emergency braking
            speed_limit = min(speed_limit, max(2.0, min_obstacle_dist / 3.0))
            self.state.acceleration = -self.max_brake * 1.5
        
        # If close to destination, slow down
        if self.current_waypoint >= len(self.waypoints) - 1 and distance < 30:
            speed_limit = max(2.78, speed_limit * (distance / 30))
        
        # Smooth acceleration/deceleration
        speed_error = speed_limit - self.state.speed
        if speed_error > 0:
            self.state.acceleration = min(self.max_acceleration, speed_error * 0.4)
            self.state.speed += self.state.acceleration * dt
        else:
            # Brake harder for obstacles and turns
            brake_force = self.max_brake
            if turn_distance < self.turn_preparation_distance:
                brake_force *= 1.5
            if min_obstacle_dist < 20:
                brake_force *= 1.8
            self.state.acceleration = -brake_force * min(1.0, abs(speed_error) / 5.0)
            self.state.speed += self.state.acceleration * dt
        
        self.state.speed = max(0, self.state.speed)
        
        # Update position
        new_x = self.state.x + self.state.speed * math.cos(self.state.heading) * dt
        new_y = self.state.y + self.state.speed * math.sin(self.state.heading) * dt
        
        dx_traveled = new_x - self.state.x
        dy_traveled = new_y - self.state.y
        traveled_dist = math.sqrt(dx_traveled*dx_traveled + dy_traveled*dy_traveled)
        self.state.distance_traveled += traveled_dist
        self.state.distance_remaining -= traveled_dist
        
        # Record traveled path
        if len(self.state.traveled_path) == 0 or traveled_dist > 5.0:
            self.state.traveled_path.append((new_x, new_y))
            if len(self.state.traveled_path) > 300:
                self.state.traveled_path = self.state.traveled_path[-150:]
        
        self.state.x = new_x
        self.state.y = new_y
        
        # Update heading
        if self.state.speed > 0.1:
            turn_radius = self.wheelbase / math.tan(self.state.steering) if abs(self.state.steering) > 0.001 else float('inf')
            if turn_radius != float('inf'):
                self.state.heading += (self.state.speed / turn_radius) * dt
        
        self.state.heading = self._normalize_angle(self.state.heading)
        
        # Calculate ETA
        if self.state.speed > 1.0:
            self.state.eta = self.state.distance_remaining / self.state.speed
        else:
            self.state.eta = float('inf')
        
        self.smooth_speed += (self.state.speed - self.smooth_speed) * 0.1
    
    def _update_remaining_distance(self):
        remaining = 0
        for i in range(self.current_waypoint, len(self.waypoints) - 1):
            dx = self.waypoints[i+1][0] - self.waypoints[i][0]
            dy = self.waypoints[i+1][1] - self.waypoints[i][1]
            remaining += math.sqrt(dx*dx + dy*dy)
        self.state.distance_remaining = remaining
    
    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

# ============================================================
# TRAFFIC MANAGER WITH 100+ CARS
# ============================================================

class TrafficManager:
    def __init__(self, nodes, num_cars=100):
        self.nodes = nodes
        self.vehicles = []
        self.next_id = 0
        self.spawn_timer = 0
        self.num_cars = num_cars
        self.colors = [
            '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
            '#1abc9c', '#e67e22', '#34495e', '#e84393', '#00b894',
            '#fd79a8', '#0984e3', '#fdcb6e', '#6c5ce7', '#00cec9',
            '#f8a5c2', '#74b9ff', '#55efc4', '#ffeaa7', '#a29bfe'
        ]
        self.vehicle_types = list(VehicleType)
        
        # Pre-spawn vehicles
        self._spawn_initial_vehicles()
        
    def _spawn_initial_vehicles(self):
        """Spawn initial fleet of vehicles"""
        for i in range(self.num_cars):
            self._spawn_vehicle()
    
    def _spawn_vehicle(self):
        if not self.nodes:
            return
        
        # Pick random node
        node_id = random.choice(list(self.nodes.keys()))
        x, y = self.nodes[node_id]
        
        heading = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 8)
        color = random.choice(self.colors)
        vehicle_type = random.choice(self.vehicle_types)
        
        # Different sizes based on vehicle type
        size = 1.0
        if vehicle_type == VehicleType.SUV:
            size = 1.2
        elif vehicle_type == VehicleType.TRUCK or vehicle_type == VehicleType.BUS:
            size = 1.5
        elif vehicle_type == VehicleType.MOTORCYCLE:
            size = 0.6
        elif vehicle_type == VehicleType.HATCHBACK:
            size = 0.8
        
        self.vehicles.append({
            'id': self.next_id,
            'x': x, 'y': y,
            'heading': heading,
            'speed': speed,
            'color': color,
            'type': vehicle_type.value,
            'size': size,
            'destination': None,
            'route': [],
            'current_waypoint': 0
        })
        self.next_id += 1
    
    def update(self, dt: float, player_pos: Tuple[float, float]):
        """Update all traffic vehicles"""
        # Maintain vehicle count
        if len(self.vehicles) < self.num_cars:
            self.spawn_timer += dt
            if self.spawn_timer > 0.5:
                self._spawn_vehicle()
                self.spawn_timer = 0
        
        # Update each vehicle
        for vehicle in self.vehicles[:]:
            # Simple AI movement
            if not vehicle.get('route') or len(vehicle.get('route', [])) == 0:
                # Wander randomly
                vehicle['heading'] += random.uniform(-0.02, 0.02) * dt
                vehicle['heading'] = self._normalize_angle(vehicle['heading'])
                vehicle['speed'] = max(1, min(8, vehicle['speed'] + random.uniform(-0.1, 0.1)))
            else:
                # Follow route
                route = vehicle['route']
                wp_index = vehicle.get('current_waypoint', 0)
                if wp_index < len(route):
                    target = route[wp_index]
                    dx = target[0] - vehicle['x']
                    dy = target[1] - vehicle['y']
                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    if dist < 5.0:
                        vehicle['current_waypoint'] = wp_index + 1
                    else:
                        target_angle = math.atan2(dy, dx)
                        angle_diff = self._normalize_angle(target_angle - vehicle['heading'])
                        vehicle['heading'] += angle_diff * 2.0 * dt
                        vehicle['heading'] = self._normalize_angle(vehicle['heading'])
                        vehicle['speed'] = min(8, vehicle['speed'] + 0.2)
            
            # Update position
            vehicle['x'] += vehicle['speed'] * math.cos(vehicle['heading']) * dt
            vehicle['y'] += vehicle['speed'] * math.sin(vehicle['heading']) * dt
            
            # Remove if too far from player
            dx = player_pos[0] - vehicle['x']
            dy = player_pos[1] - vehicle['y']
            if math.sqrt(dx*dx + dy*dy) > 2000:
                # Respawn instead of remove
                self._respawn_vehicle(vehicle)
    
    def _respawn_vehicle(self, vehicle):
        """Respawn a vehicle at a new location"""
        if not self.nodes:
            return
        
        node_id = random.choice(list(self.nodes.keys()))
        x, y = self.nodes[node_id]
        vehicle['x'] = x
        vehicle['y'] = y
        vehicle['heading'] = random.uniform(0, 2 * math.pi)
        vehicle['speed'] = random.uniform(2, 8)
        vehicle['route'] = []
        vehicle['current_waypoint'] = 0
    
    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

# ============================================================
# TRAFFIC LIGHT MANAGER
# ============================================================

class TrafficLightManager:
    def __init__(self, light_positions):
        self.lights = []
        self._init_lights(light_positions)
        
    def _init_lights(self, positions):
        for pos in positions:
            self.lights.append({
                'x': pos['x'],
                'y': pos['y'],
                'direction': pos.get('direction', 0),
                'state': pos.get('state', 'green'),
                'timer': pos.get('timer', 0)
            })
    
    def update(self, dt: float):
        for light in self.lights:
            light['timer'] += dt
            
            if light['state'] == 'green' and light['timer'] > 15:
                light['state'] = 'yellow'
                light['timer'] = 0
            elif light['state'] == 'yellow' and light['timer'] > 3:
                light['state'] = 'red'
                light['timer'] = 0
            elif light['state'] == 'red' and light['timer'] > 12:
                light['state'] = 'green'
                light['timer'] = 0
    
    def get_lights(self):
        return self.lights

# ============================================================
# GLOBAL STATE
# ============================================================

map_loader = MapLoader()
planner = None
vehicle = IntelligentVehicleController()
traffic_manager = None
traffic_light_manager = None
last_update_time = time.time()

# ============================================================
# API ENDPOINTS
# ============================================================

@app.on_event("startup")
async def startup_event():
    global planner, traffic_manager, traffic_light_manager
    
    print("Loading Chandigarh map...")
    map_loader.load_map()
    
    if map_loader.graph:
        planner = PathPlanner(map_loader.graph, map_loader.nodes)
        
        # Initialize traffic with 100 vehicles
        traffic_manager = TrafficManager(map_loader.nodes, num_cars=100)
        
        # Initialize traffic lights
        traffic_light_manager = TrafficLightManager(map_loader.traffic_light_positions)
        
        if map_loader.center:
            vehicle.state.x = map_loader.center[0]
            vehicle.state.y = map_loader.center[1]
            vehicle.state.heading = 0
            vehicle.state.traveled_path = [(vehicle.state.x, vehicle.state.y)]
            vehicle.start_time = time.time()
    
    print("🚗 Intelligent Autonomous Driving System Ready!")
    print(f"📊 Traffic: {len(traffic_manager.vehicles) if traffic_manager else 0} vehicles")
    print(f"🚦 Traffic Lights: {len(traffic_light_manager.lights) if traffic_light_manager else 0}")

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

@app.get("/api/map")
async def get_map():
    return {
        'roads': map_loader.roads,
        'bounds': map_loader.bounds,
        'center': map_loader.center,
        'nodes': list(map_loader.nodes.values()),
        'intersections': map_loader.intersections
    }

@app.post("/api/plan_route")
async def plan_route(data: Dict[str, Any]):
    if not planner:
        return {'success': False, 'error': 'Planner not initialized'}
    
    start = data.get('start')
    end = data.get('end')
    
    if not start or not end:
        return {'success': False, 'error': 'Invalid points'}
    
    route = planner.plan_route(tuple(start), tuple(end))
    
    if route:
        vehicle.set_route(route)
        vehicle.start_time = time.time()
        return {
            'success': True,
            'route': route,
            'length': len(route),
            'total_distance': vehicle.total_route_distance
        }
    else:
        return {'success': False, 'error': 'No route found'}

@app.get("/api/state")
async def get_state():
    global last_update_time
    
    if not map_loader:
        return {'success': False}
    
    current_time = time.time()
    dt = min(current_time - last_update_time, 0.05)
    last_update_time = current_time
    
    # Get traffic lights
    traffic_lights = traffic_light_manager.get_lights() if traffic_light_manager else []
    traffic_light_manager.update(dt) if traffic_light_manager else None
    
    # Get all vehicle positions for LIDAR
    all_vehicle_positions = []
    if traffic_manager:
        for v in traffic_manager.vehicles:
            all_vehicle_positions.append((v['x'], v['y']))
    
    # Update vehicle with traffic lights and nearby vehicles
    vehicle.update(dt, traffic_lights, all_vehicle_positions)
    
    # Update traffic
    if traffic_manager:
        traffic_manager.update(dt, (vehicle.state.x, vehicle.state.y))
    
    # Format ETA
    eta_str = "∞"
    if vehicle.state.eta != float('inf') and vehicle.state.eta > 0:
        minutes = int(vehicle.state.eta // 60)
        seconds = int(vehicle.state.eta % 60)
        eta_str = f"{minutes:02d}:{seconds:02d}"
    
    # Prepare LIDAR data for visualization
    lidar_points = []
    for point in vehicle.state.sensor_data.lidar_points:
        lidar_points.append({
            'x': point.x,
            'y': point.y,
            'distance': point.distance,
            'angle': point.angle,
            'intensity': point.intensity
        })
    
    return {
        'success': True,
        'player': {
            'x': vehicle.state.x,
            'y': vehicle.state.y,
            'heading': vehicle.state.heading,
            'speed': vehicle.state.speed * 3.6,
            'is_self_driving': vehicle.state.is_self_driving,
            'waypoint_index': vehicle.current_waypoint,
            'total_waypoints': len(vehicle.waypoints),
            'distance_traveled': vehicle.state.distance_traveled / 1000,
            'distance_remaining': vehicle.state.distance_remaining / 1000,
            'eta': eta_str,
            'eta_seconds': vehicle.state.eta,
            'traveled_path': vehicle.state.traveled_path,
            'current_speed_limit': vehicle.state.current_speed_limit,
            'upcoming_turn_distance': vehicle.state.upcoming_turn_distance,
            'upcoming_turn_sharpness': vehicle.state.upcoming_turn_sharpness,
            'lidar_points': lidar_points
        },
        'path': vehicle.path,
        'traffic_lights': traffic_lights,
        'ai_vehicles': traffic_manager.vehicles if traffic_manager else [],
        'stats': {
            'vehicle_count': len(traffic_manager.vehicles) if traffic_manager else 0,
            'waypoint_index': vehicle.current_waypoint,
            'total_waypoints': len(vehicle.waypoints),
            'total_distance': vehicle.total_route_distance / 1000,
            'traffic_light_count': len(traffic_lights)
        }
    }

@app.post("/api/control")
async def control_vehicle(data: Dict[str, Any]):
    if 'throttle' in data:
        vehicle.state.acceleration = data['throttle'] * vehicle.max_acceleration
    if 'brake' in data and data['brake']:
        vehicle.state.speed *= 0.95
    if 'steering' in data:
        vehicle.state.steering = data['steering'] * vehicle.max_steering
    if 'self_driving' in data:
        vehicle.state.is_self_driving = data['self_driving']
        if data['self_driving'] and vehicle.path:
            vehicle.is_following_path = True
    
    return {'success': True}

@app.post("/api/reset")
async def reset_simulation():
    global vehicle, traffic_manager
    
    if map_loader.center:
        vehicle = IntelligentVehicleController()
        vehicle.state.x = map_loader.center[0]
        vehicle.state.y = map_loader.center[1]
        vehicle.state.heading = 0
        vehicle.state.traveled_path = [(vehicle.state.x, vehicle.state.y)]
        vehicle.start_time = time.time()
        
        if traffic_manager:
            traffic_manager.vehicles = []
            traffic_manager._spawn_initial_vehicles()
    
    return {'success': True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)