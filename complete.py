import traci
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import threading
import time
import random
import sqlite3
from datetime import datetime
import queue
from threading import Lock
import os
import sys
from xml.etree import ElementTree as ET

class TrafficCluster:
    """Represents a cluster of traffic lights that can be controlled as a unit"""
    def __init__(self, name, traffic_lights):
        self.name = name
        self.traffic_lights = traffic_lights
        self.current_phase = "red"
        self.time_remaining = 30
        self.vehicle_count = 0
        self.automatic_mode = False
        self.red_time = 30
        self.yellow_time = 5
        self.green_time = 30
        
    def set_phase(self, phase):
        self.current_phase = phase
        for tl_id in self.traffic_lights:
            try:
                current_state = traci.trafficlight.getRedYellowGreenState(tl_id)
                state_length = len(current_state)
                if phase == "red":
                    new_state = "r" * state_length
                elif phase == "yellow":
                    new_state = "y" * state_length
                elif phase == "green":
                    new_state = "G" * state_length
                traci.trafficlight.setRedYellowGreenState(tl_id, new_state)
                print(f"Set {tl_id} to {phase} state: {new_state}")
            except traci.TraCIException as e:
                print(f"Error setting phase for traffic light {tl_id}: {e}")
                continue
            
    def update_vehicle_count(self):
        total_count = 0
        for tl_id in self.traffic_lights:
            try:
                lanes = traci.trafficlight.getControlledLanes(tl_id)
                for lane in lanes:
                    total_count += traci.lane.getLastStepVehicleNumber(lane)
            except traci.TraCIException as e:
                print(f"Error counting vehicles for {tl_id}: {e}")
                continue
        self.vehicle_count = total_count
        
    def get_traffic_density(self):
        if self.vehicle_count < 15:
            return "low"
        elif self.vehicle_count < 30:
            return "medium"
        else:
            return "high"

class RSU:
    """Road Side Unit implementation for collecting and analyzing traffic data"""
    def __init__(self, id, position, radius=100):
        self.id = id
        self.position = position  # (x, y) coordinates
        self.radius = radius      # detection radius in meters
        self.vehicles_in_range = set()
        self.vehicle_data = {}    # Store vehicle data for analysis
        self.congestion_level = "low"
    
    def update(self):
        """Update the RSU with current vehicle information"""
        self.vehicles_in_range.clear()
        
        # Get all vehicles in the simulation
        try:
            vehicles = traci.vehicle.getIDList()
            
            for veh_id in vehicles:
                try:
                    veh_pos = traci.vehicle.getPosition(veh_id)
                    # Calculate distance between RSU and vehicle
                    distance = ((veh_pos[0] - self.position[0])**2 + (veh_pos[1] - self.position[1])**2)**0.5
                    
                    if distance <= self.radius:
                        self.vehicles_in_range.add(veh_id)
                        
                        # Collect data about the vehicle
                        self.vehicle_data[veh_id] = {
                            "speed": traci.vehicle.getSpeed(veh_id),
                            "edge": traci.vehicle.getRoadID(veh_id),
                            "waiting_time": traci.vehicle.getWaitingTime(veh_id),
                            "distance": distance
                        }
                except traci.TraCIException as e:
                    # Vehicle might have left the simulation
                    if veh_id in self.vehicle_data:
                        del self.vehicle_data[veh_id]
                    continue
                    
            # Update congestion level based on number of vehicles
            self.update_congestion_level()
        
        except traci.TraCIException as e:
            print(f"Error updating RSU {self.id}: {e}")
    
    def update_congestion_level(self):
        """Update the congestion level based on the number of vehicles and speeds"""
        num_vehicles = len(self.vehicles_in_range)
        avg_speed = 0
        
        if num_vehicles > 0:
            total_speed = sum(data["speed"] for data in self.vehicle_data.values() if data["speed"] > 0)
            avg_speed = total_speed / num_vehicles if num_vehicles > 0 else 0
        
        # Simple congestion heuristic
        if num_vehicles < 5:
            self.congestion_level = "low"
        elif num_vehicles < 15 or avg_speed > 10:
            self.congestion_level = "medium"
        else:
            self.congestion_level = "high"
    
    def get_recommended_phase(self):
        """Get recommended traffic light phase based on congestion level"""
        if self.congestion_level == "high":
            return "green", 45  # Longer green time for high congestion
        elif self.congestion_level == "medium":
            return "green", 30  # Standard green time for medium congestion
        else:
            return "green", 20  # Shorter green time for low congestion

class ClusterWindow:
    """UI window for controlling a traffic light cluster"""
    def __init__(self, cluster, parent):
        # Create new window for this cluster
        self.window = tk.Toplevel(parent)
        self.window.title(f"{cluster.name} Control")
        self.window.geometry("600x400")
        self.cluster = cluster
        
        # Main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header_frame, text=cluster.name, font=("Arial", 16, "bold")).pack(side=tk.LEFT)
        ttk.Label(header_frame, text=f"Traffic Lights: {', '.join(cluster.traffic_lights)}", 
                 font=("Arial", 10)).pack(side=tk.LEFT, padx=10)

        # Phase Control
        phase_frame = ttk.LabelFrame(main_frame, text="Phase Control", padding="10")
        phase_frame.pack(fill=tk.X, pady=5)
        btn_frame = ttk.Frame(phase_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="RED", 
                  command=lambda: self.set_phase("red")).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="YELLOW",
                  command=lambda: self.set_phase("yellow")).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="GREEN",
                  command=lambda: self.set_phase("green")).pack(side=tk.LEFT, padx=5)
        
        self.phase_label = ttk.Label(phase_frame, text="Current Phase: -", font=("Arial", 12))
        self.phase_label.pack(pady=5)

        # Timer Settings
        timer_frame = ttk.LabelFrame(main_frame, text="Timer Settings", padding="10")
        timer_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(timer_frame, text="Red Duration:").grid(row=0, column=0, padx=5)
        self.red_time = ttk.Entry(timer_frame, width=8)
        self.red_time.insert(0, str(cluster.red_time))
        self.red_time.grid(row=0, column=1, padx=5)
        
        ttk.Label(timer_frame, text="Yellow Duration:").grid(row=0, column=2, padx=5)
        self.yellow_time = ttk.Entry(timer_frame, width=8)
        self.yellow_time.insert(0, str(cluster.yellow_time))
        self.yellow_time.grid(row=0, column=3, padx=5)
        
        ttk.Label(timer_frame, text="Green Duration:").grid(row=0, column=4, padx=5)
        self.green_time = ttk.Entry(timer_frame, width=8)
        self.green_time.insert(0, str(cluster.green_time))
        self.green_time.grid(row=0, column=5, padx=5)
        
        ttk.Button(timer_frame, text="Apply Timers",
                  command=self.apply_timers).grid(row=1, column=0, columnspan=6, pady=10)

        # Automatic Mode
        auto_frame = ttk.LabelFrame(main_frame, text="Automatic Control", padding="10")
        auto_frame.pack(fill=tk.X, pady=5)
        
        self.auto_var = tk.BooleanVar(value=cluster.automatic_mode)
        ttk.Checkbutton(auto_frame, text="Enable Automatic Cycle",
                       variable=self.auto_var,
                       command=self.toggle_automatic).pack(pady=5)
        
        # Status display
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.pack(fill=tk.X, pady=5)
        
        self.time_label = ttk.Label(status_frame, text="Time Remaining: -", font=("Arial", 12))
        self.time_label.pack(pady=2)
        self.vehicle_label = ttk.Label(status_frame, text="Vehicles: 0", font=("Arial", 12))
        self.vehicle_label.pack(pady=2)
        self.density_label = ttk.Label(status_frame, text="Density: Low", font=("Arial", 12))
        self.density_label.pack(pady=2)
        
        self.progress = ttk.Progressbar(status_frame, mode='determinate', length=200)
        self.progress.pack(pady=5)

    def apply_timers(self):
        try:
            self.cluster.red_time = int(self.red_time.get())
            self.cluster.yellow_time = int(self.yellow_time.get())
            self.cluster.green_time = int(self.green_time.get())
            print(f"Updated timers for {self.cluster.name}")
        except ValueError:
            print("Invalid timer values")

    def toggle_automatic(self):
        self.cluster.automatic_mode = self.auto_var.get()
        if self.cluster.automatic_mode:
            self.cluster.time_remaining = self.cluster.red_time
            self.cluster.current_phase = "red"
            self.cluster.set_phase("red")
            
    def set_phase(self, phase):
        self.auto_var.set(False)
        self.cluster.automatic_mode = False
        self.cluster.set_phase(phase)

    def update(self):
        self.phase_label.config(text=f"Current Phase: {self.cluster.current_phase.upper()}")
        self.time_label.config(text=f"Time Remaining: {self.cluster.time_remaining}s")
        self.vehicle_label.config(text=f"Vehicles: {self.cluster.vehicle_count}")
        self.density_label.config(text=f"Density: {self.cluster.get_traffic_density().title()}")
        
        if self.cluster.automatic_mode:
            if self.cluster.current_phase == "red":
                max_time = self.cluster.red_time
            elif self.cluster.current_phase == "yellow":
                max_time = self.cluster.yellow_time
            else:
                max_time = self.cluster.green_time
            progress = ((max_time - self.cluster.time_remaining) / max_time) * 100
            self.progress['value'] = progress
        else:
            self.progress['value'] = 0

    def close(self):
        self.window.destroy()

class ScenarioManager:
    """Manages different traffic scenarios for what-if analysis"""
    def __init__(self):
        # Define scenarios with unique vehicle types and colors
        self.scenarios = {
            "Normal Traffic": {"density": 1, "vehicle_type": "normal_vehicle", "color": (0, 255, 0, 255)},  # Green
            "Rush Hour": {"density": 10, "vehicle_type": "rush_vehicle", "color": (255, 0, 0, 255)},  # Red
            "Rainy Day": {"density": 3, "vehicle_type": "rain_vehicle", "color": (0, 0, 255, 255), "max_speed": 15.0},  # Blue
            "Foggy Morning": {"density": 5, "vehicle_type": "fog_vehicle", "color": (128, 128, 128, 255), "max_speed": 10.0},  # Gray
            "Emergency": {"density": 2, "vehicle_type": "emergency_vehicle", "color": (255, 165, 0, 255)}  # Orange
        }
        
        # Initialize vehicle types for scenarios
        for scenario, config in self.scenarios.items():
            try:
                traci.vehicletype.copy("veh_passenger", config["vehicle_type"])
                traci.vehicletype.setColor(config["vehicle_type"], config["color"])
                
                # Set max speed for weather scenarios
                if "max_speed" in config:
                    traci.vehicletype.setMaxSpeed(config["vehicle_type"], config["max_speed"])
            except traci.TraCIException as e:
                print(f"Error setting up vehicle type for {scenario}: {e}")
    
    def load_trip_edges(self):
        """Load trip edges from the trip file"""
        try:
            tree = ET.parse('osm.passenger.trips.xml')
            root = tree.getroot()
            trip_edges = []
            for trip in root.findall('.//trip'):
                from_edge = trip.get('from')
                to_edge = trip.get('to')
                if from_edge and to_edge:
                    trip_edges.append((from_edge, to_edge))
            print(f"Loaded {len(trip_edges)} routes")
            return trip_edges
        except Exception as e:
            print(f"Error loading trip file: {e}")
            # Fallback to hardcoded trips
            return [
                ("24242882#0", "15973619#6"),
                ("4611711", "-120675240#0"),
                ("-28251222", "-120675240#2"),
                ("4611693#0", "-1105574291#1"),
                ("4611693#0", "15973619#8"),
                ("147066248#1", "-120675240#0"),
                ("4611693#0", "243854725#1"),
                ("120675240#0", "68647306#5"),
                ("4611708#2", "1159156576#1"),
                ("23017853", "-1132162834#0")
            ]
    
    def apply_scenario(self, scenario_name, intensity=1):
        """Apply a specific scenario with given intensity"""
        if scenario_name not in self.scenarios:
            print(f"Unknown scenario: {scenario_name}")
            return
            
        scenario = self.scenarios[scenario_name]
        vehicle_type = scenario["vehicle_type"]
        base_density = scenario["density"]
        adjusted_density = int(base_density * intensity)
        
        trip_edges = self.load_trip_edges()
        if not trip_edges:
            print("No valid routes found")
            return
            
        # Insert vehicles for this scenario
        for i in range(adjusted_density):
            vehicle_id = f"{scenario_name.replace(' ', '_')}_{int(traci.simulation.getTime())}_{i}"
            try:
                from_edge, to_edge = random.choice(trip_edges)
                route_id = f"route_{vehicle_id}"
                traci.route.add(routeID=route_id, edges=[from_edge, to_edge])
                traci.vehicle.add(vehID=vehicle_id, routeID=route_id, typeID=vehicle_type,
                                 departLane="best", departSpeed="max")
                print(f"Added {scenario_name} vehicle {vehicle_id}")
            except traci.TraCIException as e:
                print(f"Error adding vehicle {vehicle_id}: {e}")

class DatabaseManager:
    """Manages database operations for traffic simulation"""
    def __init__(self, db_path='traffic_simulation.db'):
        self.db_path = db_path
        self.db_queue = queue.Queue()
        self.db_lock = Lock()
        self.setup_database()
        
        # Start database worker thread
        self.db_thread = threading.Thread(target=self.db_worker)
        self.db_thread.daemon = True
        self.db_thread.start()
        
    def setup_database(self):
        """Initialize database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simulation stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulation_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                vehicle_count INTEGER,
                current_phase TEXT
            )
        ''')
        
        # Traffic light states table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffic_light_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                traffic_light_id TEXT,
                state TEXT,
                phase INTEGER
            )
        ''')

        # Vehicle stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                vehicle_id TEXT,
                edge_id TEXT,
                speed REAL,
                waiting_time REAL
            )
        ''')
        
        # RSU data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rsu_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                rsu_id TEXT,
                vehicle_count INTEGER,
                congestion_level TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def db_worker(self):
        """Worker thread for database operations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        while True:
            try:
                cmd, data = self.db_queue.get()
                if cmd == 'stop':
                    break
                    
                if cmd == 'log':
                    cursor.execute(
                        'INSERT INTO simulation_stats (timestamp, vehicle_count, current_phase) VALUES (?, ?, ?)',
                        (data['timestamp'], data['vehicle_count'], data['current_phase'])
                    )
                    
                    cursor.executemany(
                        'INSERT INTO traffic_light_states (timestamp, traffic_light_id, state, phase) VALUES (?, ?, ?, ?)',
                        data['tl_data']
                    )
                    
                    cursor.executemany(
                        'INSERT INTO vehicle_stats (timestamp, vehicle_id, edge_id, speed, waiting_time) VALUES (?, ?, ?, ?, ?)',
                        data['veh_data']
                    )
                    
                elif cmd == 'log_rsu':
                    cursor.executemany(
                        'INSERT INTO rsu_data (timestamp, rsu_id, vehicle_count, congestion_level) VALUES (?, ?, ?, ?)',
                        data['rsu_data']
                    )
                    
                conn.commit()
                
            except sqlite3.Error as e:
                print(f"Database error: {e}")
                
            self.db_queue.task_done()
        
        conn.close()
        
    def log_simulation_state(self, current_phase):
        """Queue simulation state data to be logged"""
        try:
            timestamp = datetime.now().isoformat()
            vehicle_count = len(traci.vehicle.getIDList())
            
            # Collect traffic light data
            tl_data = []
            for tl_id in traci.trafficlight.getIDList():
                state = traci.trafficlight.getRedYellowGreenState(tl_id)
                phase = traci.trafficlight.getPhase(tl_id)
                tl_data.append((timestamp, tl_id, state, phase))
            
            # Collect vehicle data
            veh_data = []
            for veh_id in traci.vehicle.getIDList():
                edge = traci.vehicle.getRoadID(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                waiting_time = traci.vehicle.getWaitingTime(veh_id)
                veh_data.append((timestamp, veh_id, edge, speed, waiting_time))
            
            # Queue the data for database thread
            self.db_queue.put(('log', {
                'timestamp': timestamp,
                'vehicle_count': vehicle_count,
                'current_phase': current_phase,
                'tl_data': tl_data,
                'veh_data': veh_data
            }))
            
        except traci.TraCIException as e:
            print(f"Error collecting simulation state: {e}")

    def log_rsu_data(self, rsus):
        """Queue RSU data to be logged"""
        try:
            timestamp = datetime.now().isoformat()
            rsu_data = []
            
            for rsu in rsus:
                rsu_data.append((
                    timestamp,
                    rsu.id,
                    len(rsu.vehicles_in_range),
                    rsu.congestion_level
                ))
            
            self.db_queue.put(('log_rsu', {
                'rsu_data': rsu_data
            }))
            
        except Exception as e:
            print(f"Error logging RSU data: {e}")
            
    def stop(self):
        """Stop the database worker thread"""
        self.db_queue.put(('stop', None))
        self.db_thread.join()

class TrafficSimulationGUI:
    """Main GUI for the traffic simulation application"""
    def __init__(self):
        # Initialize SUMO
        self.sumoBinary = "sumo-gui"
        self.sumoCmd = [self.sumoBinary, "-c", "osm.sumocfg"]
        try:
            traci.start(self.sumoCmd)
            print("SUMO started successfully")
        except Exception as e:
            print(f"Error starting SUMO: {e}")
            sys.exit(1)
        
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("Integrated Traffic Simulation System")
        self.root.geometry("1000x800")
        
        # Control variables
        self.running = False
        self.exit_simulation = False
        
        # Initialize managers
        self.tl_ids = traci.trafficlight.getIDList()
        self.db_manager = DatabaseManager()
        self.scenario_manager = ScenarioManager()
        
        # Create RSUs at traffic light junctions
        self.rsus = []
        self.create_rsus()
        
        # Create traffic light clusters
        self.clusters = {}
        self.cluster_windows = {}
        self.create_clusters()
        
        # Setup GUI components
        self.setup_gui()
        
        # Start simulation threads
        self.simulation_thread = threading.Thread(target=self.run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        self.timer_thread = threading.Thread(target=self.run_timer)
        self.timer_thread.daemon = True
        self.timer_thread.start()

    def create_rsus(self):
        """Create RSUs at traffic light junctions"""
        for i, tl_id in enumerate(self.tl_ids):
            try:
                junction_pos = traci.junction.getPosition(tl_id)
                rsu = RSU(f"RSU_{tl_id}", junction_pos)
                self.rsus.append(rsu)
                print(f"Created RSU at junction {tl_id}")
            except traci.TraCIException as e:
                print(f"Error creating RSU at junction {tl_id}: {e}")

    def create_clusters(self):
        """Create traffic light clusters"""
        # For simplicity, we'll create one cluster per traffic light
        # In a real implementation, you might want to group nearby traffic lights
        for i, tl_id in enumerate(self.tl_ids):
            try:
                cluster_id = f"cluster_{tl_id}"
                cluster_name = f"Traffic Light {tl_id}"
                
                self.clusters[cluster_id] = TrafficCluster(cluster_name, [tl_id])
                print(f"Created cluster for traffic light {tl_id}")
            except traci.TraCIException as e:
                print(f"Error creating cluster for traffic light {tl_id}: {e}")

    def setup_gui(self):
        """Setup the main GUI components"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Tab 1: Simulation Control
        self.sim_control_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.sim_control_frame, text="Simulation Control")
        self.setup_simulation_controls(self.sim_control_frame)
        
        # Tab 2: Traffic Light Control
        self.tl_control_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tl_control_frame, text="Traffic Lights")
        self.setup_traffic_light_controls(self.tl_control_frame)
        
        # Tab 3: What-If Scenarios
        self.scenario_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.scenario_frame, text="What-If Scenarios")
        self.setup_scenario_controls(self.scenario_frame)
        
        # Tab 4: RSU Data
        self.rsu_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.rsu_frame, text="RSU Data")
        self.setup_rsu_controls(self.rsu_frame)

    def setup_simulation_controls(self, parent):
        """Setup simulation control components"""
        control_frame = ttk.LabelFrame(parent, text="Simulation Controls", padding="10")
        control_frame.pack(fill="x", padx=5, pady=5)
        
        self.play_button = ttk.Button(control_frame, text="Play", command=self.toggle_simulation)
        self.play_button.pack(side="left", padx=5)
        
        ttk.Button(control_frame, text="Add Vehicles", command=self.insert_vehicles).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Stop Simulation", command=self.stop_simulation).pack(side="left", padx=5)
        
        # Simulation statistics
        stats_frame = ttk.LabelFrame(parent, text="Simulation Statistics", padding="10")
        stats_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.stats_text = tk.Text(stats_frame, wrap="word", height=20)
        self.stats_text.pack(fill="both", expand=True)
        
        # Scrollbar for stats text
        scrollbar = ttk.Scrollbar(stats_frame, command=self.stats_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.stats_text.config(yscrollcommand=scrollbar.set)

    def setup_traffic_light_controls(self, parent):
        """Setup traffic light control components"""
        global_control_frame = ttk.LabelFrame(parent, text="Global Traffic Light Controls", padding="10")
        global_control_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(global_control_frame, text="All RED", 
                  command=lambda: self.set_all_lights("r")).pack(side="left", padx=5)
        ttk.Button(global_control_frame, text="All YELLOW", 
                  command=lambda: self.set_all_lights("y")).pack(side="left", padx=5)
        ttk.Button(global_control_frame, text="All GREEN", 
                  command=lambda: self.set_all_lights("g")).pack(side="left", padx=5)
        
        # Traffic light cluster controls
        clusters_frame = ttk.LabelFrame(parent, text="Traffic Light Clusters", padding="10")
        clusters_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        for cluster_id, cluster in self.clusters.items():
            cluster_frame = ttk.Frame(clusters_frame)
            cluster_frame.pack(fill="x", pady=2)
            
            ttk.Label(cluster_frame, text=cluster.name, width=20).pack(side="left", padx=5)
            
            ttk.Button(cluster_frame, 
                      text="Control Cluster", 
                      command=lambda c=cluster: self.open_cluster_window(c)).pack(side="left", padx=5)
            
            # Quick control buttons
            ttk.Button(cluster_frame, text="RED", 
                      command=lambda c=cluster: c.set_phase("red")).pack(side="left", padx=2)
            ttk.Button(cluster_frame, text="YELLOW", 
                      command=lambda c=cluster: c.set_phase("yellow")).pack(side="left", padx=2)
            ttk.Button(cluster_frame, text="GREEN", 
                      command=lambda c=cluster: c.set_phase("green")).pack(side="left", padx=2)

    def setup_scenario_controls(self, parent):
        """Setup what-if scenario control components"""
        # Scenario selection
        scenario_selection_frame = ttk.LabelFrame(parent, text="Scenario Selection", padding="10")
        scenario_selection_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(scenario_selection_frame, text="Select Scenario:").pack(side="left", padx=5)
        
        self.selected_scenario = tk.StringVar(value="Normal Traffic")
        scenario_dropdown = ttk.Combobox(scenario_selection_frame, 
                                        textvariable=self.selected_scenario,
                                        values=list(self.scenario_manager.scenarios.keys()))
        scenario_dropdown.pack(side="left", padx=5)
        
        # Intensity slider
        intensity_frame = ttk.Frame(scenario_selection_frame)
        intensity_frame.pack(side="left", padx=20)
        
        ttk.Label(intensity_frame, text="Intensity:").pack(side="left")
        self.intensity_slider = ttk.Scale(intensity_frame, from_=1, to=20, orient="horizontal", length=200)
        self.intensity_slider.set(5)  # Default value
        self.intensity_slider.pack(side="left", padx=5)
        
        # Apply button
        ttk.Button(scenario_selection_frame, text="Apply Scenario",
                  command=self.apply_selected_scenario).pack(side="left", padx=20)
        
        # Scenario effects description
        effects_frame = ttk.LabelFrame(parent, text="Scenario Effects", padding="10")
        effects_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.effects_text = tk.Text(effects_frame, wrap="word", height=10)
        self.effects_text.pack(fill="both", expand=True)
        
        # Initial text
        self.update_scenario_description()
        
        # Event binding for dropdown change
        scenario_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_scenario_description())
        
        # Custom scenario builder
        custom_frame = ttk.LabelFrame(parent, text="Custom Scenario Builder", padding="10")
        custom_frame.pack(fill="x", padx=5, pady=5)
        
        # Vehicle count
        vehicle_frame = ttk.Frame(custom_frame)
        vehicle_frame.pack(fill="x", pady=5)
        
        ttk.Label(vehicle_frame, text="Vehicle Count:").pack(side="left", padx=5)
        self.vehicle_count_var = tk.IntVar(value=10)
        ttk.Entry(vehicle_frame, textvariable=self.vehicle_count_var, width=5).pack(side="left", padx=5)
        
        # Speed limit
        speed_frame = ttk.Frame(custom_frame)
        speed_frame.pack(fill="x", pady=5)
        
        ttk.Label(speed_frame, text="Max Speed (m/s):").pack(side="left", padx=5)
        self.speed_var = tk.DoubleVar(value=13.9)  # Default ~50 km/h
        ttk.Entry(speed_frame, textvariable=self.speed_var, width=5).pack(side="left", padx=5)
        
        # Vehicle Type
        type_frame = ttk.Frame(custom_frame)
        type_frame.pack(fill="x", pady=5)
        
        ttk.Label(type_frame, text="Vehicle Type:").pack(side="left", padx=5)
        self.vehicle_type_var = tk.StringVar(value="Custom")
        vehicle_types = ["Custom", "Emergency", "Heavy", "Light"]
        ttk.Combobox(type_frame, textvariable=self.vehicle_type_var, values=vehicle_types).pack(side="left", padx=5)
        
        # Apply custom scenario
        ttk.Button(custom_frame, text="Apply Custom Scenario",
                  command=self.apply_custom_scenario).pack(pady=10)

    def setup_rsu_controls(self, parent):
        """Setup RSU data display components"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(control_frame, text="RSU Data Refresh Rate (s):").pack(side="left", padx=5)
        self.refresh_rate = tk.IntVar(value=5)
        ttk.Entry(control_frame, textvariable=self.refresh_rate, width=5).pack(side="left", padx=5)
        
        # RSU data table
        table_frame = ttk.LabelFrame(parent, text="RSU Data", padding="10")
        table_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create Treeview
        columns = ("ID", "Location", "Vehicles", "Congestion", "Avg Speed", "Recommendation")
        self.rsu_tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        
        # Set column headings
        for col in columns:
            self.rsu_tree.heading(col, text=col)
            self.rsu_tree.column(col, width=100)
        
        self.rsu_tree.pack(fill="both", expand=True)
        
        # Add initial RSU data
        self.update_rsu_tree()
        
        # Maps/visualization placeholder frame
        map_frame = ttk.LabelFrame(parent, text="Traffic Visualization (Placeholder)", padding="10")
        map_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        map_label = ttk.Label(map_frame, text="Traffic visualization would be displayed here", font=("Arial", 14))
        map_label.pack(expand=True, pady=40)

    def update_scenario_description(self):
        """Update the scenario effects description based on selected scenario"""
        scenario = self.selected_scenario.get()
        
        descriptions = {
            "Normal Traffic": "Regular traffic flow with standard vehicle behavior. "
                            "This is the baseline scenario with normal speeds and densities.",
            
            "Rush Hour": "High density of vehicles simulating peak hours. "
                        "Expect congestion at intersections and reduced average speeds. "
                        "Traffic lights may need longer green phases to handle the volume.",
            
            "Rainy Day": "Reduced visibility and slower vehicle speeds due to wet conditions. "
                        "Vehicles maintain longer following distances and brake earlier. "
                        "Maximum speed is reduced to 15 m/s (~54 km/h).",
            
            "Foggy Morning": "Severely reduced visibility with much slower vehicle speeds. "
                            "Vehicles are extra cautious, with max speed limited to 10 m/s (~36 km/h). "
                            "Traffic density is moderate but flow is significantly affected.",
            
            "Emergency": "Emergency vehicles are given priority. "
                        "Regular traffic density but includes emergency vehicles that "
                        "other vehicles should yield to."
        }
        
        self.effects_text.delete(1.0, tk.END)
        
        if scenario in descriptions:
            intensity = self.intensity_slider.get()
            
            self.effects_text.insert(tk.END, f"{scenario} Scenario\n\n")
            self.effects_text.insert(tk.END, descriptions[scenario] + "\n\n")
            self.effects_text.insert(tk.END, f"Current Intensity: {intensity}/20\n")
            self.effects_text.insert(tk.END, f"Expected Vehicle Count: ~{intensity * self.scenario_manager.scenarios[scenario]['density']}\n")
            
            if scenario != "Normal Traffic":
                self.effects_text.insert(tk.END, "\nSpecial Effects:\n")
                
                if scenario == "Rainy Day" or scenario == "Foggy Morning":
                    self.effects_text.insert(tk.END, "- Reduced visibility and speed\n")
                    self.effects_text.insert(tk.END, f"- Max speed: {self.scenario_manager.scenarios[scenario].get('max_speed', 'N/A')} m/s\n")
                
                if scenario == "Rush Hour":
                    self.effects_text.insert(tk.END, "- High congestion at intersections\n")
                    self.effects_text.insert(tk.END, "- Longer waiting times\n")
                
                if scenario == "Emergency":
                    self.effects_text.insert(tk.END, "- Emergency vehicles have priority\n")
                    self.effects_text.insert(tk.END, "- Traffic signals may adapt to emergency vehicles\n")

    def update_rsu_tree(self):
        """Update the RSU data table"""
        # Clear the current data
        for item in self.rsu_tree.get_children():
            self.rsu_tree.delete(item)
        
        # Add data for each RSU
        for rsu in self.rsus:
            # Calculate average speed of vehicles in range
            avg_speed = 0
            if rsu.vehicles_in_range:
                speeds = [data["speed"] for data in rsu.vehicle_data.values() if "speed" in data]
                avg_speed = sum(speeds) / len(speeds) if speeds else 0
            
            # Get recommended phase and duration
            recommended_phase, duration = rsu.get_recommended_phase()
            
            # Add to tree
            self.rsu_tree.insert("", "end", values=(
                rsu.id,
                f"({rsu.position[0]:.1f}, {rsu.position[1]:.1f})",
                len(rsu.vehicles_in_range),
                rsu.congestion_level.upper(),
                f"{avg_speed:.1f} m/s",
                f"{recommended_phase.upper()} for {duration}s"
            ))

    def open_cluster_window(self, cluster):
        """Open a control window for a traffic light cluster"""
        if cluster.name in self.cluster_windows:
            self.cluster_windows[cluster.name].window.lift()
        else:
            window = ClusterWindow(cluster, self.root)
            self.cluster_windows[cluster.name] = window

    def set_all_lights(self, color):
        """Set all traffic lights to the specified color"""
        for cluster in self.clusters.values():
            phase = "red" if color == "r" else "yellow" if color == "y" else "green"
            cluster.set_phase(phase)

    def apply_selected_scenario(self):
        """Apply the currently selected scenario"""
        scenario = self.selected_scenario.get()
        intensity = self.intensity_slider.get()
        self.scenario_manager.apply_scenario(scenario, intensity)
        messagebox.showinfo("Scenario Applied", f"Applied {scenario} scenario with intensity {intensity}")

    def apply_custom_scenario(self):
        """Apply a custom scenario"""
        try:
            vehicle_count = self.vehicle_count_var.get()
            max_speed = self.speed_var.get()
            vehicle_type = self.vehicle_type_var.get().lower()
            
            # Create custom vehicle type if needed
            custom_type = f"custom_{vehicle_type}_vehicle"
            try:
                traci.vehicletype.copy("veh_passenger", custom_type)
                
                # Set color based on type
                if vehicle_type == "emergency":
                    color = (255, 0, 0, 255)  # Red
                elif vehicle_type == "heavy":
                    color = (0, 0, 128, 255)  # Dark blue
                elif vehicle_type == "light":
                    color = (0, 255, 255, 255)  # Cyan
                else:
                    color = (255, 165, 0, 255)  # Orange for custom
                
                traci.vehicletype.setColor(custom_type, color)
                traci.vehicletype.setMaxSpeed(custom_type, max_speed)
            except traci.TraCIException as e:
                print(f"Error setting up custom vehicle type: {e}")
            
            # Add vehicles
            trip_edges = self.scenario_manager.load_trip_edges()
            for i in range(vehicle_count):
                vehicle_id = f"custom_{int(traci.simulation.getTime())}_{i}"
                try:
                    from_edge, to_edge = random.choice(trip_edges)
                    route_id = f"route_{vehicle_id}"
                    traci.route.add(routeID=route_id, edges=[from_edge, to_edge])
                    traci.vehicle.add(vehID=vehicle_id, routeID=route_id, typeID=custom_type,
                                    departLane="best", departSpeed="max")
                except traci.TraCIException as e:
                    print(f"Error adding custom vehicle {vehicle_id}: {e}")
            
            messagebox.showinfo("Custom Scenario", 
                               f"Added {vehicle_count} {vehicle_type} vehicles with max speed {max_speed} m/s")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply custom scenario: {e}")

    def insert_vehicles(self):
        """Insert vehicles into the simulation"""
        num_vehicles = simpledialog.askinteger("Input", "Enter number of vehicles:", parent=self.root)
        if num_vehicles is None or num_vehicles < 1:
            return
            
        trip_edges = self.scenario_manager.load_trip_edges()
        if not trip_edges:
            messagebox.showerror("Error", "No valid routes found")
            return

        for i in range(num_vehicles):
            vehicle_id = f"vehicle_{int(traci.simulation.getTime())}_{i}"
            try:
                from_edge, to_edge = random.choice(trip_edges)
                route_id = f"route_{vehicle_id}"
                traci.route.add(routeID=route_id, edges=[from_edge, to_edge])
                traci.vehicle.add(vehID=vehicle_id, routeID=route_id, typeID="veh_passenger",
                                departLane="best", departSpeed="max")
                print(f"Added vehicle {vehicle_id}")
            except traci.TraCIException as e:
                print(f"Error adding vehicle {vehicle_id}: {e}")

    def toggle_simulation(self):
        """Toggle the simulation between running and paused"""
        self.running = not self.running
        self.play_button.config(text="Pause" if self.running else "Play")

    def stop_simulation(self):
        """Stop the simulation and cleanup"""
        self.exit_simulation = True
        
        # Close all cluster windows
        for window in self.cluster_windows.values():
            window.close()
        
        # Stop database worker
        self.db_manager.stop()
        
        # Close SUMO
        try:
            traci.close()
        except:
            pass
        
        self.root.quit()

    def run_timer(self):
        """Run the timer thread for traffic light cycles"""
        last_rsu_update = 0
        
        while not self.exit_simulation:
            if self.running:
                current_time = time.time()
                
                # Update RSUs periodically
                if current_time - last_rsu_update >= self.refresh_rate.get():
                    self.update_rsu_tree()
                    last_rsu_update = current_time
                
                # Update clusters
                for cluster in self.clusters.values():
                    if cluster.automatic_mode:
                        if cluster.time_remaining <= 0:
                            if cluster.current_phase == "red":
                                cluster.current_phase = "green"
                                cluster.time_remaining = cluster.green_time
                                cluster.set_phase("green")
                            elif cluster.current_phase == "green":
                                cluster.current_phase = "yellow"
                                cluster.time_remaining = cluster.yellow_time
                                cluster.set_phase("yellow")
                            else:  # yellow
                                cluster.current_phase = "red"
                                cluster.time_remaining = cluster.red_time
                                cluster.set_phase("red")
                        else:
                            cluster.time_remaining -= 1
            
            time.sleep(1)

    def run_simulation(self):
        """Run the main simulation loop"""
        step = 0
        last_stats_update = 0
        
        while not self.exit_simulation:
            if self.running:
                try:
                    # Advance simulation
                    traci.simulationStep()
                    
                    # Update vehicle count and other stats
                    if step % 10 == 0:  # Every 10 steps (approx. 1 second in simulation time)
                        # Update cluster vehicle counts
                        for cluster in self.clusters.values():
                            cluster.update_vehicle_count()
                        
                        # Log to database
                        self.db_manager.log_simulation_state("mixed")  # Actual phase is per cluster
                    
                    # Update RSUs
                    if step % 20 == 0:  # Every 2 seconds
                        for rsu in self.rsus:
                            rsu.update()
                        self.db_manager.log_rsu_data(self.rsus)
                    
                    # Update GUI
                    if step % 50 == 0:  # Every 5 seconds
                        # Update cluster windows
                        for window in self.cluster_windows.values():
                            window.update()
                        
                        # Update stats text every 5 seconds of simulation time
                        current_time = time.time()
                        if current_time - last_stats_update >= 5:
                            self.update_stats_text()
                            last_stats_update = current_time
                    
                    step += 1
                    time.sleep(0.1)  # Slow down simulation for visibility
                
                except traci.TraCIException as e:
                    print(f"Simulation error: {e}")
                    if "connection closed by SUMO" in str(e):
                        break
            
            time.sleep(0.1)  # Prevent high CPU usage when paused

        print("Simulation ended")

    def update_stats_text(self):
        """Update the simulation statistics text"""
        try:
            vehicle_count = len(traci.vehicle.getIDList())
            stopped_count = sum(1 for v in traci.vehicle.getIDList() 
                               if traci.vehicle.getSpeed(v) < 0.1)
            
            avg_speed = 0
            avg_waiting = 0
            
            if vehicle_count > 0:
                speeds = [traci.vehicle.getSpeed(v) for v in traci.vehicle.getIDList()]
                waiting_times = [traci.vehicle.getWaitingTime(v) for v in traci.vehicle.getIDList()]
                
                avg_speed = sum(speeds) / len(speeds)
                avg_waiting = sum(waiting_times) / len(waiting_times)
            
            # Get edges with traffic
            edge_counts = {}
            for veh in traci.vehicle.getIDList():
                edge = traci.vehicle.getRoadID(veh)
                if edge in edge_counts:
                    edge_counts[edge] += 1
                else:
                    edge_counts[edge] = 1
            
            # Find congested edges (with more than 3 vehicles)
            congested_edges = {edge: count for edge, count in edge_counts.items() if count > 3}
            
            # Clear and update text
            self.stats_text.delete(1.0, tk.END)
            
            sim_time = traci.simulation.getTime()
            self.stats_text.insert(tk.END, f"Simulation Time: {sim_time:.1f}s\n\n")
            self.stats_text.insert(tk.END, f"Active Vehicles: {vehicle_count}\n")
            self.stats_text.insert(tk.END, f"Stopped Vehicles: {stopped_count}\n")
            self.stats_text.insert(tk.END, f"Average Speed: {avg_speed:.2f} m/s\n")
            self.stats_text.insert(tk.END, f"Average Waiting Time: {avg_waiting:.2f}s\n\n")
            
            self.stats_text.insert(tk.END, f"Congested Edges ({len(congested_edges)}):\n")
            for edge, count in sorted(congested_edges.items(), key=lambda x: x[1], reverse=True)[:5]:
                self.stats_text.insert(tk.END, f"- {edge}: {count} vehicles\n")
            
            self.stats_text.insert(tk.END, "\nTraffic Light States:\n")
            for tl_id in self.tl_ids[:5]:  # Show max 5 traffic lights to avoid clutter
                state = traci.trafficlight.getRedYellowGreenState(tl_id)
                phase = traci.trafficlight.getPhase(tl_id)
                self.stats_text.insert(tk.END, f"- {tl_id}: Phase {phase}, State: {state}\n")
            
            self.stats_text.see(1.0)  # Scroll to top
            
        except traci.TraCIException as e:
            print(f"Error updating stats: {e}")
            self.stats_text.insert(tk.END, f"Error updating statistics: {e}\n")

def main():
    """Main function to start the application"""
    try:
        app = TrafficSimulationGUI()
        app.root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()