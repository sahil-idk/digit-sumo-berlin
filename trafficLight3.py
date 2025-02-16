import traci
import tkinter as tk
from tkinter import ttk, simpledialog
import threading
import time
import random
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from xml.etree import ElementTree as ET

class TrafficCluster:
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

class ClusterWindow:
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

class TrafficSimulationGUI:
    def __init__(self):
        self.sumoBinary = "sumo-gui"
        self.sumoCmd = [self.sumoBinary, "-c", "osm.sumocfg"]
        traci.start(self.sumoCmd)
        
        self.tl_ids = traci.trafficlight.getIDList()
        print(f"Available traffic lights: {self.tl_ids}")
        
        self.root = tk.Tk()
        self.root.title("Traffic Control System")
        self.root.geometry("400x300")
        
        traci.vehicletype.copy("veh_passenger", "red_passenger")
        traci.vehicletype.setColor("red_passenger", (255, 0, 0, 255))
        
        self.create_clusters()
        self.cluster_windows = {}
        self.running = False
        self.exit_simulation = False
        
        self.setup_gui()
        
        self.simulation_thread = threading.Thread(target=self.run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        self.timer_thread = threading.Thread(target=self.run_timer)
        self.timer_thread.daemon = True
        self.timer_thread.start()

    def create_clusters(self):
        self.clusters = {}
        for i, tl_id in enumerate(self.tl_ids):
            try:
                x, y = traci.junction.getPosition(tl_id)
                cluster_id = "cluster1" if i == 0 else f"cluster{i+1}"
                cluster_name = f"Traffic Light Cluster {i+1}"
                
                if cluster_id not in self.clusters:
                    self.clusters[cluster_id] = TrafficCluster(cluster_name, [])
                self.clusters[cluster_id].traffic_lights.append(tl_id)
                print(f"Added traffic light {tl_id} to {cluster_name}")
            except traci.TraCIException as e:
                print(f"Error getting position for traffic light {tl_id}: {e}")
                continue
        
        self.clusters = {k: v for k, v in self.clusters.items() if v.traffic_lights}

    def setup_gui(self):
        control_frame = ttk.LabelFrame(self.root, text="Simulation Controls", padding="10")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.play_button = ttk.Button(control_frame, text="Play", command=self.toggle_simulation)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Add Vehicles", 
                  command=self.insert_vehicles).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop Simulation", 
                  command=self.stop_simulation).pack(side=tk.LEFT, padx=5)
        
        clusters_frame = ttk.LabelFrame(self.root, text="Cluster Controls", padding="10")
        clusters_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        for cluster_id, cluster in self.clusters.items():
            ttk.Button(clusters_frame, 
                      text=f"Open {cluster.name}", 
                      command=lambda c=cluster: self.open_cluster_window(c)).pack(pady=2)

    def open_cluster_window(self, cluster):
        if cluster.name in self.cluster_windows:
            self.cluster_windows[cluster.name].window.lift()
        else:
            window = ClusterWindow(cluster, self.root)
            self.cluster_windows[cluster.name] = window

    def insert_vehicles(self):
        num_vehicles = simpledialog.askinteger("Input", "Enter number of vehicles:", parent=self.root)
        if num_vehicles is None or num_vehicles < 1:
            return
            
        trip_edges = self.load_trip_edges()
        if not trip_edges:
            print("No valid routes found")
            return

        for i in range(num_vehicles):
            vehicle_id = f"vehicle_{int(traci.simulation.getTime())}_{i}"
            try:
                from_edge, to_edge = random.choice(trip_edges)
                route_id = f"route_{vehicle_id}"
                traci.route.add(routeID=route_id, edges=[from_edge, to_edge])
                traci.vehicle.add(vehID=vehicle_id, routeID=route_id, typeID="red_passenger",
                                departLane="best", departSpeed="max")
                print(f"Added vehicle {vehicle_id}")
            except traci.TraCIException as e:
                print(f"Error adding vehicle {vehicle_id}: {e}")

    def load_trip_edges(self):
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
            return []

    def update_gui(self):
        for window in self.cluster_windows.values():
            window.update()

    def toggle_simulation(self):
        self.running = not self.running
        self.play_button.config(text="Pause" if self.running else "Play")

    def stop_simulation(self):
        self.exit_simulation = True
        for window in self.cluster_windows.values():
            window.close()
        self.root.quit()

    def run_timer(self):
        while not self.exit_simulation:
            if self.running:
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
        step = 0
        while not self.exit_simulation:
            if self.running:
                try:
                    traci.simulationStep()
                    
                    # Update clusters
                    for cluster in self.clusters.values():
                        cluster.update_vehicle_count()
                    
                    # Update GUI every 10 steps
                    if step % 10 == 0:
                        self.update_gui()
                    
                    step += 1
                    time.sleep(0.1)
                
                except traci.TraCIException as e:
                    print(f"Simulation error: {e}")
                    break
            
            time.sleep(0.1)

        try:
            traci.close()
        except:
            pass
        print("Simulation ended")

    def start(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = TrafficSimulationGUI()
    app.start()