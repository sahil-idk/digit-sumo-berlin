import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, colorchooser
import threading
import time
import random
import math
import os
import sys
import traci
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime

# Set up custom styles
COLORS = {
    "primary": "#3498db",       # Blue
    "secondary": "#2ecc71",     # Green
    "warning": "#f39c12",       # Orange
    "danger": "#e74c3c",        # Red
    "dark": "#2c3e50",          # Dark blue/gray
    "light": "#ecf0f1",         # Light gray
    "text": "#34495e",          # Dark gray for text
    "background": "#f9f9f9"     # Off-white background
}

class StylishButton(tk.Button):
    """Custom styled button"""
    def __init__(self, master=None, color="primary", **kwargs):
        self.color_theme = COLORS[color]
        kwargs.update({
            "background": self.color_theme,
            "foreground": "white",
            "relief": "flat",
            "font": ("Segoe UI", 10),
            "activebackground": self._adjust_brightness(self.color_theme, -20),
            "activeforeground": "white",
            "borderwidth": 0,
            "padx": 15,
            "pady": 8,
            "cursor": "hand2"
        })
        super().__init__(master, **kwargs)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        
    def _adjust_brightness(self, hex_color, factor):
        """Adjust color brightness"""
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        
        r = max(0, min(255, r + factor))
        g = max(0, min(255, g + factor))
        b = max(0, min(255, b + factor))
        
        return f"#{r:02x}{g:02x}{b:02x}"
        
    def _on_enter(self, event):
        """Mouse enter effect"""
        self.config(background=self._adjust_brightness(self.color_theme, -10))
        
    def _on_leave(self, event):
        """Mouse leave effect"""
        self.config(background=self.color_theme)

class DashboardPanel(tk.Frame):
    """Base panel for dashboard elements"""
    def __init__(self, master=None, title="Panel", **kwargs):
        kwargs.update({
            "background": "white",
            "padx": 10,
            "pady": 10,
            "relief": "flat",
            "highlightthickness": 1,
            "highlightbackground": "#ddd"
        })
        super().__init__(master, **kwargs)
        
        # Header
        self.header = tk.Frame(self, bg="white")
        self.header.pack(fill="x", pady=(0, 10))
        
        self.title_label = tk.Label(
            self.header, 
            text=title, 
            font=("Segoe UI", 12, "bold"),
            bg="white",
            fg=COLORS["text"]
        )
        self.title_label.pack(side="left")
        
        # Content area
        self.content = tk.Frame(self, bg="white")
        self.content.pack(fill="both", expand=True)

class RSU:
    """Road Side Unit implementation for collecting and analyzing traffic data"""
    def __init__(self, id, position, radius=100):
        self.id = id
        self.position = position  # (x, y) coordinates
        self.radius = radius      # detection radius in meters
        self.vehicles_in_range = set()
        self.vehicle_data = {}    # Store vehicle data for analysis
        self.congestion_level = "low"
        self.poi_id = f"RSU_POI_{id.replace('RSU_', '')}"  # Remove RSU_ prefix if it exists
        self.range_poi_id = f"RSU_Range_{id.replace('RSU_', '')}"
        self.history = {
            "timestamps": [],
            "vehicle_counts": [],
            "avg_speeds": []
        }
    
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
                except traci.TraCIException:
                    # Vehicle might have left the simulation
                    if veh_id in self.vehicle_data:
                        del self.vehicle_data[veh_id]
                    continue
                    
            # Update congestion level based on number of vehicles
            self.update_congestion_level()
            
            # Update the POI color based on congestion level
            self.update_poi_appearance()
            
            # Record history data
            self.record_history()
        
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
            
    def update_poi_appearance(self):
        """Update the POI appearance (color/size) based on congestion level"""
        try:
            # Change color based on congestion level
            if self.congestion_level == "low":
                color = (0, 255, 0, 255)  # Green
                range_color = (0, 255, 0, 60)  # Semi-transparent green
            elif self.congestion_level == "medium":
                color = (255, 255, 0, 255)  # Yellow
                range_color = (255, 255, 0, 60)  # Semi-transparent yellow
            else:  # high
                color = (255, 0, 0, 255)  # Red
                range_color = (255, 0, 0, 60)  # Semi-transparent red
            
            # Update POI color and size with error handling
            try:
                # Update POI color
                traci.poi.setColor(self.poi_id, color)
                
                # Make the POI "blink" by changing its size - it will appear to pulse
                current_step = traci.simulation.getTime()
                if int(current_step) % 2 == 0:  # Every even second
                    size = 25  # Larger size
                else:
                    size = 15  # Normal size
                    
                traci.poi.setWidth(self.poi_id, size)
            except traci.TraCIException:
                # If the POI doesn't exist, try to create it
                try:
                    traci.poi.add(self.poi_id, self.position[0], self.position[1], 
                                 color, "RSU", 20, 1)
                except:
                    pass  # Silently ignore if we can't create it
            
            # Update range polygon color with error handling
            try:
                traci.polygon.setColor(self.range_poi_id, range_color)
            except:
                pass  # Silently ignore if the polygon doesn't exist
            
        except Exception as e:
            print(f"Error updating POI appearance for {self.id}: {e}")
            
    def record_history(self):
        """Record historical data for this RSU"""
        current_time = datetime.now().strftime("%H:%M:%S")
        vehicle_count = len(self.vehicles_in_range)
        
        # Calculate average speed
        avg_speed = 0
        if vehicle_count > 0:
            speeds = [data["speed"] for data in self.vehicle_data.values() if "speed" in data]
            avg_speed = sum(speeds) / len(speeds) if speeds else 0
        
        # Add to history (maintaining last 30 data points)
        self.history["timestamps"].append(current_time)
        self.history["vehicle_counts"].append(vehicle_count)
        self.history["avg_speeds"].append(avg_speed)
        
        # Keep only the last 30 points
        if len(self.history["timestamps"]) > 30:
            self.history["timestamps"] = self.history["timestamps"][-30:]
            self.history["vehicle_counts"] = self.history["vehicle_counts"][-30:]
            self.history["avg_speeds"] = self.history["avg_speeds"][-30:]
    
    def get_recommended_phase(self):
        """Get recommended traffic light phase based on congestion level"""
        if self.congestion_level == "high":
            return "green", 45  # Longer green time for high congestion
        elif self.congestion_level == "medium":
            return "green", 30  # Standard green time for medium congestion
        else:
            return "green", 20  # Shorter green time for low congestion

class ScenarioManager:
    """Manages different traffic scenarios for what-if analysis"""
    def __init__(self):
        # Define scenarios with unique vehicle types and colors
        self.scenarios = {
            "Normal Traffic": {"density": 1, "vehicle_type": "normal_vehicle", "color": (0, 200, 0, 255)},  # Green
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
        """Load trip edges from network or use defaults"""
        try:
            # First try to get edges from the network
            all_edges = traci.edge.getIDList()
            regular_edges = [e for e in all_edges if not e.startswith(":")]  # Filter out internal edges
            
            # Create random pairs for source/destination
            trip_edges = []
            for _ in range(10):  # Create 10 sample trips
                if len(regular_edges) >= 2:
                    from_edge = random.choice(regular_edges)
                    to_edge = random.choice([e for e in regular_edges if e != from_edge])
                    trip_edges.append((from_edge, to_edge))
            
            if trip_edges:
                return trip_edges
                
        except Exception as e:
            print(f"Error loading edges: {e}")
        
        # Fallback to hard-coded trip edges
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

class ModernTrafficGUI:
    """Modern GUI for the traffic simulation application"""
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
        self.root.title("Smart Traffic Control System")
        self.root.geometry("1280x800")
        self.root.configure(bg=COLORS["background"])
        self.root.minsize(1024, 768)
        
        # Use a more modern theme if available
        try:
            self.style = ttk.Style()
            if "clam" in self.style.theme_names():
                self.style.theme_use("clam")
                
            # Configure styles
            self.configure_styles()
        except Exception as e:
            print(f"Could not set theme: {e}")
        
        # Control variables
        self.running = False
        self.exit_simulation = False
        
        # Initialize managers
        self.tl_ids = traci.trafficlight.getIDList()
        self.scenario_manager = ScenarioManager()
        
        # Create RSUs at traffic light junctions
        self.rsus = []
        self.create_rsus()
        
        # Setup GUI components
        self.setup_gui()
        
        # Start simulation threads
        self.simulation_thread = threading.Thread(target=self.run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        # Configure SUMO GUI
        self.configure_sumo_gui()
    
    def configure_styles(self):
        """Configure ttk styles for a modern look"""
        # General styles
        self.style.configure("TFrame", background=COLORS["background"])
        self.style.configure("TLabel", background=COLORS["background"], foreground=COLORS["text"])
        self.style.configure("TButton", padding=6, relief="flat", background=COLORS["primary"])
        
        # TreeView styling
        self.style.configure(
            "Treeview", 
            background="white", 
            foreground=COLORS["text"], 
            rowheight=30,
            fieldbackground="white"
        )
        self.style.configure("Treeview.Heading", 
                            font=('Segoe UI', 10, 'bold'),
                            background=COLORS["light"],
                            foreground=COLORS["dark"])
        
        # Specific widgets
        self.style.configure("TNotebook", background=COLORS["background"], tabmargins=[0, 0, 0, 0])
        self.style.configure("TNotebook.Tab", background=COLORS["light"], padding=[15, 5], font=('Segoe UI', 10))
        self.style.map("TNotebook.Tab", 
                      background=[("selected", COLORS["primary"])],
                      foreground=[("selected", "white")])
    
    def create_rsus(self):
        """Create RSUs at traffic light junctions with visual representation"""
        for i, tl_id in enumerate(self.tl_ids):
            try:
                junction_pos = traci.junction.getPosition(tl_id)
                rsu = RSU(tl_id, junction_pos)  # Use the traffic light ID directly
                self.rsus.append(rsu)
                
                # Add visual representation to the SUMO GUI map
                try:
                    # Add a POI at the RSU location
                    traci.poi.add(rsu.poi_id, junction_pos[0], junction_pos[1], 
                                 (255, 0, 0, 255), "RSU", 20, 1)
                    
                    # Add a circle to show detection range
                    traci.polygon.add(rsu.range_poi_id, 
                                     self.create_circle_points(junction_pos, rsu.radius), 
                                     (0, 0, 255, 80), fill=True, layer=-1)
                    
                    print(f"Created RSU at junction {tl_id} with visual POI")
                except traci.TraCIException as e:
                    print(f"Error adding RSU POI for {tl_id}: {e}")
                    
            except traci.TraCIException as e:
                print(f"Error creating RSU at junction {tl_id}: {e}")
    
    def create_circle_points(self, center, radius, num_points=36):
        """Create a circle of points around the center with the given radius"""
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
        return points
    
    def configure_sumo_gui(self):
        """Configure the SUMO GUI view"""
        try:
            # Set GUI schema
            traci.gui.setSchema("View #0", "real world")
            
            # Set the boundary to show the entire network
            net_boundary = traci.simulation.getNetBoundary()
            traci.gui.setBoundary("View #0", 
                                 net_boundary[0][0], net_boundary[0][1],
                                 net_boundary[1][0], net_boundary[1][1])
            
            # Zoom out slightly to see everything
            traci.gui.setZoom("View #0", 800)
        except traci.TraCIException as e:
            print(f"Warning: Could not configure SUMO GUI: {e}")
    
    def setup_gui(self):
        """Setup the main GUI components with modern styling"""
        # Top header/toolbar
        self.setup_header()
        
        # Main content area with tabs
        self.setup_tabs()
        
        # Status bar at bottom
        self.setup_status_bar()
    
    def setup_header(self):
        """Setup the header toolbar"""
        header = tk.Frame(self.root, bg=COLORS["dark"], height=60)
        header.pack(fill="x")
        header.pack_propagate(False)  # Don't shrink
        
        # App logo/title
        title_frame = tk.Frame(header, bg=COLORS["dark"])
        title_frame.pack(side="left", padx=15)
        
        title = tk.Label(
            title_frame, 
            text="SMART TRAFFIC", 
            fg="white", 
            bg=COLORS["dark"],
            font=("Segoe UI", 16, "bold")
        )
        title.pack(side="left")
        
        subtitle = tk.Label(
            title_frame, 
            text="Control System", 
            fg="#aaa", 
            bg=COLORS["dark"],
            font=("Segoe UI", 10)
        )
        subtitle.pack(side="left", padx=10)
        
        # Control buttons
        controls = tk.Frame(header, bg=COLORS["dark"])
        controls.pack(side="right", padx=15)
        
        self.play_btn = StylishButton(
            controls, 
            text="▶ Play", 
            color="secondary",
            command=self.toggle_simulation
        )
        self.play_btn.pack(side="left", padx=5)
        
        StylishButton(
            controls, 
            text="+ Add Vehicles", 
            color="primary",
            command=self.insert_vehicles
        ).pack(side="left", padx=5)
        
        StylishButton(
            controls, 
            text="✕ Stop", 
            color="danger",
            command=self.stop_simulation
        ).pack(side="left", padx=5)
    
    def setup_tabs(self):
        """Setup the tabbed interface"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tab 1: Dashboard
        self.dashboard_tab = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(self.dashboard_tab, text="Dashboard")
        self.setup_dashboard(self.dashboard_tab)
        
        # Tab 2: RSU Monitoring
        self.rsu_tab = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(self.rsu_tab, text="RSU Monitoring")
        self.setup_rsu_monitoring(self.rsu_tab)
        
        # Tab 3: Traffic Scenarios
        self.scenario_tab = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(self.scenario_tab, text="Traffic Scenarios")
        self.setup_scenario_tab(self.scenario_tab)
        
        # Tab 4: Analytics
        self.analytics_tab = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(self.analytics_tab, text="Analytics")
        self.setup_analytics_tab(self.analytics_tab)
    
    def setup_dashboard(self, parent):
        """Setup the dashboard tab"""
        # Create a grid layout
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=0)  # Stats row
        parent.rowconfigure(1, weight=1)  # Charts/map row
        
        # Statistics panels - top row
        stats_frame = tk.Frame(parent, bg=COLORS["background"])
        stats_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        # Create 4 stat panels
        for i, (title, icon, value, unit) in enumerate([
            ("Active Vehicles", "🚗", "0", "vehicles"),
            ("Average Speed", "⚡", "0", "m/s"),
            ("Congestion Level", "🚦", "Low", ""),
            ("Simulation Time", "⏱️", "0:00", "")
        ]):
            stat_panel = self.create_stat_panel(stats_frame, title, icon, value, unit)
            stat_panel.pack(side="left", fill="both", expand=True, padx=5)
            
            # Save references to update these values
            if title == "Active Vehicles":
                self.vehicle_count_label = stat_panel.winfo_children()[1].winfo_children()[0]
            elif title == "Average Speed":
                self.avg_speed_label = stat_panel.winfo_children()[1].winfo_children()[0]
            elif title == "Congestion Level":
                self.congestion_label = stat_panel.winfo_children()[1].winfo_children()[0]
            elif title == "Simulation Time":
                self.sim_time_label = stat_panel.winfo_children()[1].winfo_children()[0]
        
        # Left panel - Vehicle Count Chart
        self.chart_panel = DashboardPanel(parent, title="Traffic Flow")
        self.chart_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        
        # Setup matplotlib figure for the chart
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor='white')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('white')
        self.ax.set_title('Vehicle Count Over Time')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Vehicles')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_panel.content)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Right panel - RSU Status
        self.rsu_panel = DashboardPanel(parent, title="RSU Status")
        self.rsu_panel.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        
        # Create treeview for RSU data
        columns = ("RSU", "Vehicles", "Status", "Recommendation")
        self.dashboard_tree = ttk.Treeview(
            self.rsu_panel.content, 
            columns=columns, 
            show="headings", 
            style="Treeview"
        )
        
        # Configure columns
        for col in columns:
            self.dashboard_tree.heading(col, text=col)
            width = 100 if col == "Recommendation" else 80
            self.dashboard_tree.column(col, width=width)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(self.rsu_panel.content, orient="vertical", command=self.dashboard_tree.yview)
        self.dashboard_tree.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        self.dashboard_tree.pack(fill="both", expand=True)
    
    def create_stat_panel(self, parent, title, icon, value, unit):
        """Create a stylish statistics panel"""
        panel = tk.Frame(parent, bg="white", padx=15, pady=15)
        panel.configure(highlightbackground="#ddd", highlightthickness=1)
        
        # Title
        tk.Label(
            panel, 
            text=title, 
            font=("Segoe UI", 10),
            fg="#777",
            bg="white"
        ).pack(anchor="w")
        
        # Value with icon
        value_frame = tk.Frame(panel, bg="white")
        value_frame.pack(fill="x", pady=5)
        
        value_label = tk.Label(
            value_frame, 
            text=value, 
            font=("Segoe UI", 22, "bold"),
            fg=COLORS["text"],
            bg="white"
        )
        value_label.pack(side="left")
        
        tk.Label(
            value_frame, 
            text=f" {unit}", 
            font=("Segoe UI", 10),
            fg="#777",
            bg="white"
        ).pack(side="left", padx=2, pady=(8, 0))
        
        # Icon on the right
        tk.Label(
            value_frame, 
            text=icon, 
            font=("Segoe UI", 22),
            fg=COLORS["primary"],
            bg="white"
        ).pack(side="right")
        
        return panel
    
    def setup_rsu_monitoring(self, parent):
        """Setup the RSU monitoring tab"""
        # Split into two panels
        parent.columnconfigure(0, weight=2)
        parent.columnconfigure(1, weight=3)
        parent.rowconfigure(0, weight=1)
        
        # Left panel - RSU list
        left_panel = DashboardPanel(parent, title="RSU Network")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)
        
        # Create RSU treeview with more details
        columns = ("ID", "Location", "Vehicles", "Congestion", "Avg Speed")
        self.rsu_tree = ttk.Treeview(
            left_panel.content, 
            columns=columns, 
            show="headings", 
            style="Treeview"
        )
        
        # Configure columns
        for col in columns:
            self.rsu_tree.heading(col, text=col)
            width = 100 if col == "Location" else 80
            self.rsu_tree.column(col, width=width)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(left_panel.content, orient="vertical", command=self.rsu_tree.yview)
        self.rsu_tree.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        self.rsu_tree.pack(fill="both", expand=True)
        
        # Bind selection event
        self.rsu_tree.bind("<<TreeviewSelect>>", self.on_rsu_selected)
        
        # Right panel - RSU details
        self.right_panel = DashboardPanel(parent, title="RSU Details")
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)
        
        # RSU details container
        self.details_container = tk.Frame(self.right_panel.content, bg="white")
        self.details_container.pack(fill="both", expand=True)
        
        # Create the initial "select an RSU" message
        tk.Label(
            self.details_container,
            text="Select an RSU from the list to view details",
            font=("Segoe UI", 12),
            fg="#777",
            bg="white"
        ).pack(expand=True)
        
        # We'll dynamically create the details when an RSU is selected
        self.current_rsu_id = None
        self.rsu_chart = None
    
    def on_rsu_selected(self, event):
        """Handle RSU selection from the treeview"""
        selection = self.rsu_tree.selection()
        if not selection:
            return
            
        # Get the selected RSU
        item = self.rsu_tree.item(selection[0])
        rsu_id = item['values'][0]
        
        # Don't redraw if it's the same RSU
        if rsu_id == self.current_rsu_id:
            return
            
        self.current_rsu_id = rsu_id
        self.update_rsu_details(rsu_id)
    
    def update_rsu_details(self, rsu_id):
        """Update the RSU details panel with information about the selected RSU"""
        # Find the RSU object
        selected_rsu = None
        for rsu in self.rsus:
            if rsu.id == rsu_id:
                selected_rsu = rsu
                break
                
        if not selected_rsu:
            return
            
        # Clear current contents
        for widget in self.details_container.winfo_children():
            widget.destroy()
            
        # Create new details view
        details_frame = tk.Frame(self.details_container, bg="white")
        details_frame.pack(fill="both", expand=True)
        
        # Header with RSU ID and status
        header = tk.Frame(details_frame, bg="white", pady=10)
        header.pack(fill="x")
        
        # RSU Icon and ID
        tk.Label(
            header,
            text="📡",
            font=("Segoe UI", 24),
            fg=COLORS["primary"],
            bg="white"
        ).pack(side="left", padx=(0, 10))
        
        id_frame = tk.Frame(header, bg="white")
        id_frame.pack(side="left")
        
        tk.Label(
            id_frame,
            text=rsu_id,
            font=("Segoe UI", 14, "bold"),
            fg=COLORS["text"],
            bg="white"
        ).pack(anchor="w")
        
        tk.Label(
            id_frame,
            text=f"Location: ({selected_rsu.position[0]:.1f}, {selected_rsu.position[1]:.1f})",
            font=("Segoe UI", 10),
            fg="#777",
            bg="white"
        ).pack(anchor="w")
        
        # Status indicator
        status_color = COLORS["secondary"] if selected_rsu.congestion_level == "low" else \
                      COLORS["warning"] if selected_rsu.congestion_level == "medium" else \
                      COLORS["danger"]
                      
        status_frame = tk.Frame(header, bg="white")
        status_frame.pack(side="right")
        
        status_label = tk.Label(
            status_frame,
            text=f"Status: {selected_rsu.congestion_level.upper()}",
            font=("Segoe UI", 11, "bold"),
            fg=status_color,
            bg="white"
        )
        status_label.pack(anchor="e")
        
        vehicles_label = tk.Label(
            status_frame,
            text=f"Vehicles: {len(selected_rsu.vehicles_in_range)}",
            font=("Segoe UI", 10),
            fg="#777",
            bg="white"
        )
        vehicles_label.pack(anchor="e")
        
        # Divider
        ttk.Separator(details_frame, orient="horizontal").pack(fill="x", pady=10)
        
        # Details in two columns
        details = tk.Frame(details_frame, bg="white")
        details.pack(fill="x", pady=10)
        details.columnconfigure(0, weight=1)
        details.columnconfigure(1, weight=1)
        
        # Left column - Stats
        stats_frame = tk.LabelFrame(details, text="Current Statistics", bg="white", fg=COLORS["text"])
        stats_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # Calculate avg speed
        avg_speed = 0
        if selected_rsu.vehicles_in_range:
            speeds = [data["speed"] for data in selected_rsu.vehicle_data.values() if "speed" in data]
            avg_speed = sum(speeds) / len(speeds) if speeds else 0
            
        # Get recommendation
        rec_phase, rec_duration = selected_rsu.get_recommended_phase()
        
        # Stats
        tk.Label(stats_frame, text=f"Vehicle Count: {len(selected_rsu.vehicles_in_range)}", anchor="w", 
                bg="white", fg=COLORS["text"], pady=3).pack(fill="x")
        tk.Label(stats_frame, text=f"Average Speed: {avg_speed:.1f} m/s", anchor="w", 
                bg="white", fg=COLORS["text"], pady=3).pack(fill="x")
        tk.Label(stats_frame, text=f"Congestion Level: {selected_rsu.congestion_level.upper()}", anchor="w", 
                bg="white", fg=COLORS["text"], pady=3).pack(fill="x")
        tk.Label(stats_frame, text=f"Recommendation: {rec_phase.upper()} for {rec_duration}s", anchor="w", 
                bg="white", fg=COLORS["text"], pady=3).pack(fill="x")
        
        # Right column - Actions
        actions_frame = tk.LabelFrame(details, text="Actions", bg="white", fg=COLORS["text"])
        actions_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        # Manual lighting control
        tk.Label(actions_frame, text="Control Traffic Lights", bg="white", fg=COLORS["text"], pady=3).pack(anchor="w")
        
        light_btns = tk.Frame(actions_frame, bg="white")
        light_btns.pack(fill="x", pady=5)
        
        StylishButton(light_btns, text="RED", color="danger", width=7,
                     command=lambda: self.set_light_phase(selected_rsu.id, "red")).pack(side="left", padx=5)
        StylishButton(light_btns, text="YELLOW", color="warning", width=7,
                     command=lambda: self.set_light_phase(selected_rsu.id, "yellow")).pack(side="left", padx=5)
        StylishButton(light_btns, text="GREEN", color="secondary", width=7,
                     command=lambda: self.set_light_phase(selected_rsu.id, "green")).pack(side="left", padx=5)
        
        # Apply recommendation button
        StylishButton(actions_frame, text="Apply Recommendation", color="primary",
                     command=lambda: self.apply_recommendation(selected_rsu.id)).pack(fill="x", pady=10)
        
        # Divider
        ttk.Separator(details_frame, orient="horizontal").pack(fill="x", pady=10)
        
        # Traffic chart
        chart_frame = tk.Frame(details_frame, bg="white")
        chart_frame.pack(fill="both", expand=True, pady=10)
        
        # Create a matplotlib figure for the RSU
        fig = Figure(figsize=(5, 3), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        # Plot vehicle count history
        timestamps = selected_rsu.history["timestamps"][-15:]  # Last 15 points
        vehicle_counts = selected_rsu.history["vehicle_counts"][-15:]
        
        ax.plot(range(len(timestamps)), vehicle_counts, 'o-', color=COLORS["primary"])
        ax.set_title('Vehicle Count History')
        ax.set_ylabel('Vehicles')
        
        # Set x-axis ticks to timestamps
        if timestamps:
            ax.set_xticks(range(len(timestamps)))
            
            # If there are many points, show only some labels
            if len(timestamps) > 7:
                show_idx = list(range(0, len(timestamps), 3))
                if len(timestamps) - 1 not in show_idx:
                    show_idx.append(len(timestamps) - 1)
                
                labels = [""] * len(timestamps)
                for idx in show_idx:
                    if idx < len(timestamps):
                        labels[idx] = timestamps[idx]
                
                ax.set_xticklabels(labels, rotation=45)
            else:
                ax.set_xticklabels(timestamps, rotation=45)
        
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Create canvas
        self.rsu_chart = FigureCanvasTkAgg(fig, master=chart_frame)
        self.rsu_chart.draw()
        self.rsu_chart.get_tk_widget().pack(fill="both", expand=True)
    
    def set_light_phase(self, rsu_id, phase):
        """Set the traffic light phase for a specific RSU"""
        # Find the traffic light ID corresponding to this RSU
        tl_id = rsu_id  # In our implementation, the RSU ID is the traffic light ID
        
        try:
            # Get the current state to determine length
            current_state = traci.trafficlight.getRedYellowGreenState(tl_id)
            state_len = len(current_state)
            
            # Create new state string
            if phase == "red":
                new_state = "r" * state_len
            elif phase == "yellow":
                new_state = "y" * state_len
            else:  # green
                new_state = "G" * state_len
                
            # Apply the new state
            traci.trafficlight.setRedYellowGreenState(tl_id, new_state)
            
            messagebox.showinfo("Traffic Light Control", 
                               f"Set traffic light at {rsu_id} to {phase.upper()}")
        except traci.TraCIException as e:
            messagebox.showerror("Error", f"Failed to control traffic light: {e}")
    
    def apply_recommendation(self, rsu_id):
        """Apply the recommended phase for this RSU"""
        # Find the RSU
        selected_rsu = None
        for rsu in self.rsus:
            if rsu.id == rsu_id:
                selected_rsu = rsu
                break
                
        if not selected_rsu:
            return
            
        # Get recommendation
        rec_phase, rec_duration = selected_rsu.get_recommended_phase()
        
        # Apply it
        self.set_light_phase(rsu_id, rec_phase)
        
        messagebox.showinfo("Applied Recommendation", 
                          f"Applied recommended phase: {rec_phase.upper()} for {rec_duration}s")
    
    def setup_scenario_tab(self, parent):
        """Setup the traffic scenario tab"""
        # Split into two columns
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # Left panel - Scenario selection
        scenario_panel = DashboardPanel(parent, title="Traffic Scenarios")
        scenario_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)
        
        # Scenario options frame
        options_frame = tk.Frame(scenario_panel.content, bg="white")
        options_frame.pack(fill="both", expand=True, pady=10)
        
        # Scenario selection
        tk.Label(options_frame, text="Select Scenario:", bg="white", 
                fg=COLORS["text"], font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 5))
        
        # Use a combobox for scenario selection
        self.scenario_var = tk.StringVar(value="Normal Traffic")
        scenarios = list(self.scenario_manager.scenarios.keys())
        
        style = ttk.Style()
        style.configure("TCombobox", arrowsize=15)
        
        scenario_cb = ttk.Combobox(options_frame, 
                                   textvariable=self.scenario_var,
                                   values=scenarios,
                                   font=("Segoe UI", 10),
                                   width=25,
                                   state="readonly")
        scenario_cb.pack(anchor="w", pady=(0, 15))
        scenario_cb.bind("<<ComboboxSelected>>", self.on_scenario_selected)
        
        # Intensity slider
        tk.Label(options_frame, text="Traffic Intensity:", bg="white", 
                fg=COLORS["text"], font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 5))
        
        # Slider with value indicator
        slider_frame = tk.Frame(options_frame, bg="white")
        slider_frame.pack(fill="x", pady=(0, 15))
        
        self.intensity_var = tk.IntVar(value=5)
        intensity_slider = ttk.Scale(slider_frame, from_=1, to=20, 
                                    orient="horizontal", 
                                    variable=self.intensity_var,
                                    length=200,
                                    command=self.update_intensity_label)
        intensity_slider.pack(side="left")
        
        self.intensity_label = tk.Label(slider_frame, text="5", bg="white", 
                                       fg=COLORS["text"], width=2, font=("Segoe UI", 10))
        self.intensity_label.pack(side="left", padx=5)
        
        # Create a separator
        ttk.Separator(options_frame, orient="horizontal").pack(fill="x", pady=15)
        
        # Apply button
        StylishButton(options_frame, text="Apply Scenario", color="primary",
                     command=self.apply_selected_scenario).pack(anchor="w", pady=10)
        
        # Description area
        desc_frame = tk.LabelFrame(options_frame, text="Scenario Description", 
                                  bg="white", fg=COLORS["text"], padx=10, pady=10)
        desc_frame.pack(fill="both", expand=True, pady=10)
        
        self.scenario_desc = tk.Text(desc_frame, height=8, wrap="word", 
                                    font=("Segoe UI", 10), bg="white", fg=COLORS["text"],
                                    borderwidth=0, highlightthickness=0)
        self.scenario_desc.pack(fill="both", expand=True)
        self.scenario_desc.config(state="disabled")
        
        # Initial description update
        self.update_scenario_description()
        
        # Right panel - Custom scenario builder
        custom_panel = DashboardPanel(parent, title="Custom Scenario Builder")
        custom_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)
        
        # Custom scenario options
        custom_frame = tk.Frame(custom_panel.content, bg="white")
        custom_frame.pack(fill="both", expand=True, pady=10)
        
        # Vehicle count
        tk.Label(custom_frame, text="Number of Vehicles:", bg="white", 
                fg=COLORS["text"], font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 5))
        
        self.vehicle_count_var = tk.IntVar(value=10)
        
        count_frame = tk.Frame(custom_frame, bg="white")
        count_frame.pack(anchor="w", pady=(0, 15))
        
        count_entry = ttk.Entry(count_frame, width=5, textvariable=self.vehicle_count_var)
        count_entry.pack(side="left")
        
        # Add plus/minus buttons
        StylishButton(count_frame, text="-", color="dark", width=2,
                     command=lambda: self.vehicle_count_var.set(max(1, self.vehicle_count_var.get() - 1))).pack(side="left", padx=5)
        StylishButton(count_frame, text="+", color="dark", width=2,
                     command=lambda: self.vehicle_count_var.set(self.vehicle_count_var.get() + 1)).pack(side="left")
        
        # Vehicle type
        tk.Label(custom_frame, text="Vehicle Type:", bg="white", 
                fg=COLORS["text"], font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 5))
        
        # Radio buttons for vehicle types
        type_frame = tk.Frame(custom_frame, bg="white")
        type_frame.pack(anchor="w", pady=(0, 15))
        
        self.vehicle_type_var = tk.StringVar(value="normal")
        
        types = [
            ("Normal", "normal"),
            ("Heavy", "heavy"),
            ("Fast", "fast"),
            ("Emergency", "emergency")
        ]
        
        for text, value in types:
            rb = ttk.Radiobutton(type_frame, text=text, value=value, variable=self.vehicle_type_var)
            rb.pack(side="left", padx=10)
        
        # Maximum speed
        tk.Label(custom_frame, text="Max Speed (m/s):", bg="white", 
                fg=COLORS["text"], font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 5))
        
        self.max_speed_var = tk.DoubleVar(value=13.9)  # ~50 km/h
        
        speed_frame = tk.Frame(custom_frame, bg="white")
        speed_frame.pack(anchor="w", pady=(0, 15))
        
        speed_entry = ttk.Entry(speed_frame, width=5, textvariable=self.max_speed_var)
        speed_entry.pack(side="left")
        
        # Add tooltip or explanation
        tk.Label(speed_frame, text="(50 km/h ≈ 13.9 m/s)", 
                fg="#777", bg="white", font=("Segoe UI", 9)).pack(side="left", padx=10)
        
        # Vehicle color
        tk.Label(custom_frame, text="Vehicle Color:", bg="white", 
                fg=COLORS["text"], font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 5))
        
        self.color_var = tk.StringVar(value="#FF0000")  # Red by default
        
        color_frame = tk.Frame(custom_frame, bg="white")
        color_frame.pack(anchor="w", pady=(0, 15))
        
        # Color preview box
        self.color_preview = tk.Frame(color_frame, width=20, height=20, bg=self.color_var.get())
        self.color_preview.pack(side="left")
        self.color_preview.pack_propagate(False)
        
        # Color picker button
        StylishButton(color_frame, text="Pick Color", color="primary",
                     command=self.pick_color).pack(side="left", padx=10)
        
        # Create a separator
        ttk.Separator(custom_frame, orient="horizontal").pack(fill="x", pady=15)
        
        # Apply button
        StylishButton(custom_frame, text="Generate Custom Scenario", color="primary", 
                     command=self.apply_custom_scenario).pack(anchor="w", pady=10)
    
    def update_intensity_label(self, *args):
        """Update the intensity label when the slider changes"""
        self.intensity_label.config(text=str(self.intensity_var.get()))
    
    def on_scenario_selected(self, event):
        """Handle scenario selection change"""
        self.update_scenario_description()
    
    def update_scenario_description(self):
        """Update the scenario description based on selection"""
        scenario = self.scenario_var.get()
        
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
        
        # Update the description text
        self.scenario_desc.config(state="normal")
        self.scenario_desc.delete(1.0, tk.END)
        
        if scenario in descriptions:
            intensity = self.intensity_var.get()
            
            self.scenario_desc.insert(tk.END, f"{scenario} Scenario\n\n")
            self.scenario_desc.insert(tk.END, descriptions[scenario] + "\n\n")
            self.scenario_desc.insert(tk.END, f"Current Intensity: {intensity}/20\n")
            
            if scenario in self.scenario_manager.scenarios:
                expected_count = intensity * self.scenario_manager.scenarios[scenario]['density']
                self.scenario_desc.insert(tk.END, f"Expected Vehicle Count: ~{expected_count}\n")
            
            if scenario != "Normal Traffic":
                self.scenario_desc.insert(tk.END, "\nSpecial Effects:\n")
                
                if scenario == "Rainy Day" or scenario == "Foggy Morning":
                    max_speed = self.scenario_manager.scenarios[scenario].get('max_speed', 'N/A')
                    self.scenario_desc.insert(tk.END, "- Reduced visibility and speed\n")
                    self.scenario_desc.insert(tk.END, f"- Max speed: {max_speed} m/s\n")
                
                if scenario == "Rush Hour":
                    self.scenario_desc.insert(tk.END, "- High congestion at intersections\n")
                    self.scenario_desc.insert(tk.END, "- Longer waiting times\n")
                
                if scenario == "Emergency":
                    self.scenario_desc.insert(tk.END, "- Emergency vehicles have priority\n")
                    self.scenario_desc.insert(tk.END, "- Traffic signals may adapt to emergency vehicles\n")
        
        self.scenario_desc.config(state="disabled")
    
    def pick_color(self):
        """Open a color picker dialog"""
        color = colorchooser.askcolor(initialcolor=self.color_var.get())[1]
        if color:
            self.color_var.set(color)
            self.color_preview.config(bg=color)
    
    def apply_selected_scenario(self):
        """Apply the currently selected scenario"""
        scenario = self.scenario_var.get()
        intensity = self.intensity_var.get()
        
        self.scenario_manager.apply_scenario(scenario, intensity)
        messagebox.showinfo("Scenario Applied", f"Applied {scenario} scenario with intensity {intensity}")
    
    def apply_custom_scenario(self):
        """Apply a custom scenario"""
        try:
            vehicle_count = self.vehicle_count_var.get()
            max_speed = self.max_speed_var.get()
            vehicle_type = self.vehicle_type_var.get()
            color = self.color_var.get()
            
            # Convert hex color to RGBA tuple
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            rgba = (r, g, b, 255)
            
            # Create custom vehicle type
            custom_type = f"custom_{vehicle_type}_vehicle"
            try:
                traci.vehicletype.copy("veh_passenger", custom_type)
                traci.vehicletype.setColor(custom_type, rgba)
                traci.vehicletype.setMaxSpeed(custom_type, max_speed)
            except traci.TraCIException as e:
                print(f"Error setting up custom vehicle type: {e}")
            
            # Add vehicles
            edges = self.scenario_manager.load_trip_edges()
            added = 0
            
            for i in range(vehicle_count):
                try:
                    vehicle_id = f"custom_{int(traci.simulation.getTime())}_{i}"
                    from_edge, to_edge = random.choice(edges)
                    route_id = f"route_{vehicle_id}"
                    
                    traci.route.add(routeID=route_id, edges=[from_edge, to_edge])
                    traci.vehicle.add(vehID=vehicle_id, routeID=route_id, 
                                     typeID=custom_type, departLane="best", departSpeed="max")
                    added += 1
                except traci.TraCIException as e:
                    print(f"Error adding vehicle {i}: {e}")
            
            messagebox.showinfo("Custom Scenario", 
                               f"Added {added} custom vehicles with max speed {max_speed:.1f} m/s")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply custom scenario: {e}")
    
    def setup_analytics_tab(self, parent):
        """Setup the analytics tab with graphs and statistics"""
        # Create a matplotlib figure for overall vehicle count
        fig = Figure(figsize=(12, 6), dpi=100, facecolor='white')
        
        # Vehicle count subplot
        ax1 = fig.add_subplot(211)
        self.vehicle_count_line, = ax1.plot([], [], 'o-', color=COLORS["primary"], lw=2)
        ax1.set_title('Total Vehicles in Simulation')
        ax1.set_xlabel('Simulation Steps')
        ax1.set_ylabel('Vehicle Count')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Average speed subplot
        ax2 = fig.add_subplot(212)
        self.avg_speed_line, = ax2.plot([], [], 'o-', color=COLORS["secondary"], lw=2)
        ax2.set_title('Average Vehicle Speed')
        ax2.set_xlabel('Simulation Steps')
        ax2.set_ylabel('Speed (m/s)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        fig.tight_layout()
        
        # Store the axes for updating
        self.analytics_axes = [ax1, ax2]
        
        # Create the canvas
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        # Data storage for the graphs
        self.analytics_data = {
            "steps": [],
            "vehicle_counts": [],
            "avg_speeds": []
        }
    
    def update_analytics_graphs(self):
        """Update the analytics graphs with current data"""
        # Add current simulation data
        sim_time = traci.simulation.getTime()
        
        # Only update every few steps to avoid too many points
        if not self.analytics_data["steps"] or sim_time - self.analytics_data["steps"][-1] >= 10:
            vehicles = traci.vehicle.getIDList()
            vehicle_count = len(vehicles)
            
            # Calculate average speed
            avg_speed = 0
            if vehicle_count > 0:
                speeds = [traci.vehicle.getSpeed(veh) for veh in vehicles]
                avg_speed = sum(speeds) / vehicle_count
            
            # Add to data storage
            self.analytics_data["steps"].append(sim_time)
            self.analytics_data["vehicle_counts"].append(vehicle_count)
            self.analytics_data["avg_speeds"].append(avg_speed)
            
            # Keep only last 30 points
            if len(self.analytics_data["steps"]) > 30:
                self.analytics_data["steps"] = self.analytics_data["steps"][-30:]
                self.analytics_data["vehicle_counts"] = self.analytics_data["vehicle_counts"][-30:]
                self.analytics_data["avg_speeds"] = self.analytics_data["avg_speeds"][-30:]
            
            # Update the plots
            self.vehicle_count_line.set_data(range(len(self.analytics_data["steps"])), 
                                            self.analytics_data["vehicle_counts"])
            self.avg_speed_line.set_data(range(len(self.analytics_data["steps"])), 
                                        self.analytics_data["avg_speeds"])
            
            # Adjust axes limits
            for i, ax in enumerate(self.analytics_axes):
                ax.relim()
                ax.autoscale_view()
                
                # Set x-axis ticks
                if self.analytics_data["steps"]:
                    ax.set_xticks(range(len(self.analytics_data["steps"])))
                    
                    # If there are many points, show only some labels
                    if len(self.analytics_data["steps"]) > 10:
                        show_idx = list(range(0, len(self.analytics_data["steps"]), 3))
                        if len(self.analytics_data["steps"]) - 1 not in show_idx:
                            show_idx.append(len(self.analytics_data["steps"]) - 1)
                        
                        labels = [""] * len(self.analytics_data["steps"])
                        for idx in show_idx:
                            if idx < len(self.analytics_data["steps"]):
                                labels[idx] = f"{self.analytics_data['steps'][idx]:.0f}"
                        
                        ax.set_xticklabels(labels)
                    else:
                        ax.set_xticklabels([f"{s:.0f}" for s in self.analytics_data["steps"]])
            
            # Redraw the canvas
            if hasattr(self, 'rsu_chart') and self.rsu_chart:
                self.rsu_chart.draw()
    
    def setup_status_bar(self):
        """Setup the status bar at the bottom of the window"""
        status_frame = tk.Frame(self.root, bg=COLORS["dark"], height=25)
        status_frame.pack(side="bottom", fill="x")
        
        # Left side - status message
        self.status_text = tk.StringVar(value="Simulation ready")
        status_label = tk.Label(
            status_frame, 
            textvariable=self.status_text,
            bg=COLORS["dark"],
            fg="white",
            font=("Segoe UI", 9),
            padx=10
        )
        status_label.pack(side="left")
        
        # Right side - simulation time
        self.time_text = tk.StringVar(value="Time: 0.0s")
        time_label = tk.Label(
            status_frame, 
            textvariable=self.time_text,
            bg=COLORS["dark"],
            fg="white",
            font=("Segoe UI", 9),
            padx=10
        )
        time_label.pack(side="right")
    
    def toggle_simulation(self):
        """Toggle the simulation between running and paused"""
        self.running = not self.running
        if self.running:
            self.play_btn.config(text="⏸ Pause")
            self.status_text.set("Simulation running")
        else:
            self.play_btn.config(text="▶ Play")
            self.status_text.set("Simulation paused")
    
    def insert_vehicles(self):
        """Insert vehicles into the simulation"""
        try:
            num_vehicles = simpledialog.askinteger(
                "Add Vehicles", 
                "Enter number of vehicles:",
                parent=self.root,
                minvalue=1,
                maxvalue=100
            )
            
            if not num_vehicles:
                return
                
            edges = self.scenario_manager.load_trip_edges()
            if not edges:
                messagebox.showerror("Error", "No valid routes found")
                return

            # Add the vehicles
            added = 0
            for i in range(num_vehicles):
                try:
                    vehicle_id = f"vehicle_{int(traci.simulation.getTime())}_{i}"
                    from_edge, to_edge = random.choice(edges)
                    route_id = f"route_{vehicle_id}"
                    traci.route.add(routeID=route_id, edges=[from_edge, to_edge])
                    traci.vehicle.add(vehID=vehicle_id, routeID=route_id, typeID="veh_passenger",
                                    departLane="best", departSpeed="max")
                    added += 1
                except traci.TraCIException as e:
                    print(f"Error adding vehicle {vehicle_id}: {e}")
            
            self.status_text.set(f"Added {added} vehicles to the simulation")
            messagebox.showinfo("Vehicles Added", f"Successfully added {added} vehicles to the simulation")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add vehicles: {e}")
    
    def stop_simulation(self):
        """Stop the simulation and cleanup"""
        if messagebox.askyesno("Stop Simulation", "Are you sure you want to stop the simulation?"):
            self.exit_simulation = True
            
            # Close SUMO
            try:
                traci.close()
            except:
                pass
            
            self.root.quit()
    
    def update_rsu_tree(self):
        """Update the RSU information in the treeview"""
        # Clear the current contents
        for item in self.rsu_tree.get_children():
            self.rsu_tree.delete(item)
            
        # Add updated data for each RSU
        for rsu in self.rsus:
            # Calculate average speed
            avg_speed = 0
            if rsu.vehicles_in_range:
                speeds = [data["speed"] for data in rsu.vehicle_data.values() if "speed" in data]
                avg_speed = sum(speeds) / len(speeds) if speeds else 0
            
            # Add to tree
            self.rsu_tree.insert("", "end", values=(
                rsu.id,
                f"({rsu.position[0]:.1f}, {rsu.position[1]:.1f})",
                len(rsu.vehicles_in_range),
                rsu.congestion_level.upper(),
                f"{avg_speed:.1f} m/s"
            ))
    
    def update_dashboard_tree(self):
        """Update the RSU information in the dashboard treeview"""
        # Clear the current contents
        for item in self.dashboard_tree.get_children():
            self.dashboard_tree.delete(item)
            
        # Add updated data for each RSU
        for rsu in self.rsus:
            # Get recommendation
            rec_phase, rec_duration = rsu.get_recommended_phase()
            
            # Add to tree with background color based on congestion
            tag = rsu.congestion_level
            self.dashboard_tree.insert("", "end", tags=(tag,), values=(
                rsu.id,
                len(rsu.vehicles_in_range),
                rsu.congestion_level.upper(),
                f"{rec_phase.upper()} for {rec_duration}s"
            ))
        
        # Configure tag colors
        self.dashboard_tree.tag_configure("low", background="#e8f5e9")    # Light green
        self.dashboard_tree.tag_configure("medium", background="#fff9c4")  # Light yellow
        self.dashboard_tree.tag_configure("high", background="#ffebee")   # Light red
    
    def update_dashboard_chart(self):
        """Update the dashboard chart with current vehicle count data"""
        # Get the total vehicle count over time
        total_counts = []
        timestamps = []
        
        # Use data from all RSUs, or just get the current total
        vehicles = traci.vehicle.getIDList()
        total_count = len(vehicles)
        
        # Add current timestamp and count
        current_time = datetime.now().strftime("%H:%M:%S")
        timestamps.append(current_time)
        total_counts.append(total_count)
        
        # Limit to last 10 points
        if len(timestamps) > 10:
            timestamps = timestamps[-10:]
            total_counts = total_counts[-10:]
        
        # Clear the previous plot
        self.ax.clear()
        
        # Create the new plot
        self.ax.plot(range(len(timestamps)), total_counts, 'o-', color=COLORS["primary"], lw=2)
        self.ax.set_title('Vehicle Count Over Time')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Vehicles')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set x-axis ticks to timestamps
        if timestamps:
            self.ax.set_xticks(range(len(timestamps)))
            
            # If there are many points, show only some labels
            if len(timestamps) > 5:
                show_idx = list(range(0, len(timestamps), 2))
                if len(timestamps) - 1 not in show_idx:
                    show_idx.append(len(timestamps) - 1)
                
                labels = [""] * len(timestamps)
                for idx in show_idx:
                    if idx < len(timestamps):
                        labels[idx] = timestamps[idx]
                
                self.ax.set_xticklabels(labels, rotation=45)
            else:
                self.ax.set_xticklabels(timestamps, rotation=45)
        
        # Redraw the canvas
        self.canvas.draw()
    
    def update_dashboard_stats(self):
        """Update the dashboard statistics"""
        # Calculate overall statistics
        vehicles = traci.vehicle.getIDList()
        vehicle_count = len(vehicles)
        
        # Calculate average speed
        avg_speed = 0
        if vehicle_count > 0:
            speeds = [traci.vehicle.getSpeed(veh) for veh in vehicles]
            avg_speed = sum(speeds) / vehicle_count if speeds else 0
        
        # Determine overall congestion level
        congestion_counts = {"low": 0, "medium": 0, "high": 0}
        for rsu in self.rsus:
            congestion_counts[rsu.congestion_level] += 1
        
        if congestion_counts["high"] > len(self.rsus) * 0.3:
            overall_congestion = "High"
        elif congestion_counts["medium"] > len(self.rsus) * 0.3:
            overall_congestion = "Medium"
        else:
            overall_congestion = "Low"
        
        # Get simulation time
        sim_time = traci.simulation.getTime()
        minutes = int(sim_time) // 60
        seconds = int(sim_time) % 60
        time_str = f"{minutes}:{seconds:02d}"
        
        # Update the labels
        self.vehicle_count_label.config(text=str(vehicle_count))
        self.avg_speed_label.config(text=f"{avg_speed:.1f}")
        self.congestion_label.config(text=overall_congestion)
        self.sim_time_label.config(text=time_str)
        
        # Update the status bar time
        self.time_text.set(f"Time: {sim_time:.1f}s")
    
    def run_simulation(self):
        """Main simulation loop"""
        update_interval = 0.5  # Update GUI every half second
        last_update = 0
        
        while not self.exit_simulation:
            if self.running:
                try:
                    # Step the simulation
                    traci.simulationStep()
                    
                    # Current time
                    current_time = time.time()
                    
                    # Update RSUs
                    for rsu in self.rsus:
                        rsu.update()
                    
                    # Update GUI components periodically
                    if current_time - last_update >= update_interval:
                        # Update dashboard statistics
                        self.update_dashboard_stats()
                        
                        # Update RSU treeviews
                        self.update_rsu_tree()
                        self.update_dashboard_tree()
                        
                        # Update charts
                        self.update_dashboard_chart()
                        self.update_analytics_graphs()
                        
                        # If an RSU is selected, update its details
                        if self.current_rsu_id:
                            self.update_rsu_details(self.current_rsu_id)
                        
                        last_update = current_time
                    
                    # Slight delay to avoid excessive CPU usage
                    time.sleep(0.05)
                    
                except traci.TraCIException as e:
                    print(f"Simulation error: {e}")
                    if "connection closed by SUMO" in str(e):
                        self.exit_simulation = True
                        break
                    self.status_text.set(f"Error: {e}")
            
            # Process GUI events
            self.root.update()
            time.sleep(0.05)
        
        print("Simulation ended")

def main():
    """Main function to start the application"""
    try:
        # Check if SUMO_HOME environment variable is set
        if 'SUMO_HOME' not in os.environ:
            print("Please set SUMO_HOME environment variable")
            print("On Windows: set SUMO_HOME=C:\\path\\to\\sumo")
            print("On Linux: export SUMO_HOME=/path/to/sumo")
            sys.exit(1)
            
        # Start the GUI application
        app = ModernTrafficGUI()
        app.root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()