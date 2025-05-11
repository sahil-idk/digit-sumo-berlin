"""
Main application class for the Smart Traffic Control System
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import traci

from config.settings import APP_CONFIG
from config.styles import COLORS, TTK_STYLES
from models.rsu import RSU
from models.scenario_manager import ScenarioManager
from utils.sumo_utils import start_sumo, close_sumo, configure_sumo_gui, get_simulation_stats

from gui.components.stylish_button import StylishButton
from gui.tabs.dashboard_tab import DashboardTab
from gui.tabs.rsu_tab import RSUTab
from gui.tabs.scenario_tab import ScenarioTab
from gui.tabs.analytics_tab import AnalyticsTab
from gui.tabs.prediction_tab import PredictionTab

class TrafficSimulationApp:
    """Main application class for the Smart Traffic Control System"""
    def __init__(self):
        # Initialize SUMO
        if not start_sumo():
            raise Exception("Failed to start SUMO. Please check your installation.")
            
        # Initialize main window
        self.root = tk.Tk()
        self.root.title(APP_CONFIG["title"])
        self.root.geometry(f"{APP_CONFIG['width']}x{APP_CONFIG['height']}")
        self.root.configure(bg=COLORS["background"])
        self.root.minsize(APP_CONFIG["min_width"], APP_CONFIG["min_height"])
        
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
        
        # Configure SUMO GUI
        configure_sumo_gui()
        
        # Shared data for analytics
        self.analytics_data = {
            "steps": [],
            "vehicle_counts": [],
            "avg_speeds": []
        }
        
        # Store tab instances for updates
        self.tabs = {}
    
    def configure_styles(self):
        """Configure ttk styles for a modern look"""
        # Apply all style configurations
        for style_name, config in TTK_STYLES.items():
            if "configure" in config:
                self.style.configure(style_name, **config["configure"])
            if "map" in config:
                self.style.map(style_name, **config["map"])
    
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
                    circle_points = rsu.create_circle_points()
                    traci.polygon.add(rsu.range_poi_id, 
                                     circle_points, 
                                     (0, 0, 255, 80), fill=True, layer=-1)
                    
                    print(f"Created RSU at junction {tl_id} with visual POI")
                except traci.TraCIException as e:
                    print(f"Error adding RSU POI for {tl_id}: {e}")
                    
            except traci.TraCIException as e:
                print(f"Error creating RSU at junction {tl_id}: {e}")
    
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
        dashboard_tab = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(dashboard_tab, text="Dashboard")
        self.tabs['dashboard'] = DashboardTab(dashboard_tab, self)
        
        # Tab 2: RSU Monitoring
        rsu_tab = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(rsu_tab, text="RSU Monitoring")
        self.tabs['rsu'] = RSUTab(rsu_tab, self)
        
        # Tab 3: Traffic Scenarios
        scenario_tab = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(scenario_tab, text="Traffic Scenarios")
        self.tabs['scenario'] = ScenarioTab(scenario_tab, self)
        
        # Tab 4: Analytics
        analytics_tab = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(analytics_tab, text="Analytics")
        self.tabs['analytics'] = AnalyticsTab(analytics_tab, self)
        
        # Tab 5: Predictions
        prediction_tab = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(prediction_tab, text="Predictions")
        self.tabs['prediction'] = PredictionTab(prediction_tab, self)
    
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
        """Prompt user and insert vehicles into the simulation"""
        # This implementation would normally use a dialog
        # For now, just insert 10 vehicles
        from tkinter import simpledialog
        
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
            
        # Use the scenario manager to insert vehicles
        result = self.scenario_manager.apply_scenario("Normal Traffic", num_vehicles / 5)
        
        if result:
            self.status_text.set(f"Added vehicles to the simulation")
            messagebox.showinfo("Vehicles Added", f"Successfully added vehicles to the simulation")
        else:
            messagebox.showerror("Error", "Failed to add vehicles")
    
    def stop_simulation(self):
        """Stop the simulation and cleanup"""
        if messagebox.askyesno("Stop Simulation", "Are you sure you want to stop the simulation?"):
            self.exit_simulation = True
            
            # Close SUMO
            close_sumo()
            
            self.root.quit()
    
    def update_gui(self):
        """Update all GUI components"""
        # Update RSUs
        for rsu in self.rsus:
            rsu.update()
        
        # Update stats
        stats = get_simulation_stats()
        
        # Update status bar
        sim_time = stats["sim_time"]
        minutes = int(sim_time) // 60
        seconds = int(sim_time) % 60
        self.time_text.set(f"Time: {sim_time:.1f}s ({minutes}:{seconds:02d})")
        
        # Add current simulation data to analytics
        self.analytics_data["steps"].append(sim_time)
        self.analytics_data["vehicle_counts"].append(stats["vehicle_count"])
        self.analytics_data["avg_speeds"].append(stats["avg_speed"])
        
        # Keep only last N points
        max_points = APP_CONFIG.get("max_data_points", 30)
        if len(self.analytics_data["steps"]) > max_points:
            self.analytics_data["steps"] = self.analytics_data["steps"][-max_points:]
            self.analytics_data["vehicle_counts"] = self.analytics_data["vehicle_counts"][-max_points:]
            self.analytics_data["avg_speeds"] = self.analytics_data["avg_speeds"][-max_points:]
        
        # Update each tab
        for tab_name, tab in self.tabs.items():
            if hasattr(tab, 'update'):
                tab.update()
    
    def run_simulation(self):
        """Main simulation loop"""
        update_interval = APP_CONFIG["update_interval"]
        last_update = 0
        
        while not self.exit_simulation:
            if self.running:
                try:
                    # Step the simulation
                    traci.simulationStep()
                    
                    # Current time
                    current_time = time.time()
                    
                    # Update GUI components periodically
                    if current_time - last_update >= update_interval:
                        self.update_gui()
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
    
    def run(self):
        """Start the simulation thread and run the main loop"""
        # Start simulation thread
        simulation_thread = threading.Thread(target=self.run_simulation)
        simulation_thread.daemon = True
        simulation_thread.start()
        
        # Run the tkinter main loop
        self.root.mainloop()