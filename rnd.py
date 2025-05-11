import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, colorchooser
import threading
import time
import random
import math
import os
import sys
import traci
from datetime import datetime
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

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