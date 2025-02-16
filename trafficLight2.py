import traci
import tkinter as tk
from tkinter import ttk, simpledialog
import threading
import time
import random

# List of pre-defined valid trip edges
trip_edges = [
    ("24242882#0", "15973619#6"),
    ("4611711", "-120675240#0"),
    ("-28251222", "-120675240#2"),
    ("4611693#0", "-1105574291#1"),
    ("4611693#0", "15973619#8"),
    ("147066248#1", "-120675240#0"),
    ("4611693#0", "243854725#1"),
    ("120675240#0", "68647306#5"),
    ("4611708#2", "1159156576#1"),
    ("23017853", "-1132162834#0"),
    ("35557143#1", "-43231842#0"),
    ("-1233798019", "-24242882#0"),
    ("147066248#3", "-35557161#3")
]

class TrafficSimulationGUI:
    def __init__(self):
        # Initialize SUMO
        self.sumoBinary = "sumo-gui"
        self.sumoCmd = [self.sumoBinary, "-c", "osm.sumocfg"]
        traci.start(self.sumoCmd)
        
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("SUMO Traffic Control")
        self.root.geometry("1000x800")
        
        # Control variables
        self.running = False
        self.exit_simulation = False
        self.current_phase = "red"
        self.phase_time_remaining = 0
        
        # Timer variables
        self.red_time = tk.IntVar(value=30)
        self.yellow_time = tk.IntVar(value=5)
        self.green_time = tk.IntVar(value=30)
        
        # Store traffic light states
        self.tl_ids = traci.trafficlight.getIDList()
        
        # Define vehicle type for visualization
        traci.vehicletype.copy("veh_passenger", "red_passenger")
        traci.vehicletype.setColor("red_passenger", (255, 0, 0, 255))
        
        self.setup_gui()
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self.run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        # Start timer thread
        self.timer_thread = threading.Thread(target=self.run_timer)
        self.timer_thread.daemon = True
        self.timer_thread.start()

    def setup_gui(self):
        """Set up the main GUI components"""
        # Create frames
        self.control_frame = ttk.LabelFrame(self.root, text="Simulation Controls", padding="10")
        self.control_frame.pack(fill="x", padx=5, pady=5)
        
        self.traffic_frame = ttk.LabelFrame(self.root, text="Traffic Light Controls", padding="10")
        self.traffic_frame.pack(fill="x", padx=5, pady=5)
        
        self.timer_frame = ttk.LabelFrame(self.root, text="Traffic Light Timers", padding="10")
        self.timer_frame.pack(fill="x", padx=5, pady=5)
        
        # Simulation control buttons
        self.play_button = ttk.Button(self.control_frame, text="Play", command=self.toggle_simulation)
        self.play_button.pack(side="left", padx=5)
        
        ttk.Button(self.control_frame, text="Add Vehicles", command=self.insert_vehicles).pack(side="left", padx=5)
        ttk.Button(self.control_frame, text="Stop Simulation", command=self.stop_simulation).pack(side="left", padx=5)

        # Traffic light manual control buttons
        ttk.Label(self.traffic_frame, text="Manual Control:").pack(side="left", padx=5)
        
        red_btn = tk.Button(self.traffic_frame, text="RED", command=lambda: self.set_all_lights("r"),
                           bg='red', fg='white', width=10)
        red_btn.pack(side="left", padx=5)
        
        yellow_btn = tk.Button(self.traffic_frame, text="YELLOW", command=lambda: self.set_all_lights("y"),
                             bg='yellow', fg='black', width=10)
        yellow_btn.pack(side="left", padx=5)
        
        green_btn = tk.Button(self.traffic_frame, text="GREEN", command=lambda: self.set_all_lights("g"),
                            bg='green', fg='white', width=10)
        green_btn.pack(side="left", padx=5)

        # Status display
        self.status_label = ttk.Label(self.traffic_frame, text="Current Status: -")
        self.status_label.pack(side="left", padx=20)

        # Timer controls
        self.setup_timer_controls()

    def setup_timer_controls(self):
        """Set up the timer controls and progress bars"""
        # Red light timer
        red_frame = ttk.Frame(self.timer_frame)
        red_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(red_frame, text="Red Light Duration (s):").pack(side="left", padx=5)
        ttk.Entry(red_frame, textvariable=self.red_time, width=5).pack(side="left", padx=5)
        self.red_progress = ttk.Progressbar(red_frame, length=200, mode='determinate')
        self.red_progress.pack(side="left", padx=5)
        
        # Yellow light timer
        yellow_frame = ttk.Frame(self.timer_frame)
        yellow_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(yellow_frame, text="Yellow Light Duration (s):").pack(side="left", padx=5)
        ttk.Entry(yellow_frame, textvariable=self.yellow_time, width=5).pack(side="left", padx=5)
        self.yellow_progress = ttk.Progressbar(yellow_frame, length=200, mode='determinate')
        self.yellow_progress.pack(side="left", padx=5)
        
        # Green light timer
        green_frame = ttk.Frame(self.timer_frame)
        green_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(green_frame, text="Green Light Duration (s):").pack(side="left", padx=5)
        ttk.Entry(green_frame, textvariable=self.green_time, width=5).pack(side="left", padx=5)
        self.green_progress = ttk.Progressbar(green_frame, length=200, mode='determinate')
        self.green_progress.pack(side="left", padx=5)

        # Timer control buttons
        control_frame = ttk.Frame(self.timer_frame)
        control_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(control_frame, text="Start Automatic Cycle", 
                  command=self.start_automatic_cycle).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Stop Automatic Cycle", 
                  command=self.stop_automatic_cycle).pack(side="left", padx=5)

        # Phase indicator
        self.phase_label = ttk.Label(control_frame, text="Current Phase: -")
        self.phase_label.pack(side="left", padx=20)
        self.time_remaining_label = ttk.Label(control_frame, text="Time Remaining: -")
        self.time_remaining_label.pack(side="left", padx=20)

    def start_automatic_cycle(self):
        """Start the automatic traffic light cycle"""
        self.automatic_cycle = True
        self.phase_time_remaining = self.red_time.get()
        self.current_phase = "red"
        self.set_all_lights("r")

    def stop_automatic_cycle(self):
        """Stop the automatic traffic light cycle"""
        self.automatic_cycle = False
        self.phase_time_remaining = 0
        self.reset_progress_bars()

    def reset_progress_bars(self):
        """Reset all progress bars to zero"""
        self.red_progress['value'] = 0
        self.yellow_progress['value'] = 0
        self.green_progress['value'] = 0

    def run_timer(self):
        """Run the timer for automatic traffic light cycling"""
        self.automatic_cycle = False
        
        while not self.exit_simulation:
            if self.automatic_cycle and self.running:
                if self.phase_time_remaining <= 0:
                    # Switch to next phase
                    if self.current_phase == "red":
                        self.current_phase = "green"
                        self.phase_time_remaining = self.green_time.get()
                        self.set_all_lights("g")
                    elif self.current_phase == "green":
                        self.current_phase = "yellow"
                        self.phase_time_remaining = self.yellow_time.get()
                        self.set_all_lights("y")
                    else:  # yellow
                        self.current_phase = "red"
                        self.phase_time_remaining = self.red_time.get()
                        self.set_all_lights("r")
                    
                    self.reset_progress_bars()
                
                # Update progress bar for current phase
                if self.current_phase == "red":
                    progress = ((self.red_time.get() - self.phase_time_remaining) / self.red_time.get()) * 100
                    self.red_progress['value'] = progress
                elif self.current_phase == "yellow":
                    progress = ((self.yellow_time.get() - self.phase_time_remaining) / self.yellow_time.get()) * 100
                    self.yellow_progress['value'] = progress
                else:  # green
                    progress = ((self.green_time.get() - self.phase_time_remaining) / self.green_time.get()) * 100
                    self.green_progress['value'] = progress
                
                # Update labels
                self.phase_label.config(text=f"Current Phase: {self.current_phase.upper()}")
                self.time_remaining_label.config(text=f"Time Remaining: {self.phase_time_remaining}s")
                
                self.phase_time_remaining -= 1
                
            time.sleep(1)

    def set_all_lights(self, color):
        """Set all traffic lights to specified color (r, y, or g)"""
        try:
            for tl_id in self.tl_ids:
                current_state = traci.trafficlight.getRedYellowGreenState(tl_id)
                new_state = color * len(current_state)
                traci.trafficlight.setRedYellowGreenState(tl_id, new_state)
            
            color_text = {"r": "RED", "y": "YELLOW", "g": "GREEN"}
            self.status_label.config(text=f"Current Status: All lights {color_text[color]}")
            
        except traci.TraCIException as e:
            print(f"Error setting traffic lights: {e}")
            self.status_label.config(text=f"Error: Failed to set lights")

    def insert_vehicles(self):
        """Insert new vehicles into the simulation"""
        num_vehicles = simpledialog.askinteger("Input", "Enter number of vehicles:", parent=self.root)
        if num_vehicles is None or num_vehicles < 1:
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
                print(f"Error adding vehicle: {e}")

    def toggle_simulation(self):
        """Toggle simulation between running and paused states"""
        self.running = not self.running
        self.play_button.config(text="Pause" if self.running else "Play")

    def stop_simulation(self):
        """Stop the simulation"""
        self.exit_simulation = True
        self.root.quit()

    def run_simulation(self):
        """Main simulation loop"""
        step = 0
        while not self.exit_simulation:
            if self.running:
                try:
                    traci.simulationStep()
                    
                    if step % 50 == 0:  # Every 5 seconds
                        self.log_simulation_state()
                    
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
        print("Simulation ended.")

    def log_simulation_state(self):
        """Log the current state of the simulation"""
        try:
            vehicle_count = len(traci.vehicle.getIDList())
            print(f"\nSimulation State:")
            print(f"Active vehicles: {vehicle_count}")
            
            for tl_id in self.tl_ids:
                state = traci.trafficlight.getRedYellowGreenState(tl_id)
                print(f"Traffic Light {tl_id}: State={state}")
            
        except traci.TraCIException as e:
            print(f"Error logging simulation state: {e}")

    def start(self):
        """Start the GUI main loop"""
        self.root.mainloop()

if __name__ == "__main__":
    app = TrafficSimulationGUI()
    app.start()