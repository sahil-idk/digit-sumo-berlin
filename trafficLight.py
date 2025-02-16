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
        self.root.geometry("800x600")
        
        # Control variables
        self.running = False
        self.exit_simulation = False
        
        # Store traffic light states
        self.tl_ids = traci.trafficlight.getIDList()
        
        # Define vehicle type for visualization
        traci.vehicletype.copy("veh_passenger", "red_passenger")
        traci.vehicletype.setColor("red_passenger", (255, 0, 0, 255))
        
        self.setup_gui()
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self.run_simulation)
        self.simulation_thread.start()

    def setup_gui(self):
        """Set up the main GUI components"""
        # Create frames
        self.control_frame = ttk.LabelFrame(self.root, text="Simulation Controls", padding="10")
        self.control_frame.pack(fill="x", padx=5, pady=5)
        
        self.traffic_frame = ttk.LabelFrame(self.root, text="Global Traffic Light Controls", padding="10")
        self.traffic_frame.pack(fill="x", padx=5, pady=5)
        
        # Simulation control buttons
        self.play_button = ttk.Button(self.control_frame, text="Play", command=self.toggle_simulation)
        self.play_button.pack(side="left", padx=5)
        
        ttk.Button(self.control_frame, text="Add Vehicles", command=self.insert_vehicles).pack(side="left", padx=5)
        ttk.Button(self.control_frame, text="Stop Simulation", command=self.stop_simulation).pack(side="left", padx=5)

        # Global traffic light control buttons
        ttk.Label(self.traffic_frame, text="Set all traffic lights to:").pack(side="left", padx=5)
        
        # Red button with red text
        red_btn = tk.Button(self.traffic_frame, text="RED", command=lambda: self.set_all_lights("r"),
                           bg='red', fg='white', width=10)
        red_btn.pack(side="left", padx=5)
        
        # Yellow button with yellow background
        yellow_btn = tk.Button(self.traffic_frame, text="YELLOW", command=lambda: self.set_all_lights("y"),
                             bg='yellow', fg='black', width=10)
        yellow_btn.pack(side="left", padx=5)
        
        # Green button with green background
        green_btn = tk.Button(self.traffic_frame, text="GREEN", command=lambda: self.set_all_lights("g"),
                            bg='green', fg='white', width=10)
        green_btn.pack(side="left", padx=5)

        # Status display
        self.status_label = ttk.Label(self.traffic_frame, text="Current Status: -")
        self.status_label.pack(side="left", padx=20)

    def set_all_lights(self, color):
        """Set all traffic lights to specified color (r, y, or g)"""
        try:
            for tl_id in self.tl_ids:
                current_state = traci.trafficlight.getRedYellowGreenState(tl_id)
                # Create new state string with all signals set to specified color
                new_state = color * len(current_state)
                traci.trafficlight.setRedYellowGreenState(tl_id, new_state)
            
            # Update status label
            color_text = {"r": "RED", "y": "YELLOW", "g": "GREEN"}
            self.status_label.config(text=f"Current Status: All lights {color_text[color]}")
            print(f"Set all traffic lights to {color_text[color]}")
        
        except traci.TraCIException as e:
            print(f"Error setting traffic lights: {e}")
            self.status_label.config(text=f"Error: Failed to set lights")

    def insert_vehicles(self):
        """Insert new vehicles into the simulation using predefined trip edges"""
        num_vehicles = simpledialog.askinteger("Input", "Enter number of vehicles:", parent=self.root)
        if num_vehicles is None or num_vehicles < 1:
            return

        for i in range(num_vehicles):
            vehicle_id = f"vehicle_{int(traci.simulation.getTime())}_{i}"
            try:
                # Randomly choose a trip (from_edge, to_edge)
                from_edge, to_edge = random.choice(trip_edges)
                route_id = f"route_{vehicle_id}"
                
                # Create a route for the vehicle
                traci.route.add(routeID=route_id, edges=[from_edge, to_edge])
                
                # Add the vehicle with the red passenger type
                traci.vehicle.add(vehID=vehicle_id, routeID=route_id, typeID="red_passenger",
                                departLane="best", departSpeed="max")
                print(f"Added vehicle {vehicle_id} from {from_edge} to {to_edge}")
            except traci.TraCIException as e:
                print(f"Error adding vehicle {vehicle_id}: {e}")

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
                    
                    # Log simulation state periodically
                    if step % 50 == 0:  # Every 5 seconds (assuming 0.1s steps)
                        self.log_simulation_state()
                    
                    step += 1
                    time.sleep(0.1)
                
                except traci.TraCIException as e:
                    print(f"Simulation error: {e}")
                    break
            
            time.sleep(0.1)  # Prevent high CPU usage when paused

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
            
            # Log traffic light states
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