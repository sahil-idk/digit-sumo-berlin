import traci
import tkinter as tk
from tkinter import ttk, simpledialog
import threading
import time
import random

# Predefined valid trip edges from reference code
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
        self.sumoBinary = "sumo-gui"
        self.sumoCmd = [self.sumoBinary, "-c", "osm.sumocfg", "--device.btreceiver.all-recognitions", "--bt-output", "bt_output.xml"]
        
        try:
            traci.start(self.sumoCmd)
        except Exception as e:
            print(f"Error starting SUMO: {e}")
            return
        
        self.root = tk.Tk()
        self.root.title("SUMO Traffic & RSU Control")
        self.root.geometry("1000x800")
        
        self.running = False
        self.exit_simulation = False
        self.rsu_active = True
        
        self.tl_ids = traci.trafficlight.getIDList()
        self.rsus = {tl: f"rsu_{tl}" for tl in self.tl_ids}  # Assign RSUs to traffic lights
        
        self.setup_gui()
        
        self.simulation_thread = threading.Thread(target=self.run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        # Highlight RSUs on SUMO interface
        self.highlight_rsus()

    def setup_gui(self):
        self.control_frame = ttk.LabelFrame(self.root, text="Simulation Controls", padding="10")
        self.control_frame.pack(fill="x", padx=5, pady=5)
        
        self.rsu_frame = ttk.LabelFrame(self.root, text="RSU Control", padding="10")
        self.rsu_frame.pack(fill="x", padx=5, pady=5)
        
        self.vehicle_frame = ttk.LabelFrame(self.root, text="Vehicle Control", padding="10")
        self.vehicle_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(self.control_frame, text="Play", command=self.toggle_simulation).pack(side="left", padx=5)
        ttk.Button(self.control_frame, text="Stop", command=self.stop_simulation).pack(side="left", padx=5)
        
        self.rsu_toggle_btn = ttk.Button(self.rsu_frame, text="Deactivate RSU", command=self.toggle_rsu)
        self.rsu_toggle_btn.pack(side="left", padx=5)
        
        self.rsu_status_label = ttk.Label(self.rsu_frame, text="RSU Active: Yes")
        self.rsu_status_label.pack(side="left", padx=5)
        
        self.rsu_details_label = ttk.Label(self.rsu_frame, text="RSU Details: None")
        self.rsu_details_label.pack(side="left", padx=5)
        
        ttk.Button(self.vehicle_frame, text="Add Vehicle", command=self.add_vehicle).pack(side="left", padx=5)
    
    def toggle_simulation(self):
        self.running = not self.running
        if self.running:
            self.root.after(100, self.update_ui)

    def stop_simulation(self):
        self.exit_simulation = True
        self.running = False
        traci.close()
        self.root.quit()
        print("Simulation ended.")

    def toggle_rsu(self):
        self.rsu_active = not self.rsu_active
        status = "Yes" if self.rsu_active else "No"
        self.rsu_status_label.config(text=f"RSU Active: {status}")
        self.rsu_toggle_btn.config(text="Deactivate RSU" if self.rsu_active else "Activate RSU")

    def add_vehicle(self):
        vehicle_id = f"veh{random.randint(1000, 9999)}"
        try:
            from_edge, to_edge = random.choice(trip_edges)
            route_id = f"route_{vehicle_id}"
            traci.route.add(routeID=route_id, edges=[from_edge, to_edge])
            traci.vehicle.add(vehID=vehicle_id, routeID=route_id, departLane="best", departSpeed="max")
            print(f"Added vehicle: {vehicle_id} from {from_edge} to {to_edge}")
        except Exception as e:
            print(f"Error adding vehicle: {e}")

    def highlight_rsus(self):
        """Focus SUMO view on RSU locations"""
        for tl_id in self.tl_ids:
            try:
                traci.gui.trackVehicle("View #0", tl_id)
            except Exception as e:
                print(f"Error highlighting RSU at {tl_id}: {e}")

    def run_simulation(self):
        while not self.exit_simulation:
            if self.running:
                try:
                    traci.simulationStep()
                    if self.rsu_active:
                        self.handle_rsu()
                except Exception as e:
                    print(f"Error during simulation step: {e}")
                time.sleep(0.1)
    
    def handle_rsu(self):
        try:
            detected_info = []
            for tl_id, rsu_id in self.rsus.items():
                rsu_pos = traci.junction.getPosition(tl_id)
                detected_vehicles = [veh for veh in traci.vehicle.getIDList()
                                     if ((rsu_pos[0] - traci.vehicle.getPosition(veh)[0])**2 +
                                         (rsu_pos[1] - traci.vehicle.getPosition(veh)[1])**2) ** 0.5 <= 50]
                
                if detected_vehicles:
                    print(f"RSU at {tl_id} detected vehicles: {detected_vehicles}")
                    detected_info.append(f"{tl_id}: {', '.join(detected_vehicles)}")
                    for veh in detected_vehicles:
                        traci.vehicle.setSpeed(veh, random.uniform(5, 15))
                    
            rsu_details = " | ".join(detected_info) if detected_info else "None"
            self.rsu_details_label.config(text=f"RSU Details: {rsu_details}")
        except Exception as e:
            print(f"Error in RSU handling: {e}")

    def update_ui(self):
        if self.running:
            self.root.after(100, self.update_ui)

    def start(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = TrafficSimulationGUI()
    app.start()
