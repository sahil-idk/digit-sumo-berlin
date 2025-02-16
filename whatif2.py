import traci
import tkinter as tk
from tkinter import simpledialog, ttk
import threading
import time
import random

# Define trips and scenarios
trip_edges = [
    ("24242882#0", "15973619#6"),
    ("4611711", "-120675240#0"),
    # Add more trip edges as necessary
]

# Define scenarios with unique vehicle types and colors
scenarios = {
    "Normal Traffic": {"density": 1, "vehicle_type": "normal_vehicle", "color": (0, 255, 0, 255)},  # Green for normal
    "Rush Hour": {"density": 10, "vehicle_type": "rush_vehicle", "color": (255, 0, 0, 255)},  # Red for rush hour
    "Rainy Day": {"density": 3, "vehicle_type": "rain_vehicle", "color": (0, 0, 255, 255)},  # Blue for rainy day
    "Foggy Morning": {"density": 5, "vehicle_type": "fog_vehicle", "color": (128, 128, 128, 255)},  # Gray for fog
}

# Start SUMO simulation
sumoBinary = "sumo-gui"
sumoCmd = [sumoBinary, "-c", "osm.sumocfg"]
traci.start(sumoCmd)

# Tkinter GUI setup
root = tk.Tk()
root.title("OSM Map Simulation Control")

# Define each scenario's vehicle type and color
for scenario, config in scenarios.items():
    traci.vehicletype.copy("veh_passenger", config["vehicle_type"])
    traci.vehicletype.setColor(config["vehicle_type"], config["color"])

running = False
exit_simulation = False

# Dropdown menu for selecting scenarios
selected_scenario = tk.StringVar(value="Normal Traffic")
scenario_menu = ttk.Combobox(root, textvariable=selected_scenario, values=list(scenarios.keys()))
scenario_menu.pack()

# Intensity slider to adjust vehicle density dynamically
intensity_label = tk.Label(root, text="Adjust Intensity")
intensity_label.pack()
intensity_slider = tk.Scale(root, from_=1, to=20, orient="horizontal")
intensity_slider.pack()

# Function to update density based on slider and scenario
def update_density():
    scenario = selected_scenario.get()
    base_density = scenarios[scenario]["density"]
    adjusted_density = base_density * intensity_slider.get()
    return adjusted_density

# Function to insert vehicles for the current scenario
def insert_vehicles(adjusted_density):
    scenario = selected_scenario.get()
    vehicle_type = scenarios[scenario]["vehicle_type"]

    for i in range(int(adjusted_density)):
        vehicle_id = f"{scenario}_vehicle_{traci.simulation.getTime()}_{i}"
        try:
            from_edge, to_edge = random.choice(trip_edges)
            route_id = f"route_{vehicle_id}"
            traci.route.add(routeID=route_id, edges=[from_edge, to_edge])
            traci.vehicle.add(vehID=vehicle_id, routeID=route_id, typeID=vehicle_type, departLane="best", departSpeed="max")
            print(f"Inserted {scenario} vehicle {vehicle_id} on route from {from_edge} to {to_edge}")
        except traci.TraCIException as e:
            print(f"Error inserting vehicle {vehicle_id}: {e}")

# Function to apply the selected scenario with adjusted density
def apply_scenario():
    density = update_density()
    insert_vehicles(density)

apply_button = tk.Button(root, text="Apply Scenario", command=apply_scenario)
apply_button.pack()

# Simulation control functions
def toggle_play_pause():
    global running
    running = not running
    play_pause_button.config(text="Pause" if running else "Play")

def stop_simulation():
    global exit_simulation
    exit_simulation = True

play_pause_button = tk.Button(root, text="Play", command=toggle_play_pause)
play_pause_button.pack()
stop_button = tk.Button(root, text="Stop Simulation", command=stop_simulation)
stop_button.pack()

# Simulation thread to run in the background
def run_simulation():
    step = 0
    while not exit_simulation:
        if running:
            traci.simulationStep()
            step += 1
            time.sleep(0.1)
        root.update()
    traci.close()

simulation_thread = threading.Thread(target=run_simulation)
simulation_thread.start()
root.mainloop()
