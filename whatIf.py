import traci
import tkinter as tk
from tkinter import simpledialog, ttk
import threading
import time
import random

# Define trips and edge details for different scenarios
rush_hour_edges = [
    ("24242882#0", "15973619#6"),
    ("4611711", "-120675240#0"),
    ("4611693#0", "-1105574291#1"),
    # More edges can be added for dense rush hour traffic
]

rainy_day_edges = [
    ("147066248#1", "-120675240#0"),
    ("4611693#0", "243854725#1"),
    ("120675240#0", "68647306#5"),
    # More edges for a moderate number of vehicles with lower speed
]

foggy_morning_edges = [
    ("35557143#1", "-43231842#0"),
    ("-1233798019", "-24242882#0"),
    ("147066248#3", "-35557161#3"),
    # Edges with sparse traffic for lower visibility
]

# Start SUMO simulation with the osm.sumocfg file
sumoBinary = "sumo-gui"  # Use "sumo" for headless mode
sumoCmd = [sumoBinary, "-c", "osm.sumocfg"]
traci.start(sumoCmd)

# Tkinter setup for a simple GUI
root = tk.Tk()
root.title("OSM Map Simulation Control")

# Define a vehicle type for each scenario
def define_vehicle_types():
    traci.vehicletype.copy("veh_passenger", "rush_vehicle")
    traci.vehicletype.setColor("rush_vehicle", (255, 0, 0, 255))  # Red color for rush hour

    traci.vehicletype.copy("veh_passenger", "rain_vehicle")
    traci.vehicletype.setColor("rain_vehicle", (0, 0, 255, 255))  # Blue color for rainy day

    traci.vehicletype.copy("veh_passenger", "fog_vehicle")
    traci.vehicletype.setColor("fog_vehicle", (128, 128, 128, 255))  # Gray color for foggy morning

define_vehicle_types()

# Function to insert vehicles based on scenario
def insert_vehicles_for_scenario(scenario):
    edge_set = {
        "Rush Hour": rush_hour_edges,
        "Rainy Day": rainy_day_edges,
        "Foggy Morning": foggy_morning_edges,
    }.get(scenario, [])

    vehicle_type = {
        "Rush Hour": "rush_vehicle",
        "Rainy Day": "rain_vehicle",
        "Foggy Morning": "fog_vehicle",
    }.get(scenario, "veh_passenger")

    num_vehicles = 20 if scenario == "Rush Hour" else 10 if scenario == "Rainy Day" else 5

    for i in range(num_vehicles):
        vehicle_id = f"{vehicle_type}_{traci.simulation.getTime()}_{i}"
        try:
            from_edge, to_edge = random.choice(edge_set)
            route_id = f"route_{vehicle_id}"

            # Create a route and add vehicle based on the scenario
            traci.route.add(routeID=route_id, edges=[from_edge, to_edge])
            traci.vehicle.add(vehID=vehicle_id, routeID=route_id, typeID=vehicle_type, departLane="best", departSpeed="max")
            print(f"Inserted {scenario} vehicle {vehicle_id} on route from {from_edge} to {to_edge}")
        except traci.TraCIException as e:
            print(f"Error inserting vehicle {vehicle_id}: {e}")

# Function to handle scenario selection
def scenario_selected(event):
    scenario = scenario_dropdown.get()
    insert_vehicles_for_scenario(scenario)

# Function to stop the simulation
def stop_simulation():
    global exit_simulation
    exit_simulation = True
    print("Stopping simulation...")

# Define GUI components
scenario_label = tk.Label(root, text="Select Scenario:")
scenario_label.pack()

# Dropdown menu for selecting the scenario
scenario_dropdown = ttk.Combobox(root, values=["Rush Hour", "Rainy Day", "Foggy Morning"])
scenario_dropdown.bind("<<ComboboxSelected>>", scenario_selected)
scenario_dropdown.pack()

stop_button = tk.Button(root, text="Stop Simulation", command=stop_simulation)
stop_button.pack()

# Function to run the simulation in the background
def run_simulation():
    step = 0
    while not exit_simulation:
        if running:
            traci.simulationStep()
            step += 1
            time.sleep(0.1)
        
        root.update()

    traci.close()
    print("Simulation ended.")

# Start the simulation in a separate thread
exit_simulation = False
running = True
simulation_thread = threading.Thread(target=run_simulation)
simulation_thread.start()

# Start the Tkinter main loop
root.mainloop()
