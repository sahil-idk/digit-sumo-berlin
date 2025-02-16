import traci
import tkinter as tk
from tkinter import simpledialog
import threading
import time
import random

# List of trips with (from_edge, to_edge) pairs
# predict and feed can be fed here 
# for example, predict = [from_edge, to_edge]
# feed = [from_edge, to_edge]
# trip_edges = [predict, feed]

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
    ("147066248#3", "-35557161#3"),
    # Add more trip edges as necessary
]

# Start SUMO simulation with the osm.sumocfg file
sumoBinary = "sumo-gui"  # Use "sumo" for headless mode
sumoCmd = [sumoBinary, "-c", "osm.sumocfg"]  # Reference to your osm.sumocfg file
traci.start(sumoCmd)

# Tkinter setup for a simple GUI
root = tk.Tk()
root.title("OSM Map Simulation Control")

# Define the red vehicle type
traci.vehicletype.copy("veh_passenger", "red_passenger")  # Copy the default vehicle type
traci.vehicletype.setColor("red_passenger", (255, 0, 0, 255))  # Set color to red (RGBA)

# Flags to control simulation state
running = False  # Starts paused
exit_simulation = False

# Function to insert more vehicles dynamically
def insert_vehicle():
    num_vehicles = simpledialog.askinteger("Input", "Enter number of vehicles to insert:", parent=root)
    if num_vehicles is None or num_vehicles < 1:
        return

    # Insert vehicles onto valid edges and create dynamic routes
    for i in range(num_vehicles):
        vehicle_id = f"red_vehicle_{traci.simulation.getTime()}_{i}"
        try:
            # Randomly choose a trip (from_edge, to_edge)
            from_edge, to_edge = random.choice(trip_edges)
            route_id = f"route_{vehicle_id}"
            # Create a route for the vehicle dynamically called from a model rather than reading from a csv 
            # This is where the predict and feed can be used to create the route
            # Create a route for the vehicle dynamically
            traci.route.add(routeID=route_id, edges=[from_edge, to_edge])
            # Add the red vehicle
            traci.vehicle.add(vehID=vehicle_id, routeID=route_id, typeID="red_passenger", departLane="best", departSpeed="max")
            print(f"Inserted red vehicle {vehicle_id} on route from {from_edge} to {to_edge}")
        except traci.TraCIException as e:
            print(f"Error inserting vehicle {vehicle_id}: {e}")

# Function to adjust the speed of all vehicles
def adjust_vehicle_speed():
    speed_factor = simpledialog.askfloat("Input", "Enter speed multiplier (e.g., 1.2 to increase speed by 20%):", parent=root)
    if speed_factor is None or speed_factor <= 0:
        return

    # Get all vehicles in the simulation and adjust their speed
    vehicle_ids = traci.vehicle.getIDList()
    for vehicle_id in vehicle_ids:
        try:
            current_speed = traci.vehicle.getSpeed(vehicle_id)
            new_speed = current_speed * speed_factor
            traci.vehicle.setSpeed(vehicle_id, new_speed)
            print(f"Adjusted speed of {vehicle_id} to {new_speed}")
        except traci.TraCIException as e:
            print(f"Error adjusting speed for {vehicle_id}: {e}")

# Function to toggle play/pause
def toggle_play_pause():
    global running
    running = not running
    play_pause_button.config(text="Pause" if running else "Play")
    print("Simulation resumed" if running else "Simulation paused")

# Function to stop the simulation
def stop_simulation():
    global exit_simulation
    exit_simulation = True
    print("Stopping simulation...")

# Create GUI components
insert_button = tk.Button(root, text="Insert Red Vehicles", command=insert_vehicle)
insert_button.pack()
adjust_speed_button = tk.Button(root, text="Adjust Speed of All Vehicles", command=adjust_vehicle_speed)
adjust_speed_button.pack()
play_pause_button = tk.Button(root, text="Play", command=toggle_play_pause)
play_pause_button.pack()
stop_button = tk.Button(root, text="Stop Simulation", command=stop_simulation)
stop_button.pack()

# Function to log vehicle counts at each node every 5 seconds
def log_vehicle_counts(step):
    if step % 50 == 0:  # Assuming each step is 0.1 seconds, 50 steps = 5 seconds
        edges = traci.edge.getIDList()  # Log vehicle counts across all edges in OSM map
        for edge in edges:
            vehicle_count = traci.edge.getLastStepVehicleNumber(edge)
            print(f"At step {step}, {vehicle_count} vehicles on edge {edge}")

# Run simulation in the background
def run_simulation():
    step = 0
    while not exit_simulation:
        if running:
            traci.simulationStep()  # Advance simulation step
            
            # Log vehicle counts every 5 seconds
            log_vehicle_counts(step)
            
            step += 1
            time.sleep(0.1)  # Small delay to control simulation speed
        
        root.update()  # Keep GUI responsive

    traci.close()
    print("Simulation ended.")

# Start the simulation in a separate thread
simulation_thread = threading.Thread(target=run_simulation)
simulation_thread.start()

# Start the Tkinter main loop
root.mainloop()
