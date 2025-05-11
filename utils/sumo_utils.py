"""
SUMO utility functions for the Smart Traffic Control System
"""

import os
import subprocess
import traci
from config.settings import SUMO_CONFIG

def start_sumo():
    """Start the SUMO simulation"""
    try:
        sumo_binary = SUMO_CONFIG["sumoBinary"]
        sumo_config = SUMO_CONFIG["sumoConfig"]
        sumo_cmd = [sumo_binary, "-c", sumo_config]
        
        traci.start(sumo_cmd)
        print("SUMO started successfully")
        return True
    except Exception as e:
        print(f"Error starting SUMO: {e}")
        return False

def close_sumo():
    """Close the SUMO connection"""
    try:
        traci.close()
        print("SUMO connection closed")
        return True
    except Exception as e:
        print(f"Error closing SUMO connection: {e}")
        return False

def execute_build_script():
    """Execute the build script to generate routes"""
    try:
        # Format the randomTrips.py command
        random_trips_path = SUMO_CONFIG["randomTripsPath"]
        
        # Build the command
        cmd = [
            "python", random_trips_path,
            "-n", "osm.net.xml.gz",
            "--fringe-factor", "5",
            "--insertion-density", "12",
            "-o", "osm.passenger.trips.xml",
            "-r", "osm.passenger.rou.xml",
            "-b", "0",
            "-e", "3600",
            "--trip-attributes", "departLane=\"best\"",
            "--fringe-start-attributes", "departSpeed=\"max\"",
            "--validate",
            "--remove-loops",
            "--via-edge-types", "highway.motorway,highway.motorway_link,highway.trunk_link,highway.primary_link,highway.secondary_link,highway.tertiary_link",
            "--vehicle-class", "passenger",
            "--vclass", "passenger",
            "--prefix", "veh",
            "--min-distance", "300",
            "--min-distance.fringe", "10",
            "--allow-fringe.min-length", "1000",
            "--lanes"
        ]
        
        # Execute the command
        subprocess.run(cmd, check=True)
        print("Route generation successful")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing build script: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during route generation: {e}")
        return False

def configure_sumo_gui():
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
        
        print("SUMO GUI configured successfully")
        return True
    except traci.TraCIException as e:
        print(f"Warning: Could not configure SUMO GUI: {e}")
        return False

def add_vehicle(vehicle_id, from_edge, to_edge, vtype="veh_passenger"):
    """Add a vehicle to the simulation"""
    try:
        route_id = f"route_{vehicle_id}"
        traci.route.add(routeID=route_id, edges=[from_edge, to_edge])
        traci.vehicle.add(
            vehID=vehicle_id, 
            routeID=route_id, 
            typeID=vtype,
            departLane="best", 
            departSpeed="max"
        )
        return True
    except traci.TraCIException as e:
        print(f"Error adding vehicle {vehicle_id}: {e}")
        return False

def set_traffic_light_phase(tl_id, phase):
    """Set the traffic light phase"""
    try:
        # Get the current state to determine length
        current_state = traci.trafficlight.getRedYellowGreenState(tl_id)
        state_len = len(current_state)
        
        # Create new state string based on the requested phase
        if phase.lower() == "red":
            new_state = "r" * state_len
        elif phase.lower() == "yellow":
            new_state = "y" * state_len
        else:  # green
            new_state = "G" * state_len
            
        # Apply the new state
        traci.trafficlight.setRedYellowGreenState(tl_id, new_state)
        return True
    except traci.TraCIException as e:
        print(f"Failed to control traffic light: {e}")
        return False

def get_simulation_stats():
    """Get current simulation statistics"""
    try:
        # Get all vehicles
        vehicles = traci.vehicle.getIDList()
        vehicle_count = len(vehicles)
        
        # Calculate average speed
        avg_speed = 0
        if vehicle_count > 0:
            speeds = [traci.vehicle.getSpeed(veh) for veh in vehicles]
            avg_speed = sum(speeds) / vehicle_count if speeds else 0
        
        # Get simulation time
        sim_time = traci.simulation.getTime()
        
        return {
            "vehicle_count": vehicle_count,
            "avg_speed": avg_speed,
            "sim_time": sim_time
        }
    except traci.TraCIException as e:
        print(f"Error getting simulation stats: {e}")
        return {
            "vehicle_count": 0,
            "avg_speed": 0,
            "sim_time": 0
        }