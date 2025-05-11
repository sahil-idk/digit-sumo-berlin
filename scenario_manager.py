"""
Traffic scenario management for what-if analysis in traffic simulation
"""

import traci
import random
import xml.etree.ElementTree as ET

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
        self.initialize_vehicle_types()
        
    def initialize_vehicle_types(self):
        """Initialize the vehicle types used in scenarios"""
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
        """Load valid trip edges from the network with improved filtering"""
        # First try to load from pre-defined trips file
        try:
            tree = ET.parse('osm.passenger.trips.xml')
            root = tree.getroot()
            trip_edges = []
            for trip in root.findall('.//trip'):
                from_edge = trip.get('from')
                to_edge = trip.get('to')
                if from_edge and to_edge:
                    # Validate that edges exist in the network
                    try:
                        traci.edge.getIDList().index(from_edge)
                        traci.edge.getIDList().index(to_edge)
                        trip_edges.append((from_edge, to_edge))
                    except (ValueError, traci.TraCIException):
                        continue
                        
            if trip_edges:
                print(f"Loaded {len(trip_edges)} routes from trips file")
                return trip_edges
        except Exception as e:
            print(f"Error loading trip file: {e}")
        
        # If trips file failed, try to find valid edges dynamically
        try:
            # Get all valid edges (not internal)
            all_edges = traci.edge.getIDList()
            valid_edges = []
            
            # Filter for edges that allow vehicles to depart on them
            for edge in all_edges:
                if not edge.startswith(":") and len(traci.edge.getLanes(edge)) > 0:
                    # Check if the edge has at least one lane with valid length
                    for lane_idx in range(len(traci.edge.getLanes(edge))):
                        lane_id = f"{edge}_{lane_idx}"
                        try:
                            if traci.lane.getLength(lane_id) > 10:  # Reasonable minimum length
                                valid_edges.append(edge)
                                break
                        except traci.TraCIException:
                            continue
            
            # Create random pairs for source/destination
            trip_edges = []
            for _ in range(min(20, len(valid_edges))):  # Create up to 20 sample trips
                if len(valid_edges) >= 2:
                    from_edge = random.choice(valid_edges)
                    to_edge = random.choice([e for e in valid_edges if e != from_edge])
                    trip_edges.append((from_edge, to_edge))
            
            if trip_edges:
                print(f"Generated {len(trip_edges)} routes dynamically")
                return trip_edges
        except Exception as e:
            print(f"Error generating edges dynamically: {e}")
        
        # Fallback to hardcoded trip edges
        fallback_edges = [
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
        
        # Verify fallback edges exist in the network
        verified_edges = []
        edge_ids = set(traci.edge.getIDList())
        for from_edge, to_edge in fallback_edges:
            if from_edge in edge_ids and to_edge in edge_ids:
                verified_edges.append((from_edge, to_edge))
        
        if verified_edges:
            print(f"Using {len(verified_edges)} fallback routes")
            return verified_edges
        
        print("Warning: No valid routes found")
        return []
    
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
        
        # Insert vehicles with retry logic for failures
        added = 0
        max_attempts = adjusted_density * 3  # Allow for retries
        attempt = 0
        
        while added < adjusted_density and attempt < max_attempts:
            vehicle_id = f"{scenario_name.replace(' ', '_')}_{int(traci.simulation.getTime())}_{attempt}"
            try:
                from_edge, to_edge = random.choice(trip_edges)
                route_id = f"route_{vehicle_id}"
                traci.route.add(routeID=route_id, edges=[from_edge, to_edge])
                traci.vehicle.add(vehID=vehicle_id, routeID=route_id, typeID=vehicle_type,
                                 departLane="best", departSpeed="max")
                print(f"Added {scenario_name} vehicle {vehicle_id}")
                added += 1
            except traci.TraCIException as e:
                print(f"Error adding vehicle {vehicle_id}: {e}")
            
            attempt += 1
        
        print(f"Added {added}/{adjusted_density} vehicles for {scenario_name} scenario")
    
    def create_custom_scenario(self, vehicle_count, vehicle_type="custom", max_speed=13.9, color=(255, 0, 0, 255)):
        """Create a custom scenario with specified parameters"""
        # Create custom vehicle type
        custom_type = f"custom_{vehicle_type}_vehicle"
        try:
            # Register the custom vehicle type
            traci.vehicletype.copy("veh_passenger", custom_type)
            traci.vehicletype.setColor(custom_type, color)
            traci.vehicletype.setMaxSpeed(custom_type, max_speed)
        except traci.TraCIException as e:
            print(f"Error setting up custom vehicle type: {e}")
        
        # Add vehicles with retry logic
        trip_edges = self.load_trip_edges()
        if not trip_edges:
            print("No valid routes found")
            return 0
            
        added = 0
        max_attempts = vehicle_count * 3
        attempt = 0
        
        while added < vehicle_count and attempt < max_attempts:
            try:
                vehicle_id = f"custom_{int(traci.simulation.getTime())}_{attempt}"
                from_edge, to_edge = random.choice(trip_edges)
                route_id = f"route_{vehicle_id}"
                
                traci.route.add(routeID=route_id, edges=[from_edge, to_edge])
                traci.vehicle.add(vehID=vehicle_id, routeID=route_id, 
                                 typeID=custom_type, departLane="best", departSpeed="max")
                added += 1
            except traci.TraCIException as e:
                # Silently continue
                pass
            
            attempt += 1
        
        return added