"""
Synthetic Traffic Generator for SUMO Simulation
- Reads real-world vehicle data from CSV files
- Generates realistic traffic patterns for simulation
- Integrates with existing SUMO simulation framework
"""

import os
import random
import numpy as np
import pandas as pd
import traci
import time
from datetime import datetime, timedelta
import math
from pathlib import Path

class SyntheticTrafficGenerator:
    """Class to generate synthetic traffic based on real-world data"""
    
    def __init__(self, csv_path=None):
        """Initialize the traffic generator with CSV data"""
        self.vehicle_data = None
        self.current_index = 0
        self.csv_path = csv_path
        self.vehicle_count_history = []
        self.last_generation_time = 0
        self.generation_interval = 5  # Generate vehicles every 5 seconds
        self.trip_edges = []
        self.vehicle_types = {
            "slow": {"max_speed": 8.33, "color": (0, 0, 255, 255)},      # 30 km/h - Blue
            "normal": {"max_speed": 13.89, "color": (0, 255, 0, 255)},   # 50 km/h - Green
            "fast": {"max_speed": 19.44, "color": (255, 165, 0, 255)},   # 70 km/h - Orange
            "very_fast": {"max_speed": 27.78, "color": (255, 0, 0, 255)} # 100 km/h - Red
        }
        self.load_data()
        self.initialize_vehicle_types()
    
    def load_data(self):
        """Load and preprocess data from CSV file"""
        if not self.csv_path or not os.path.exists(self.csv_path):
            # Try to find the CSV in current directory
            possible_files = list(Path('.').glob('*.csv'))
            if possible_files:
                self.csv_path = str(possible_files[0])
                print(f"Using CSV file: {self.csv_path}")
            else:
                print("Warning: No CSV file found, using synthetic data only")
                return
        
        try:
            # Load the CSV data
            self.vehicle_data = pd.read_csv(self.csv_path)
            print(f"Loaded {len(self.vehicle_data)} records from {self.csv_path}")
            
            # Basic preprocessing
            # If there's a timestamp column, ensure it's in the right format
            if 'Transmitted_timestamp' in self.vehicle_data.columns:
                # Keep the timestamps as they are for now
                pass
                
            # Get speed data if available
            if 'Transmitted_speed (km/hr)' in self.vehicle_data.columns:
                # Convert km/h to m/s (SUMO uses m/s for speed)
                self.vehicle_data['speed_ms'] = self.vehicle_data['Transmitted_speed (km/hr)'] / 3.6
                
            # Create additional features for traffic generation
            self.prepare_traffic_features()
            
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            # Create dummy data for testing
            self.create_dummy_data()
    
    def prepare_traffic_features(self):
        """Prepare additional features for traffic generation"""
        # If we don't have specific columns we need, add calculated ones
        if 'vehicle_count' not in self.vehicle_data.columns:
            # Create a vehicle count based on speed patterns
            # Higher speeds often correlate with more vehicles on highways
            if 'speed_ms' in self.vehicle_data.columns:
                # Normalize speeds to a range and add some randomness
                speeds = self.vehicle_data['speed_ms'].values
                min_speed = speeds.min() if not np.isnan(speeds.min()) else 0
                max_speed = speeds.max() if not np.isnan(speeds.max()) else 30
                
                # Scale to a reasonable vehicle count (5-25)
                normalized_speeds = (speeds - min_speed) / (max_speed - min_speed + 0.001)
                base_counts = 5 + normalized_speeds * 20
                
                # Add some randomness
                random_factors = np.random.normal(1, 0.3, len(base_counts))
                vehicle_counts = np.clip(base_counts * random_factors, 1, 30).astype(int)
                self.vehicle_data['vehicle_count'] = vehicle_counts
            else:
                # If no speed data, create random counts between 5 and 25
                self.vehicle_data['vehicle_count'] = np.random.randint(5, 26, len(self.vehicle_data))
    
    def create_dummy_data(self):
        """Create dummy data if CSV loading fails"""
        # Create a simple dataset with mock speed and vehicle count data
        num_rows = 100
        timestamps = [datetime.now() + timedelta(minutes=i) for i in range(num_rows)]
        speeds_kmh = np.random.normal(60, 15, num_rows)  # Normal distribution around 60 km/h
        speeds_ms = speeds_kmh / 3.6  # Convert to m/s
        
        # Create wave pattern for vehicle counts (simulating rush hours)
        base_counts = 10 + 15 * np.sin(np.linspace(0, 4*np.pi, num_rows))
        random_factors = np.random.normal(1, 0.3, num_rows)
        vehicle_counts = np.clip(base_counts * random_factors, 1, 30).astype(int)
        
        # Create the dataframe
        self.vehicle_data = pd.DataFrame({
            'timestamp': timestamps,
            'speed_ms': speeds_ms,
            'vehicle_count': vehicle_counts
        })
        
        print("Using dummy data for simulation")
    
    def initialize_vehicle_types(self):
        """Initialize the vehicle types in SUMO"""
        try:
            for vtype, props in self.vehicle_types.items():
                try:
                    traci.vehicletype.copy("veh_passenger", f"synthetic_{vtype}")
                    traci.vehicletype.setColor(f"synthetic_{vtype}", props["color"])
                    traci.vehicletype.setMaxSpeed(f"synthetic_{vtype}", props["max_speed"])
                except traci.TraCIException as e:
                    print(f"Error setting up vehicle type {vtype}: {e}")
        except Exception as e:
            print(f"Failed to initialize vehicle types: {e}")
    
    def load_trip_edges(self):
        """Load valid trip edges from the network"""
        # Don't reload if we already have edges
        if self.trip_edges:
            return self.trip_edges
            
        # Try to load from existing network
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
            
            # Create pairs for source/destination
            for _ in range(min(30, len(valid_edges))):  # Create up to 30 sample trips
                if len(valid_edges) >= 2:
                    from_edge = random.choice(valid_edges)
                    to_edge = random.choice([e for e in valid_edges if e != from_edge])
                    self.trip_edges.append((from_edge, to_edge))
            
            if self.trip_edges:
                print(f"Generated {len(self.trip_edges)} routes for synthetic traffic")
                return self.trip_edges
        except Exception as e:
            print(f"Error generating edges: {e}")
        
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
        edge_ids = set(traci.edge.getIDList())
        for from_edge, to_edge in fallback_edges:
            if from_edge in edge_ids and to_edge in edge_ids:
                self.trip_edges.append((from_edge, to_edge))
        
        print(f"Using {len(self.trip_edges)} fallback routes")
        return self.trip_edges
    
    def select_vehicle_type(self, speed):
        """Select an appropriate vehicle type based on speed"""
        if speed < 10:  # < 36 km/h
            return "synthetic_slow"
        elif speed < 15:  # < 54 km/h
            return "synthetic_normal"
        elif speed < 20:  # < 72 km/h
            return "synthetic_fast"
        else:  # >= 72 km/h
            return "synthetic_very_fast"
    
    def get_current_vehicle_count(self):
        """Get the vehicle count for the current time from the data"""
        if self.vehicle_data is None or len(self.vehicle_data) == 0:
            return random.randint(5, 15)  # Return a random count if no data
        
        # Get the current record
        index = self.current_index % len(self.vehicle_data)
        record = self.vehicle_data.iloc[index]
        
        # Move to the next record for next time
        self.current_index += 1
        
        # Return the vehicle count
        if 'vehicle_count' in record:
            return int(record['vehicle_count'])
        return random.randint(5, 15)  # Fallback
    
    def get_current_vehicle_speed(self):
        """Get the vehicle speed for the current time from the data"""
        if self.vehicle_data is None or len(self.vehicle_data) == 0:
            return random.uniform(8, 20)  # Return a random speed if no data
        
        # Get the current record
        index = (self.current_index - 1) % len(self.vehicle_data)  # Use the same index as get_current_vehicle_count
        record = self.vehicle_data.iloc[index]
        
        # Return the speed in m/s
        if 'speed_ms' in record:
            return float(record['speed_ms'])
        return random.uniform(8, 20)  # Fallback
    
    def generate_vehicles(self):
        """Generate vehicles based on the current data"""
        # Check if it's time to generate vehicles
        current_time = traci.simulation.getTime()
        if current_time - self.last_generation_time < self.generation_interval:
            return 0  # Not time to generate yet
        
        self.last_generation_time = current_time
        
        # Get the number of vehicles to add
        target_count = self.get_current_vehicle_count()
        avg_speed = self.get_current_vehicle_speed()
        
        # Get or load trip edges
        edges = self.load_trip_edges()
        if not edges:
            print("No valid edges found for vehicle generation")
            return 0
        
        # Add the vehicles with retry logic
        added = 0
        max_attempts = target_count * 3  # Allow more attempts than requested vehicles
        attempt = 0
        
        # Keep track of this generation for history
        vehicle_batch = []
        
        while added < target_count and attempt < max_attempts:
            try:
                # Create a unique ID for this vehicle
                vehicle_id = f"synthetic_{int(current_time)}_{attempt}"
                
                # Randomly select a source and destination edge
                from_edge, to_edge = random.choice(edges)
                route_id = f"route_{vehicle_id}"
                
                # Add some variation to the speed
                speed_variation = random.uniform(0.8, 1.2)
                vehicle_speed = avg_speed * speed_variation
                
                # Select vehicle type based on speed
                veh_type = self.select_vehicle_type(vehicle_speed)
                
                # Create the route and add the vehicle
                traci.route.add(routeID=route_id, edges=[from_edge, to_edge])
                traci.vehicle.add(
                    vehID=vehicle_id, 
                    routeID=route_id, 
                    typeID=veh_type,
                    departLane="best", 
                    departSpeed="max"
                )
                
                # Set specific speed if needed
                traci.vehicle.setMaxSpeed(vehicle_id, vehicle_speed)
                
                # Record this vehicle
                vehicle_batch.append({
                    'id': vehicle_id,
                    'type': veh_type,
                    'speed': vehicle_speed,
                    'from_edge': from_edge,
                    'to_edge': to_edge,
                    'time': current_time
                })
                
                added += 1
            except traci.TraCIException as e:
                # Silently continue if there's an error
                pass
            
            attempt += 1
        
        # Add to history if any vehicles were added
        if added > 0:
            self.vehicle_count_history.append({
                'time': current_time,
                'count': added,
                'avg_speed': avg_speed,
                'vehicles': vehicle_batch
            })
            
            print(f"Generated {added} synthetic vehicles at time {current_time:.1f}s")
        
        return added
    
    def get_generation_stats(self):
        """Get statistics about the generated vehicles"""
        total_vehicles = sum(entry['count'] for entry in self.vehicle_count_history)
        
        if not self.vehicle_count_history:
            return {
                'total_generated': 0,
                'avg_per_interval': 0,
                'avg_speed': 0,
                'last_batch_size': 0,
                'last_generation_time': 0
            }
        
        # Calculate average vehicles per generation
        avg_vehicles = total_vehicles / len(self.vehicle_count_history)
        
        # Calculate average speed
        all_speeds = []
        for batch in self.vehicle_count_history:
            if 'avg_speed' in batch:
                all_speeds.append(batch['avg_speed'])
        
        avg_speed = sum(all_speeds) / len(all_speeds) if all_speeds else 0
        
        return {
            'total_generated': total_vehicles,
            'avg_per_interval': avg_vehicles,
            'avg_speed': avg_speed,
            'last_batch_size': self.vehicle_count_history[-1]['count'] if self.vehicle_count_history else 0,
            'last_generation_time': self.vehicle_count_history[-1]['time'] if self.vehicle_count_history else 0
        }

# This class can be integrated with the existing ModernTrafficGUI class
# by adding an instance variable and calling the generate_vehicles method
# in the simulation loop