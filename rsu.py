"""
Road Side Unit (RSU) implementation for traffic monitoring and data collection
"""

import traci
import math
from datetime import datetime

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

def create_circle_points(center, radius, num_points=36):
    """Create a circle of points around the center with the given radius"""
    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append((x, y))
    return points

def create_rsus(tl_ids):
    """Create RSUs at traffic light junctions with visual representation"""
    rsus = []
    
    for tl_id in tl_ids:
        try:
            junction_pos = traci.junction.getPosition(tl_id)
            rsu = RSU(tl_id, junction_pos)  # Use the traffic light ID directly
            rsus.append(rsu)
            
            # Add visual representation to the SUMO GUI map
            try:
                # Add a POI at the RSU location
                traci.poi.add(rsu.poi_id, junction_pos[0], junction_pos[1], 
                             (255, 0, 0, 255), "RSU", 20, 1)
                
                # Add a circle to show detection range
                traci.polygon.add(rsu.range_poi_id, 
                                 create_circle_points(junction_pos, rsu.radius), 
                                 (0, 0, 255, 80), fill=True, layer=-1)
                
                print(f"Created RSU at junction {tl_id} with visual POI")
            except traci.TraCIException as e:
                print(f"Error adding RSU POI for {tl_id}: {e}")
                
        except traci.TraCIException as e:
            print(f"Error creating RSU at junction {tl_id}: {e}")
    
    return rsus