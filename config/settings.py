"""
Configuration settings for the Smart Traffic Control System
"""

# SUMO configuration
SUMO_CONFIG = {
    "sumoBinary": "sumo-gui",  # Use 'sumo' for headless mode
    "sumoConfig": "osm.sumocfg",
    "randomTripsPath": "%SUMO_HOME%\\tools\\randomTrips.py",
}

# Application configuration
APP_CONFIG = {
    "title": "Smart Traffic Control System",
    "width": 1280,
    "height": 800,
    "min_width": 1024,
    "min_height": 768,
    "update_interval": 0.5,  # Update GUI every half second
}

# Simulation defaults
SIM_DEFAULTS = {
    "default_vehicle_count": 10,
    "default_intensity": 5,
    "max_data_points": 30,  # Maximum data points to keep in history
}

# RSU configuration
RSU_CONFIG = {
    "detection_radius": 100,  # detection radius in meters
    "congestion_thresholds": {
        "low": 5,       # Less than 5 vehicles
        "medium": 15,   # Between 5 and 15 vehicles
        "high": float('inf')  # More than 15 vehicles
    }
}

# Prediction configuration
PREDICTION_CONFIG = {
    "default_seq_length": 10,
    "default_horizon": 5,
    "default_data_source": "Current Simulation",
    "default_csv_path": "1.csv"
}