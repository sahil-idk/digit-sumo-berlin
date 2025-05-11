#!/usr/bin/env python
"""
Smart Traffic Control System - Main Application Entry Point
"""

import os
import sys
import traceback
from gui.application import TrafficSimulationApp

def main():
    """Main function to start the application"""
    try:
        # Check if SUMO_HOME environment variable is set
        if 'SUMO_HOME' not in os.environ:
            print("Please set SUMO_HOME environment variable")
            print("On Windows: set SUMO_HOME=C:\\path\\to\\sumo")
            print("On Linux: export SUMO_HOME=/path/to/sumo")
            sys.exit(1)
            
        # Start the GUI application
        app = TrafficSimulationApp()
        
        # Start the main loop
        app.run()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()