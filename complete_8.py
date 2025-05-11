"""
Smart Traffic Control System - Main Application
Integrates all components for a modular traffic simulation system with ML prediction
"""

import os
import sys
import traceback
import tkinter as tk
from tkinter import messagebox

# Check if SUMO_HOME is set
if 'SUMO_HOME' not in os.environ:
    print("Please set the SUMO_HOME environment variable")
    print("On Windows: set SUMO_HOME=C:\\path\\to\\sumo")
    print("On Linux: export SUMO_HOME=/path/to/sumo")
    sys.exit(1)

# Add SUMO to the path
if os.path.exists(os.environ['SUMO_HOME']):
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    print(f"SUMO_HOME is set, but the directory {os.environ['SUMO_HOME']} does not exist")
    sys.exit(1)

# Now we can import traci after setting up the path
try:
    import traci
except ImportError:
    print("Could not import traci. Make sure SUMO is installed correctly.")
    sys.exit(1)

# Import our modules
try:
    from traffic_gui import ModernTrafficGUI
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required files are in the same directory:")
    print("- gui_components.py")
    print("- rsu.py")
    print("- scenario_manager.py")
    print("- traffic_prediction.py")
    print("- traffic_gui.py")
    sys.exit(1)

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing_packages = []
    
    # Check numpy
    try:
        import numpy
    except ImportError:
        missing_packages.append("numpy")
    
    # Check pandas
    try:
        import pandas
    except ImportError:
        missing_packages.append("pandas")
    
    # Check matplotlib
    try:
        import matplotlib
    except ImportError:
        missing_packages.append("matplotlib")
    
    # Check PIL/Pillow (special case as it's imported as PIL)
    try:
        from PIL import Image, ImageTk
    except ImportError:
        missing_packages.append("pillow")
    
    if missing_packages:
        print("The following packages are required but not installed:")
        for package in missing_packages:
            print(f"- {package}")
        print("\nPlease install them using pip:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True
def check_sumo_config():
    """Check if the required SUMO configuration files exist"""
    required_files = ["osm.sumocfg"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("The following SUMO configuration files are missing:")
        for file in missing_files:
            print(f"- {file}")
        return False
    
    return True

def main():
    """Main entry point for the application"""
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check SUMO config files
    if not check_sumo_config():
        print("Warning: Some SUMO configuration files are missing.")
        result = input("Do you want to continue anyway? (y/n): ")
        if result.lower() != 'y':
            sys.exit(1)
    
    # Create and start the application
    try:
        # Create a splash window to show during initialization
        splash_root = tk.Tk()
        splash_root.overrideredirect(True)  # No window decorations
        splash_root.geometry("400x200+{}+{}".format(
            int(splash_root.winfo_screenwidth()/2 - 200),
            int(splash_root.winfo_screenheight()/2 - 100)
        ))
        splash_root.configure(bg="#2c3e50")
        
        # Add title and progress message
        tk.Label(
            splash_root, 
            text="Smart Traffic Control System",
            font=("Segoe UI", 18, "bold"),
            fg="white",
            bg="#2c3e50"
        ).pack(pady=(50, 10))
        
        progress_var = tk.StringVar(value="Starting SUMO and initializing components...")
        tk.Label(
            splash_root, 
            textvariable=progress_var,
            font=("Segoe UI", 10),
            fg="#ecf0f1",
            bg="#2c3e50"
        ).pack(pady=10)
        
        splash_root.update()
        
        # Initialize the GUI
        try:
            app = ModernTrafficGUI()
            
            # Close the splash screen
            splash_root.destroy()
            
            # Start the main GUI
            app.root.mainloop()
            
        except Exception as e:
            # Close splash screen in case of error
            splash_root.destroy()
            
            # Show error dialog
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            
            error_message = f"Error starting the application:\n\n{str(e)}\n\n"
            error_message += "See the console for more details."
            
            messagebox.showerror("Error", error_message)
            
            # Print detailed error to console
            print(f"Error starting application: {e}")
            traceback.print_exc()
            
            sys.exit(1)
    
    except Exception as e:
        print(f"Error initializing application: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()