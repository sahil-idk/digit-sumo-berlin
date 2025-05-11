# gui/tabs/scenario_tab.py
"""
Traffic scenarios tab implementation for the Smart Traffic Control System
"""

import tkinter as tk
from tkinter import ttk, messagebox, colorchooser
from config.styles import COLORS
from gui.components.dashboard_panel import DashboardPanel
from gui.components.stylish_button import StylishButton
from config.settings import SIM_DEFAULTS

class ScenarioTab:
    """Traffic scenarios tab for configuring what-if analyses"""
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        
        # Split into two columns
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # Setup predefined scenario panel
        self.setup_scenario_panel()
        
        # Setup custom scenario panel
        self.setup_custom_panel()
        
    def setup_scenario_panel(self):
        """Setup the predefined scenario panel"""
        scenario_panel = DashboardPanel(self.parent, title="Traffic Scenarios")
        scenario_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)
        
        # Scenario options frame
        options_frame = tk.Frame(scenario_panel.content, bg="white")
        options_frame.pack(fill="both", expand=True, pady=10)
        
        # Scenario selection
        tk.Label(options_frame, text="Select Scenario:", bg="white", 
                fg=COLORS["text"], font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 5))
        
        # Use a combobox for scenario selection
        self.scenario_var = tk.StringVar(value="Normal Traffic")
        scenarios = list(self.app.scenario_manager.scenarios.keys())
        
        self.scenario_cb = ttk.Combobox(options_frame, 
                                   textvariable=self.scenario_var,
                                   values=scenarios,
                                   font=("Segoe UI", 10),
                                   width=25,
                                   state="readonly")
        self.scenario_cb.pack(anchor="w", pady=(0, 15))
        self.scenario_cb.bind("<<ComboboxSelected>>", self.on_scenario_selected)
        
        # Intensity slider
        tk.Label(options_frame, text="Traffic Intensity:", bg="white", 
                fg=COLORS["text"], font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 5))
        
        # Slider with value indicator
        slider_frame = tk.Frame(options_frame, bg="white")
        slider_frame.pack(fill="x", pady=(0, 15))
        
        self.intensity_var = tk.IntVar(value=SIM_DEFAULTS["default_intensity"])
        self.intensity_slider = ttk.Scale(slider_frame, from_=1, to=20, 
                                    orient="horizontal", 
                                    variable=self.intensity_var,
                                    length=200,
                                    command=self.update_intensity_label)
        self.intensity_slider.pack(side="left")
        
        self.intensity_label = tk.Label(slider_frame, text=str(SIM_DEFAULTS["default_intensity"]), bg="white", 
                                       fg=COLORS["text"], width=2, font=("Segoe UI", 10))
        self.intensity_label.pack(side="left", padx=5)
        
        # Create a separator
        ttk.Separator(options_frame, orient="horizontal").pack(fill="x", pady=15)
        
        # Apply button
        StylishButton(options_frame, text="Apply Scenario", color="primary",
                     command=self.apply_selected_scenario).pack(anchor="w", pady=10)
        
        # Description area
        desc_frame = tk.LabelFrame(options_frame, text="Scenario Description", 
                                  bg="white", fg=COLORS["text"], padx=10, pady=10)
        desc_frame.pack(fill="both", expand=True, pady=10)
        
        self.scenario_desc = tk.Text(desc_frame, height=8, wrap="word", 
                                    font=("Segoe UI", 10), bg="white", fg=COLORS["text"],
                                    borderwidth=0, highlightthickness=0)
        self.scenario_desc.pack(fill="both", expand=True)
        self.scenario_desc.config(state="disabled")
        
        # Initial description update
        self.update_scenario_description()
        
    def setup_custom_panel(self):
        """Setup the custom scenario panel"""
        custom_panel = DashboardPanel(self.parent, title="Custom Scenario Builder")
        custom_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)
        
        # Custom scenario options
        custom_frame = tk.Frame(custom_panel.content, bg="white")
        custom_frame.pack(fill="both", expand=True, pady=10)
        
        # Vehicle count
        tk.Label(custom_frame, text="Number of Vehicles:", bg="white", 
                fg=COLORS["text"], font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 5))
        
        self.vehicle_count_var = tk.IntVar(value=SIM_DEFAULTS["default_vehicle_count"])
        
        count_frame = tk.Frame(custom_frame, bg="white")
        count_frame.pack(anchor="w", pady=(0, 15))
        
        count_entry = ttk.Entry(count_frame, width=5, textvariable=self.vehicle_count_var)
        count_entry.pack(side="left")
        
        # Add plus/minus buttons
        StylishButton(count_frame, text="-", color="dark", width=2,
                     command=lambda: self.vehicle_count_var.set(max(1, self.vehicle_count_var.get() - 1))).pack(side="left", padx=5)
        StylishButton(count_frame, text="+", color="dark", width=2,
                     command=lambda: self.vehicle_count_var.set(self.vehicle_count_var.get() + 1)).pack(side="left")
        
        # Vehicle type
        tk.Label(custom_frame, text="Vehicle Type:", bg="white", 
                fg=COLORS["text"], font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 5))
        
        # Radio buttons for vehicle types
        type_frame = tk.Frame(custom_frame, bg="white")
        type_frame.pack(anchor="w", pady=(0, 15))
        
        self.vehicle_type_var = tk.StringVar(value="normal")
        
        types = [
            ("Normal", "normal"),
            ("Heavy", "heavy"),
            ("Fast", "fast"),
            ("Emergency", "emergency")
        ]
        
        for text, value in types:
            rb = ttk.Radiobutton(type_frame, text=text, value=value, variable=self.vehicle_type_var)
            rb.pack(side="left", padx=10)
        
        # Maximum speed
        tk.Label(custom_frame, text="Max Speed (m/s):", bg="white", 
                fg=COLORS["text"], font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 5))
        
        self.max_speed_var = tk.DoubleVar(value=13.9)  # ~50 km/h
        
        speed_frame = tk.Frame(custom_frame, bg="white")
        speed_frame.pack(anchor="w", pady=(0, 15))
        
        speed_entry = ttk.Entry(speed_frame, width=5, textvariable=self.max_speed_var)
        speed_entry.pack(side="left")
        
        # Add tooltip or explanation
        tk.Label(speed_frame, text="(50 km/h ≈ 13.9 m/s)", 
                fg="#777", bg="white", font=("Segoe UI", 9)).pack(side="left", padx=10)
        
        # Vehicle color
        tk.Label(custom_frame, text="Vehicle Color:", bg="white", 
                fg=COLORS["text"], font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 5))
        
        self.color_var = tk.StringVar(value="#FF0000")  # Red by default
        
        color_frame = tk.Frame(custom_frame, bg="white")
        color_frame.pack(anchor="w", pady=(0, 15))
        
        # Color preview box
        self.color_preview = tk.Frame(color_frame, width=20, height=20, bg=self.color_var.get())
        self.color_preview.pack(side="left")
        self.color_preview.pack_propagate(False)
        
        # Color picker button
        StylishButton(color_frame, text="Pick Color", color="primary",
                     command=self.pick_color).pack(side="left", padx=10)
        
        # Create a separator
        ttk.Separator(custom_frame, orient="horizontal").pack(fill="x", pady=15)
        
        # Apply button
        StylishButton(custom_frame, text="Generate Custom Scenario", color="primary", 
                     command=self.apply_custom_scenario).pack(anchor="w", pady=10)
                     
    def update_intensity_label(self, *args):
        """Update the intensity label when the slider changes"""
        self.intensity_label.config(text=str(self.intensity_var.get()))
    
    def on_scenario_selected(self, event):
        """Handle scenario selection change"""
        self.update_scenario_description()
    
    def update_scenario_description(self):
        """Update the scenario description based on selection"""
        scenario = self.scenario_var.get()
        
        description = self.app.scenario_manager.get_scenario_description(
            scenario, self.intensity_var.get())
            
        # Update the description text
        self.scenario_desc.config(state="normal")
        self.scenario_desc.delete(1.0, tk.END)
        self.scenario_desc.insert(tk.END, description)
        self.scenario_desc.config(state="disabled")
    
    def pick_color(self):
        """Open a color picker dialog"""
        color = colorchooser.askcolor(initialcolor=self.color_var.get())[1]
        if color:
            self.color_var.set(color)
            self.color_preview.config(bg=color)
    
    def apply_selected_scenario(self):
        """Apply the currently selected scenario"""
        scenario = self.scenario_var.get()
        intensity = self.intensity_var.get()
        
        result = self.app.scenario_manager.apply_scenario(scenario, intensity)
        
        if result:
            messagebox.showinfo("Scenario Applied", f"Applied {scenario} scenario with intensity {intensity}")
        else:
            messagebox.showerror("Error", "Failed to apply scenario")
    
    def apply_custom_scenario(self):
        """Apply a custom scenario"""
        try:
            vehicle_count = self.vehicle_count_var.get()
            max_speed = self.max_speed_var.get()
            vehicle_type = self.vehicle_type_var.get()
            color = self.color_var.get()
            
            # Convert hex color to RGBA tuple
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            rgba = (r, g, b, 255)
            
            result = self.app.scenario_manager.apply_custom_scenario(
                vehicle_count, vehicle_type, max_speed, rgba)
                
            if result:
                messagebox.showinfo("Custom Scenario", 
                                   f"Added {vehicle_count} custom vehicles with max speed {max_speed:.1f} m/s")
            else:
                messagebox.showerror("Error", "Failed to apply custom scenario")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply custom scenario: {e}")
            
    def update(self):
        """Update the scenario tab (not much to update here)"""
        pass