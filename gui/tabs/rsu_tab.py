# gui/tabs/rsu_tab.py
"""
RSU monitoring tab implementation for the Smart Traffic Control System
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from config.styles import COLORS
from gui.components.dashboard_panel import DashboardPanel
from utils.sumo_utils import set_traffic_light_phase

class RSUTab:
    """RSU monitoring tab with detailed RSU information"""
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.current_rsu_id = None
        self.rsu_chart = None
        
        # Split into two panels
        parent.columnconfigure(0, weight=2)
        parent.columnconfigure(1, weight=3)
        parent.rowconfigure(0, weight=1)
        
        # Setup RSU list panel
        self.setup_rsu_list()
        
        # Setup RSU details panel
        self.setup_rsu_details()
        
    def setup_rsu_list(self):
        """Setup the RSU list panel with treeview"""
        left_panel = DashboardPanel(self.parent, title="RSU Network")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)
        
        # Create RSU treeview with details
        columns = ("ID", "Location", "Vehicles", "Congestion", "Avg Speed")
        self.rsu_tree = ttk.Treeview(
            left_panel.content, 
            columns=columns, 
            show="headings", 
            style="Treeview"
        )
        
        # Configure columns
        for col in columns:
            self.rsu_tree.heading(col, text=col)
            width = 100 if col == "Location" else 80
            self.rsu_tree.column(col, width=width)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(left_panel.content, orient="vertical", command=self.rsu_tree.yview)
        self.rsu_tree.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        self.rsu_tree.pack(fill="both", expand=True)
        
        # Bind selection event
        self.rsu_tree.bind("<<TreeviewSelect>>", self.on_rsu_selected)
        
    def setup_rsu_details(self):
        """Setup the RSU details panel"""
        self.right_panel = DashboardPanel(self.parent, title="RSU Details")
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)
        
        # RSU details container
        self.details_container = tk.Frame(self.right_panel.content, bg="white")
        self.details_container.pack(fill="both", expand=True)
        
        # Create the initial "select an RSU" message
        tk.Label(
            self.details_container,
            text="Select an RSU from the list to view details",
            font=("Segoe UI", 12),
            fg="#777",
            bg="white"
        ).pack(expand=True)
    
    def on_rsu_selected(self, event):
        """Handle RSU selection from the treeview"""
        selection = self.rsu_tree.selection()
        if not selection:
            return
            
        # Get the selected RSU
        item = self.rsu_tree.item(selection[0])
        rsu_id = item['values'][0]
        
        # Don't redraw if it's the same RSU
        if rsu_id == self.current_rsu_id:
            return
            
        self.current_rsu_id = rsu_id
        self.update_rsu_details(rsu_id)
    
    def update_rsu_details(self, rsu_id):
        """Update the RSU details panel with information about the selected RSU"""
        # Find the RSU object
        selected_rsu = None
        for rsu in self.app.rsus:
            if rsu.id == rsu_id:
                selected_rsu = rsu
                break
                
        if not selected_rsu:
            return
            
        # Clear current contents
        for widget in self.details_container.winfo_children():
            widget.destroy()
            
        # Create new details view
        details_frame = tk.Frame(self.details_container, bg="white")
        details_frame.pack(fill="both", expand=True)
        
        # Header with RSU ID and status
        header = tk.Frame(details_frame, bg="white", pady=10)
        header.pack(fill="x")
        
        # RSU Icon and ID
        tk.Label(
            header,
            text="📡",
            font=("Segoe UI", 24),
            fg=COLORS["primary"],
            bg="white"
        ).pack(side="left", padx=(0, 10))
        
        id_frame = tk.Frame(header, bg="white")
        id_frame.pack(side="left")
        
        tk.Label(
            id_frame,
            text=rsu_id,
            font=("Segoe UI", 14, "bold"),
            fg=COLORS["text"],
            bg="white"
        ).pack(anchor="w")
        
        tk.Label(
            id_frame,
            text=f"Location: ({selected_rsu.position[0]:.1f}, {selected_rsu.position[1]:.1f})",
            font=("Segoe UI", 10),
            fg="#777",
            bg="white"
        ).pack(anchor="w")
        
        # Status indicator
        status_color = COLORS["secondary"] if selected_rsu.congestion_level == "low" else \
                      COLORS["warning"] if selected_rsu.congestion_level == "medium" else \
                      COLORS["danger"]
                      
        status_frame = tk.Frame(header, bg="white")
        status_frame.pack(side="right")
        
        status_label = tk.Label(
            status_frame,
            text=f"Status: {selected_rsu.congestion_level.upper()}",
            font=("Segoe UI", 11, "bold"),
            fg=status_color,
            bg="white"
        )
        status_label.pack(anchor="e")
        
        vehicles_label = tk.Label(
            status_frame,
            text=f"Vehicles: {len(selected_rsu.vehicles_in_range)}",
            font=("Segoe UI", 10),
            fg="#777",
            bg="white"
        )
        vehicles_label.pack(anchor="e")
        
        # Divider
        ttk.Separator(details_frame, orient="horizontal").pack(fill="x", pady=10)
        
        # Details in two columns
        details = tk.Frame(details_frame, bg="white")
        details.pack(fill="x", pady=10)
        details.columnconfigure(0, weight=1)
        details.columnconfigure(1, weight=1)
        
        # Left column - Stats
        stats_frame = tk.LabelFrame(details, text="Current Statistics", bg="white", fg=COLORS["text"])
        stats_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # Calculate avg speed
        avg_speed = selected_rsu.get_average_speed()
            
        # Get recommendation
        rec_phase, rec_duration = selected_rsu.get_recommended_phase()
        
        # Stats
        tk.Label(stats_frame, text=f"Vehicle Count: {len(selected_rsu.vehicles_in_range)}", anchor="w", 
                bg="white", fg=COLORS["text"], pady=3).pack(fill="x")
        tk.Label(stats_frame, text=f"Average Speed: {avg_speed:.1f} m/s", anchor="w", 
                bg="white", fg=COLORS["text"], pady=3).pack(fill="x")
        tk.Label(stats_frame, text=f"Congestion Level: {selected_rsu.congestion_level.upper()}", anchor="w", 
                bg="white", fg=COLORS["text"], pady=3).pack(fill="x")
        tk.Label(stats_frame, text=f"Recommendation: {rec_phase.upper()} for {rec_duration}s", anchor="w", 
                bg="white", fg=COLORS["text"], pady=3).pack(fill="x")
        
        # Right column - Actions
        actions_frame = tk.LabelFrame(details, text="Actions", bg="white", fg=COLORS["text"])
        actions_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        # Manual lighting control
        tk.Label(actions_frame, text="Control Traffic Lights", bg="white", fg=COLORS["text"], pady=3).pack(anchor="w")
        
        light_btns = tk.Frame(actions_frame, bg="white")
        light_btns.pack(fill="x", pady=5)
        
        from gui.components.stylish_button import StylishButton
        
        StylishButton(light_btns, text="RED", color="danger", width=7,
                     command=lambda: self.set_light_phase(selected_rsu.id, "red")).pack(side="left", padx=5)
        StylishButton(light_btns, text="YELLOW", color="warning", width=7,
                     command=lambda: self.set_light_phase(selected_rsu.id, "yellow")).pack(side="left", padx=5)
        StylishButton(light_btns, text="GREEN", color="secondary", width=7,
                     command=lambda: self.set_light_phase(selected_rsu.id, "green")).pack(side="left", padx=5)
        
        # Apply recommendation button
        StylishButton(actions_frame, text="Apply Recommendation", color="primary",
                     command=lambda: self.apply_recommendation(selected_rsu.id)).pack(fill="x", pady=10)
        
        # Divider
        ttk.Separator(details_frame, orient="horizontal").pack(fill="x", pady=10)
        
        # Traffic chart
        chart_frame = tk.Frame(details_frame, bg="white")
        chart_frame.pack(fill="both", expand=True, pady=10)
        
        # Create a matplotlib figure for the RSU
        fig = Figure(figsize=(5, 3), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        # Plot vehicle count history
        timestamps = selected_rsu.history["timestamps"][-15:]  # Last 15 points
        vehicle_counts = selected_rsu.history["vehicle_counts"][-15:]
        
        ax.plot(range(len(timestamps)), vehicle_counts, 'o-', color=COLORS["primary"])
        ax.set_title('Vehicle Count History')
        ax.set_ylabel('Vehicles')
        
        # Set x-axis ticks to timestamps
        if timestamps:
            ax.set_xticks(range(len(timestamps)))
            
            # If there are many points, show only some labels
            if len(timestamps) > 7:
                show_idx = list(range(0, len(timestamps), 3))
                if len(timestamps) - 1 not in show_idx:
                    show_idx.append(len(timestamps) - 1)
                
                labels = [""] * len(timestamps)
                for idx in show_idx:
                    if idx < len(timestamps):
                        labels[idx] = timestamps[idx]
                
                ax.set_xticklabels(labels, rotation=45)
            else:
                ax.set_xticklabels(timestamps, rotation=45)
        
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Create canvas
        self.rsu_chart = FigureCanvasTkAgg(fig, master=chart_frame)
        self.rsu_chart.draw()
        self.rsu_chart.get_tk_widget().pack(fill="both", expand=True)
    
    def set_light_phase(self, rsu_id, phase):
        """Set the traffic light phase for a specific RSU"""
        result = set_traffic_light_phase(rsu_id, phase)
        
        if result:
            messagebox.showinfo("Traffic Light Control", 
                               f"Set traffic light at {rsu_id} to {phase.upper()}")
        else:
            messagebox.showerror("Error", "Failed to control traffic light")
    
    def apply_recommendation(self, rsu_id):
        """Apply the recommended phase for this RSU"""
        # Find the RSU
        selected_rsu = None
        for rsu in self.app.rsus:
            if rsu.id == rsu_id:
                selected_rsu = rsu
                break
                
        if not selected_rsu:
            return
            
        # Get recommendation
        rec_phase, rec_duration = selected_rsu.get_recommended_phase()
        
        # Apply it
        self.set_light_phase(rsu_id, rec_phase)
        
        messagebox.showinfo("Applied Recommendation", 
                          f"Applied recommended phase: {rec_phase.upper()} for {rec_duration}s")
        
    def update(self):
        """Update the RSU tab with current data"""
        self.update_rsu_tree()
        
        # If an RSU is selected, update its details
        if self.current_rsu_id:
            self.update_rsu_details(self.current_rsu_id)
            
    def update_rsu_tree(self):
        """Update the RSU information in the treeview"""
        # Clear the current contents
        for item in self.rsu_tree.get_children():
            self.rsu_tree.delete(item)
            
        # Add updated data for each RSU
        for rsu in self.app.rsus:
            # Calculate average speed
            avg_speed = rsu.get_average_speed()
            
            # Add to tree
            self.rsu_tree.insert("", "end", values=(
                rsu.id,
                f"({rsu.position[0]:.1f}, {rsu.position[1]:.1f})",
                len(rsu.vehicles_in_range),
                rsu.congestion_level.upper(),
                f"{avg_speed:.1f} m/s"
            ))