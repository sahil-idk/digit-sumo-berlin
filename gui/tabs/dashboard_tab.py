"""
Dashboard tab implementation for the Smart Traffic Control System
"""

import tkinter as tk
from tkinter import ttk
from datetime import datetime
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from config.styles import COLORS, STAT_PANELS
from gui.components.dashboard_panel import DashboardPanel

class DashboardTab:
    """Dashboard tab with statistics and overview information"""
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.stat_labels = {}
        
        # Create a grid layout
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=0)  # Stats row
        parent.rowconfigure(1, weight=1)  # Charts/map row
        
        # Setup components
        self.setup_statistics()
        self.setup_charts()
        self.setup_rsu_status()
        
    def setup_statistics(self):
        """Setup the statistics panels at the top"""
        stats_frame = tk.Frame(self.parent, bg=COLORS["background"])
        stats_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        # Create stat panels
        for i, panel_info in enumerate(STAT_PANELS):
            stat_panel, value_label, _, _ = self.create_stat_panel(
                stats_frame, 
                panel_info["title"], 
                panel_info["icon"], 
                panel_info["value"], 
                panel_info["unit"]
            )
            stat_panel.pack(side="left", fill="both", expand=True, padx=5)
            
            # Save reference to update these values
            self.stat_labels[panel_info["title"]] = value_label
            
    def create_stat_panel(self, parent, title, icon, value, unit):
        """Create a stylish statistics panel"""
        panel = tk.Frame(parent, bg="white", padx=15, pady=15)
        panel.configure(highlightbackground="#ddd", highlightthickness=1)
        
        # Title
        tk.Label(
            panel, 
            text=title, 
            font=("Segoe UI", 10),
            fg="#777",
            bg="white"
        ).pack(anchor="w")
        
        # Value with icon
        value_frame = tk.Frame(panel, bg="white")
        value_frame.pack(fill="x", pady=5)
        
        value_label = tk.Label(
            value_frame, 
            text=value, 
            font=("Segoe UI", 22, "bold"),
            fg=COLORS["text"],
            bg="white"
        )
        value_label.pack(side="left")
        
        unit_label = tk.Label(
            value_frame, 
            text=f" {unit}", 
            font=("Segoe UI", 10),
            fg="#777",
            bg="white"
        )
        unit_label.pack(side="left", padx=2, pady=(8, 0))
        
        # Icon on the right
        icon_label = tk.Label(
            value_frame, 
            text=icon, 
            font=("Segoe UI", 22),
            fg=COLORS["primary"],
            bg="white"
        )
        icon_label.pack(side="right")
        
        return panel, value_label, unit_label, icon_label
        
    def setup_charts(self):
        """Setup the traffic flow chart panel"""
        self.chart_panel = DashboardPanel(self.parent, title="Traffic Flow")
        self.chart_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        
        # Setup matplotlib figure for the chart
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor='white')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('white')
        self.ax.set_title('Vehicle Count Over Time')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Vehicles')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_panel.content)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def setup_rsu_status(self):
        """Setup the RSU status panel with treeview"""
        self.rsu_panel = DashboardPanel(self.parent, title="RSU Status")
        self.rsu_panel.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        
        # Create treeview for RSU data
        columns = ("RSU", "Vehicles", "Status", "Recommendation")
        self.dashboard_tree = ttk.Treeview(
            self.rsu_panel.content, 
            columns=columns, 
            show="headings", 
            style="Treeview"
        )
        
        # Configure columns
        for col in columns:
            self.dashboard_tree.heading(col, text=col)
            width = 100 if col == "Recommendation" else 80
            self.dashboard_tree.column(col, width=width)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(self.rsu_panel.content, orient="vertical", command=self.dashboard_tree.yview)
        self.dashboard_tree.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        self.dashboard_tree.pack(fill="both", expand=True)
        
    def update(self):
        """Update the dashboard components with current data"""
        self.update_statistics()
        self.update_chart()
        self.update_rsu_status()
        
    def update_statistics(self):
        """Update the statistics panels with current data"""
        # Get current data
        stats = self.get_simulation_stats()
        
        # Update stat labels
        self.stat_labels["Active Vehicles"].config(text=str(stats["vehicle_count"]))
        self.stat_labels["Average Speed"].config(text=f"{stats['avg_speed']:.1f}")
        self.stat_labels["Congestion Level"].config(text=stats["congestion_level"])
        
        # Format time
        minutes = int(stats["sim_time"]) // 60
        seconds = int(stats["sim_time"]) % 60
        time_str = f"{minutes}:{seconds:02d}"
        self.stat_labels["Simulation Time"].config(text=time_str)
        
    def update_chart(self):
        """Update the traffic flow chart"""
        # Clear the previous plot
        self.ax.clear()
        
        # Get the data from analytics
        steps = self.app.analytics_data["steps"]
        counts = self.app.analytics_data["vehicle_counts"]
        
        if not steps or not counts:
            self.ax.text(0.5, 0.5, "No data available", 
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=self.ax.transAxes)
            self.canvas.draw()
            return
            
        # Format times for better display
        formatted_times = []
        for step in steps:
            mins = int(step) // 60
            secs = int(step) % 60
            formatted_times.append(f"{mins}:{secs:02d}")
        
        # Create the new plot
        self.ax.plot(range(len(steps)), counts, 'o-', color=COLORS["primary"], lw=2)
        self.ax.set_title('Vehicle Count Over Time')
        self.ax.set_xlabel('Simulation Time')
        self.ax.set_ylabel('Vehicles')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set x-axis ticks to formatted times
        if formatted_times:
            self.ax.set_xticks(range(len(formatted_times)))
            
            # If there are many points, show only some labels
            if len(formatted_times) > 5:
                show_idx = list(range(0, len(formatted_times), 2))
                if len(formatted_times) - 1 not in show_idx:
                    show_idx.append(len(formatted_times) - 1)
                
                labels = [""] * len(formatted_times)
                for idx in show_idx:
                    if idx < len(formatted_times):
                        labels[idx] = formatted_times[idx]
                
                self.ax.set_xticklabels(labels, rotation=45)
            else:
                self.ax.set_xticklabels(formatted_times, rotation=45)
        
        # Redraw the canvas
        self.canvas.draw()
        
    def update_rsu_status(self):
        """Update the RSU status treeview"""
        # Clear current entries
        for item in self.dashboard_tree.get_children():
            self.dashboard_tree.delete(item)
            
        # Add updated data for each RSU
        for rsu in self.app.rsus:
            # Get recommendation
            rec_phase, rec_duration = rsu.get_recommended_phase()
            
            # Add to tree with background color based on congestion
            tag = rsu.congestion_level
            self.dashboard_tree.insert("", "end", tags=(tag,), values=(
                rsu.id,
                len(rsu.vehicles_in_range),
                rsu.congestion_level.upper(),
                f"{rec_phase.upper()} for {rec_duration}s"
            ))
        
        # Configure tag colors
        self.dashboard_tree.tag_configure("low", background="#e8f5e9")    # Light green
        self.dashboard_tree.tag_configure("medium", background="#fff9c4")  # Light yellow
        self.dashboard_tree.tag_configure("high", background="#ffebee")   # Light red
        
    def get_simulation_stats(self):
        """Calculate overall simulation statistics"""
        # Get basic stats
        vehicle_count = 0
        avg_speed = 0
        
        if self.app.analytics_data["vehicle_counts"]:
            vehicle_count = self.app.analytics_data["vehicle_counts"][-1]
            
        if self.app.analytics_data["avg_speeds"]:
            avg_speed = self.app.analytics_data["avg_speeds"][-1]
            
        # Determine overall congestion level
        congestion_counts = {"low": 0, "medium": 0, "high": 0}
        for rsu in self.app.rsus:
            congestion_counts[rsu.congestion_level] += 1
        
        if congestion_counts["high"] > len(self.app.rsus) * 0.3:
            overall_congestion = "High"
        elif congestion_counts["medium"] > len(self.app.rsus) * 0.3:
            overall_congestion = "Medium"
        else:
            overall_congestion = "Low"
            
        # Get simulation time
        sim_time = 0
        if self.app.analytics_data["steps"]:
            sim_time = self.app.analytics_data["steps"][-1]
            
        return {
            "vehicle_count": vehicle_count,
            "avg_speed": avg_speed,
            "congestion_level": overall_congestion,
            "sim_time": sim_time
        }