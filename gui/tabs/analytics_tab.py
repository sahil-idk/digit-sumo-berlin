# gui/tabs/analytics_tab.py
"""
Analytics tab implementation for the Smart Traffic Control System
"""

import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from config.styles import COLORS

class AnalyticsTab:
    """Analytics tab with graphs and statistics"""
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        
        # Create a matplotlib figure for overall vehicle count
        self.fig = Figure(figsize=(12, 6), dpi=100, facecolor='white')
        
        # Vehicle count subplot
        self.ax1 = self.fig.add_subplot(211)
        self.vehicle_count_line, = self.ax1.plot([], [], 'o-', color=COLORS["primary"], lw=2)
        self.ax1.set_title('Total Vehicles in Simulation')
        self.ax1.set_xlabel('Simulation Steps')
        self.ax1.set_ylabel('Vehicle Count')
        self.ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Average speed subplot
        self.ax2 = self.fig.add_subplot(212)
        self.avg_speed_line, = self.ax2.plot([], [], 'o-', color=COLORS["secondary"], lw=2)
        self.ax2.set_title('Average Vehicle Speed')
        self.ax2.set_xlabel('Simulation Steps')
        self.ax2.set_ylabel('Speed (m/s)')
        self.ax2.grid(True, linestyle='--', alpha=0.7)
        
        self.fig.tight_layout()
        
        # Create the canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
    def update(self):
        """Update the analytics graphs with current data"""
        # Get the data
        steps = self.app.analytics_data["steps"]
        vehicle_counts = self.app.analytics_data["vehicle_counts"]
        avg_speeds = self.app.analytics_data["avg_speeds"]
        
        if not steps or not vehicle_counts or not avg_speeds:
            return
            
        # Update the plots
        self.vehicle_count_line.set_data(range(len(steps)), vehicle_counts)
        self.avg_speed_line.set_data(range(len(steps)), avg_speeds)
        
        # Adjust axes limits
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # Set x-axis ticks
        self._set_axis_ticks(self.ax1, steps)
        self._set_axis_ticks(self.ax2, steps)
        
        # Force redraw the canvas
        self.canvas.draw()
        
    def _set_axis_ticks(self, ax, steps):
        """Set x-axis ticks for an axis"""
        if not steps:
            return
            
        ax.set_xticks(range(len(steps)))
        
        # Format times for better display
        formatted_times = []
        for step in steps:
            mins = int(step) // 60
            secs = int(step) % 60
            formatted_times.append(f"{mins}:{secs:02d}")
            
        # If there are many points, show only some labels
        if len(steps) > 10:
            show_idx = list(range(0, len(steps), 3))
            if len(steps) - 1 not in show_idx:
                show_idx.append(len(steps) - 1)
            
            labels = [""] * len(steps)
            for idx in show_idx:
                if idx < len(steps):
                    labels[idx] = formatted_times[idx]
            
            ax.set_xticklabels(labels)
        else:
            ax.set_xticklabels(formatted_times)