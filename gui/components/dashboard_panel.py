"""
Dashboard panel component for the Smart Traffic Control System
"""

import tkinter as tk
from config.styles import COLORS, FONTS

class DashboardPanel(tk.Frame):
    """Base panel for dashboard elements with standard styling"""
    def __init__(self, master=None, title="Panel", **kwargs):
        # Update kwargs with default styling
        kwargs.update({
            "background": "white",
            "padx": 10,
            "pady": 10,
            "relief": "flat",
            "highlightthickness": 1,
            "highlightbackground": "#ddd"
        })
        super().__init__(master, **kwargs)
        
        # Header
        self.header = tk.Frame(self, bg="white")
        self.header.pack(fill="x", pady=(0, 10))
        
        self.title_label = tk.Label(
            self.header, 
            text=title, 
            font=FONTS["subheader"],
            bg="white",
            fg=COLORS["text"]
        )
        self.title_label.pack(side="left")
        
        # Right side of header (for optional buttons/controls)
        self.header_right = tk.Frame(self.header, bg="white")
        self.header_right.pack(side="right")
        
        # Content area
        self.content = tk.Frame(self, bg="white")
        self.content.pack(fill="both", expand=True)
        
    def add_header_button(self, text, command, color="primary"):
        """Add a button to the panel header"""
        from gui.components.stylish_button import StylishButton
        
        button = StylishButton(
            self.header_right,
            text=text,
            color=color,
            command=command
        )
        button.pack(side="right", padx=2)
        return button
        
    def set_title(self, title):
        """Update the panel title"""
        self.title_label.config(text=title)
        
    def clear_content(self):
        """Clear all widgets from the content area"""
        for widget in self.content.winfo_children():
            widget.destroy()
    
    def create_stat_panel(self, title, icon, value, unit):
        """Create a statistics panel within this dashboard panel"""
        panel = tk.Frame(self.content, bg="white", padx=15, pady=15)
        panel.configure(highlightbackground="#ddd", highlightthickness=1)
        
        # Title
        tk.Label(
            panel, 
            text=title, 
            font=FONTS["small"],
            fg="#777",
            bg="white"
        ).pack(anchor="w")
        
        # Value with icon
        value_frame = tk.Frame(panel, bg="white")
        value_frame.pack(fill="x", pady=5)
        
        value_label = tk.Label(
            value_frame, 
            text=value, 
            font=FONTS["stats"],
            fg=COLORS["text"],
            bg="white"
        )
        value_label.pack(side="left")
        
        # Unit label
        unit_label = tk.Label(
            value_frame, 
            text=f" {unit}", 
            font=FONTS["small"],
            fg="#777",
            bg="white"
        )
        unit_label.pack(side="left", padx=2, pady=(8, 0))
        
        # Icon on the right
        icon_label = tk.Label(
            value_frame, 
            text=icon, 
            font=FONTS["stats"],
            fg=COLORS["primary"],
            bg="white"
        )
        icon_label.pack(side="right")
        
        return panel, value_label, unit_label, icon_label