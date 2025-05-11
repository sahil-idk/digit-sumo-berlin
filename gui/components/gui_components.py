"""
Custom GUI components for the traffic simulation application
"""

import tkinter as tk
from tkinter import ttk

# Define color scheme for consistent UI
COLORS = {
    "primary": "#3498db",       # Blue
    "secondary": "#2ecc71",     # Green
    "warning": "#f39c12",       # Orange
    "danger": "#e74c3c",        # Red
    "dark": "#2c3e50",          # Dark blue/gray
    "light": "#ecf0f1",         # Light gray
    "text": "#34495e",          # Dark gray for text
    "background": "#f9f9f9"     # Off-white background
}

class StylishButton(tk.Button):
    """Custom styled button with hover effects"""
    def __init__(self, master=None, color="primary", **kwargs):
        self.color_theme = COLORS[color]
        kwargs.update({
            "background": self.color_theme,
            "foreground": "white",
            "relief": "flat",
            "font": ("Segoe UI", 10),
            "activebackground": self._adjust_brightness(self.color_theme, -20),
            "activeforeground": "white",
            "borderwidth": 0,
            "padx": 15,
            "pady": 8,
            "cursor": "hand2"
        })
        super().__init__(master, **kwargs)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        
    def _adjust_brightness(self, hex_color, factor):
        """Adjust color brightness"""
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        
        r = max(0, min(255, r + factor))
        g = max(0, min(255, g + factor))
        b = max(0, min(255, b + factor))
        
        return f"#{r:02x}{g:02x}{b:02x}"
        
    def _on_enter(self, event):
        """Mouse enter effect"""
        self.config(background=self._adjust_brightness(self.color_theme, -10))
        
    def _on_leave(self, event):
        """Mouse leave effect"""
        self.config(background=self.color_theme)

class DashboardPanel(tk.Frame):
    """Panel for dashboard elements with a title and content area"""
    def __init__(self, master=None, title="Panel", **kwargs):
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
            font=("Segoe UI", 12, "bold"),
            bg="white",
            fg=COLORS["text"]
        )
        self.title_label.pack(side="left")
        
        # Content area
        self.content = tk.Frame(self, bg="white")
        self.content.pack(fill="both", expand=True)

def configure_styles():
    """Configure ttk styles for a modern look"""
    style = ttk.Style()
    
    # Try to use a more modern theme if available
    if "clam" in style.theme_names():
        style.theme_use("clam")
    
    # General styles
    style.configure("TFrame", background=COLORS["background"])
    style.configure("TLabel", background=COLORS["background"], foreground=COLORS["text"])
    style.configure("TButton", padding=6, relief="flat", background=COLORS["primary"])
    
    # TreeView styling
    style.configure(
        "Treeview", 
        background="white", 
        foreground=COLORS["text"], 
        rowheight=30,
        fieldbackground="white"
    )
    style.configure("Treeview.Heading", 
                   font=('Segoe UI', 10, 'bold'),
                   background=COLORS["light"],
                   foreground=COLORS["dark"])
    
    # Notebook styling
    style.configure("TNotebook", background=COLORS["background"], tabmargins=[0, 0, 0, 0])
    style.configure("TNotebook.Tab", background=COLORS["light"], padding=[15, 5], font=('Segoe UI', 10))
    style.map("TNotebook.Tab", 
             background=[("selected", COLORS["primary"])],
             foreground=[("selected", "white")])
    
    return style

def create_stat_panel(parent, title, icon, value, unit):
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
    
    tk.Label(
        value_frame, 
        text=f" {unit}", 
        font=("Segoe UI", 10),
        fg="#777",
        bg="white"
    ).pack(side="left", padx=2, pady=(8, 0))
    
    # Icon on the right
    tk.Label(
        value_frame, 
        text=icon, 
        font=("Segoe UI", 22),
        fg=COLORS["primary"],
        bg="white"
    ).pack(side="right")
    
    return panel