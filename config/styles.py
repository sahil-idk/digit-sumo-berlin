"""
UI styles and color configuration for the Smart Traffic Control System
"""

# Color palette
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

# Font configurations
FONTS = {
    "header": ("Segoe UI", 16, "bold"),
    "subheader": ("Segoe UI", 12, "bold"),
    "default": ("Segoe UI", 10),
    "small": ("Segoe UI", 9),
    "button": ("Segoe UI", 10),
    "stats": ("Segoe UI", 22, "bold"),
}

# Button styles
BUTTON_STYLES = {
    "padding_x": 15,
    "padding_y": 8,
    "border_width": 0,
    "cursor": "hand2",
    "relief": "flat",
}

# Widget styles for ttk
TTK_STYLES = {
    "TFrame": {
        "configure": {
            "background": COLORS["background"]
        }
    },
    "TLabel": {
        "configure": {
            "background": COLORS["background"],
            "foreground": COLORS["text"]
        }
    },
    "TButton": {
        "configure": {
            "padding": 6,
            "relief": "flat",
            "background": COLORS["primary"]
        }
    },
    "Treeview": {
        "configure": {
            "background": "white",
            "foreground": COLORS["text"],
            "rowheight": 30,
            "fieldbackground": "white"
        },
        "map": {}
    },
    "Treeview.Heading": {
        "configure": {
            "font": ('Segoe UI', 10, 'bold'),
            "background": COLORS["light"],
            "foreground": COLORS["dark"]
        }
    },
    "TNotebook": {
        "configure": {
            "background": COLORS["background"],
            "tabmargins": [0, 0, 0, 0]
        }
    },
    "TNotebook.Tab": {
        "configure": {
            "background": COLORS["light"],
            "padding": [15, 5],
            "font": ('Segoe UI', 10)
        },
        "map": {
            "background": [("selected", COLORS["primary"])],
            "foreground": [("selected", "white")]
        }
    }
}

# Stats panel configurations
STAT_PANELS = [
    {"title": "Active Vehicles", "icon": "🚗", "value": "0", "unit": "vehicles"},
    {"title": "Average Speed", "icon": "⚡", "value": "0", "unit": "m/s"},
    {"title": "Congestion Level", "icon": "🚦", "value": "Low", "unit": ""},
    {"title": "Simulation Time", "icon": "⏱️", "value": "0:00", "unit": ""}
]

# Scenario descriptions for the scenario tab
SCENARIO_DESCRIPTIONS = {
    "Normal Traffic": "Regular traffic flow with standard vehicle behavior. "
                    "This is the baseline scenario with normal speeds and densities.",
    
    "Rush Hour": "High density of vehicles simulating peak hours. "
                "Expect congestion at intersections and reduced average speeds. "
                "Traffic lights may need longer green phases to handle the volume.",
    
    "Rainy Day": "Reduced visibility and slower vehicle speeds due to wet conditions. "
                "Vehicles maintain longer following distances and brake earlier. "
                "Maximum speed is reduced to 15 m/s (~54 km/h).",
    
    "Foggy Morning": "Severely reduced visibility with much slower vehicle speeds. "
                    "Vehicles are extra cautious, with max speed limited to 10 m/s (~36 km/h). "
                    "Traffic density is moderate but flow is significantly affected.",
    
    "Emergency": "Emergency vehicles are given priority. "
                "Regular traffic density but includes emergency vehicles that "
                "other vehicles should yield to."
}