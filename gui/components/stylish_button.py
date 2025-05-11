"""
Stylish button implementation for the Smart Traffic Control System
"""

import tkinter as tk
from config.styles import COLORS, FONTS, BUTTON_STYLES

class StylishButton(tk.Button):
    """Custom styled button with hover effects"""
    def __init__(self, master=None, color="primary", **kwargs):
        self.color_theme = COLORS[color]
        
        # Update kwargs with default styling
        kwargs.update({
            "background": self.color_theme,
            "foreground": "white",
            "relief": BUTTON_STYLES["relief"],
            "font": FONTS["button"],
            "activebackground": self._adjust_brightness(self.color_theme, -20),
            "activeforeground": "white",
            "borderwidth": BUTTON_STYLES["border_width"],
            "padx": BUTTON_STYLES["padding_x"],
            "pady": BUTTON_STYLES["padding_y"],
            "cursor": BUTTON_STYLES["cursor"]
        })
        
        super().__init__(master, **kwargs)
        
        # Bind hover events
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