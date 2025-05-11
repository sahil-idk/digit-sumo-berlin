# gui/tabs/prediction_tab.py
"""
Prediction tab implementation for the Smart Traffic Control System
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from config.styles import COLORS
from config.settings import PREDICTION_CONFIG
from gui.components.dashboard_panel import DashboardPanel
from gui.components.stylish_button import StylishButton
from prediction.prediction_service import PredictionService

class PredictionTab:
    """Prediction tab for ML model integration"""
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.prediction_service = PredictionService()
        
        # Create a main frame with two columns
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # Setup components
        self.setup_prediction_controls()
        self.setup_prediction_results()
        
    def setup_prediction_controls(self):
        """Setup the prediction controls panel"""
        prediction_panel = DashboardPanel(self.parent, title="Traffic Prediction")
        prediction_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)
        
        # Control frame
        control_frame = tk.Frame(prediction_panel.content, bg="white")
        control_frame.pack(fill="both", expand=True, pady=10)
        
        # Prediction section
        tk.Label(control_frame, text="ML Prediction Settings", bg="white", 
                fg=COLORS["text"], font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 10))
        
        # Input sequence length
        tk.Label(control_frame, text="Input Sequence Length:", bg="white", 
                fg=COLORS["text"], font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 5))
        
        self.seq_length_var = tk.IntVar(value=PREDICTION_CONFIG["default_seq_length"])
        seq_frame = tk.Frame(control_frame, bg="white")
        seq_frame.pack(anchor="w", pady=(0, 15))
        
        seq_entry = ttk.Entry(seq_frame, width=5, textvariable=self.seq_length_var)
        seq_entry.pack(side="left")
        tk.Label(seq_frame, text="timesteps", bg="white", fg="#777").pack(side="left", padx=5)
        
        # Prediction horizon
        tk.Label(control_frame, text="Prediction Horizon:", bg="white", 
                fg=COLORS["text"], font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 5))
        
        self.horizon_var = tk.IntVar(value=PREDICTION_CONFIG["default_horizon"])
        horizon_frame = tk.Frame(control_frame, bg="white")
        horizon_frame.pack(anchor="w", pady=(0, 15))
        
        horizon_entry = ttk.Entry(horizon_frame, width=5, textvariable=self.horizon_var)
        horizon_entry.pack(side="left")
        tk.Label(horizon_frame, text="future timesteps", bg="white", fg="#777").pack(side="left", padx=5)
        
        # Data source selection
        tk.Label(control_frame, text="Data Source:", bg="white", 
                fg=COLORS["text"], font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 5))
        
        self.data_source_var = tk.StringVar(value=PREDICTION_CONFIG["default_data_source"])
        source_frame = tk.Frame(control_frame, bg="white")
        source_frame.pack(anchor="w", pady=(0, 15))
        
        ttk.Radiobutton(source_frame, text="Current Simulation", 
                       variable=self.data_source_var, value="Current Simulation").pack(anchor="w")
        ttk.Radiobutton(source_frame, text="Historical Data (CSV)", 
                       variable=self.data_source_var, value="Historical Data").pack(anchor="w")
        
        # CSV file selection (enabled only when Historical Data is selected)
        csv_frame = tk.Frame(control_frame, bg="white")
        csv_frame.pack(anchor="w", pady=(0, 15))
        
        self.csv_path_var = tk.StringVar(value=PREDICTION_CONFIG["default_csv_path"])
        csv_entry = ttk.Entry(csv_frame, width=20, textvariable=self.csv_path_var)
        csv_entry.pack(side="left")
        
        StylishButton(csv_frame, text="Browse", color="primary",
                     command=self.browse_csv_file).pack(side="left", padx=5)
        
        # Create a separator
        ttk.Separator(control_frame, orient="horizontal").pack(fill="x", pady=15)
        
        # Run prediction button
        StylishButton(control_frame, text="Run Prediction", color="primary",
                     command=self.run_ml_prediction).pack(anchor="w", pady=10)
        
        # Status label
        self.prediction_status_var = tk.StringVar(value="Ready to generate predictions")
        tk.Label(control_frame, textvariable=self.prediction_status_var, 
                bg="white", fg=COLORS["text"]).pack(anchor="w", pady=10)
    
    def setup_prediction_results(self):
        """Setup the prediction results panel"""
        results_panel = DashboardPanel(self.parent, title="Prediction Results")
        results_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)
        
        # Results container
        self.prediction_results = tk.Frame(results_panel.content, bg="white")
        self.prediction_results.pack(fill="both", expand=True)
        
        # Create the initial empty chart
        self.setup_prediction_chart(self.prediction_results)

    def setup_prediction_chart(self, parent):
        """Setup the prediction chart display"""
        # Create matplotlib figure for prediction results
        self.pred_fig = Figure(figsize=(5, 4), dpi=100, facecolor='white')
        self.pred_ax = self.pred_fig.add_subplot(111)
        self.pred_ax.set_title('Traffic Flow Prediction')
        self.pred_ax.set_xlabel('Time Step')
        self.pred_ax.set_ylabel('Vehicle Count')
        self.pred_ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add a text annotation for initial state
        self.pred_ax.annotate('No prediction data available yet', 
                            xy=(0.5, 0.5), xycoords='axes fraction',
                            ha='center', va='center', fontsize=12)
        
        # Create canvas
        self.pred_canvas = FigureCanvasTkAgg(self.pred_fig, master=parent)
        self.pred_canvas.draw()
        self.pred_canvas.get_tk_widget().pack(fill="both", expand=True)

    def browse_csv_file(self):
        """Browse for CSV file"""
        filepath = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            self.csv_path_var.set(filepath)

    def run_ml_prediction(self):
        """Run the ML prediction model"""
        try:
            # Update status
            self.prediction_status_var.set("Processing data...")
            self.parent.update()
            
            # Get parameters
            seq_length = self.seq_length_var.get()
            horizon = self.horizon_var.get()
            data_source = self.data_source_var.get()
            
            # Prepare input data
            if data_source == "Current Simulation":
                # Get vehicle counts from analytics data
                vehicle_counts = self.app.analytics_data["vehicle_counts"]
                
                # Run the prediction
                result = self.prediction_service.run_prediction(
                    "simulation", vehicle_counts, seq_length, horizon)
            else:  # Historical data
                csv_path = self.csv_path_var.get()
                
                # Run prediction on CSV data
                result = self.prediction_service.run_prediction(
                    "csv", csv_path, seq_length, horizon)
            
            # Check result
            if result["status"] == "success":
                # Display the predictions
                self.display_predictions(result)
                self.prediction_status_var.set("Prediction completed successfully")
                
                # Also display the input sequence specifically
                self.display_input_sequence()
            else:
                self.prediction_status_var.set(f"Error: {result['error']}")
        
        except Exception as e:
            self.prediction_status_var.set(f"Error: {str(e)}")
            print(f"Prediction process error: {e}")
            import traceback
            traceback.print_exc()

    def display_input_sequence(self):
        """Display the input sequence used for the prediction"""
        # Get prediction data
        pred_data = self.prediction_service.get_latest_predictions()
        if not pred_data:
            return
            
        # Find the prediction results frame
        input_frame = None
        
        # Look for existing input_sequence_frame
        for widget in self.prediction_results.winfo_children():
            if hasattr(widget, 'input_sequence_tag') and widget.input_sequence_tag:
                input_frame = widget
                break
        
        # If we didn't find it, create a new one
        if not input_frame:
            input_frame = tk.Frame(self.prediction_results, bg="white")
            input_frame.input_sequence_tag = True  # Add a tag to identify this frame
            input_frame.pack(fill="x", padx=10, pady=10, after=self.pred_canvas.get_tk_widget())
        
        # Clear any existing content
        for widget in input_frame.winfo_children():
            widget.destroy()
        
        # Create a title for the input sequence section
        tk.Label(input_frame, text="Input Sequence Used", 
                font=("Segoe UI", 11, "bold"), bg="white", fg=COLORS["text"]).pack(anchor="w")
        
        # Create a frame for the sequence values
        seq_frame = tk.Frame(input_frame, bg="white")
        seq_frame.pack(fill="x", pady=5)
        
        # Display the historical data as a horizontal sequence
        historical_data = pred_data["input_sequence"]
        
        if historical_data:
            # Create a grid display for the sequence
            seq_grid = tk.Frame(seq_frame, bg="white")
            seq_grid.pack(fill="x", pady=5)
            
            # Add index headers
            for i in range(len(historical_data)):
                idx_label = tk.Label(seq_grid, text=f"t-{len(historical_data)-i-1}", 
                                   font=("Segoe UI", 9, "bold"), bg=COLORS["light"], fg=COLORS["dark"],
                                   width=8, padx=5, pady=3, borderwidth=1, relief="solid")
                idx_label.grid(row=0, column=i, sticky="nsew")
            
            # Add value row
            for i, value in enumerate(historical_data):
                value_label = tk.Label(seq_grid, text=f"{value:.1f}", 
                                     font=("Segoe UI", 10), bg="white", fg=COLORS["primary"],
                                     width=8, padx=5, pady=5, borderwidth=1, relief="solid")
                value_label.grid(row=1, column=i, sticky="nsew")
            
            # Add explanation
            explanation = tk.Label(seq_frame, text="This sequence of historical vehicle counts was used as input for the prediction model.",
                                 font=("Segoe UI", 9), bg="white", fg="#777", justify="left", wraplength=400)
            explanation.pack(anchor="w", pady=5)
        else:
            # If no data, show a message
            tk.Label(seq_frame, text="No input sequence data available", 
                    font=("Segoe UI", 10), bg="white", fg="#777").pack(anchor="w")

    def display_predictions(self, result):
        """Display the prediction results on the chart"""
        # Clear previous plot
        self.pred_ax.clear()
        
        # Get data
        input_sequence = self.prediction_service.input_sequence
        predictions = result["predictions"]
        
        if not input_sequence and not predictions:
            self.pred_ax.annotate('No prediction data available yet', 
                                xy=(0.5, 0.5), xycoords='axes fraction',
                                ha='center', va='center', fontsize=12)
            self.pred_canvas.draw()
            return
        
        # Create time steps
        time_steps = list(range(len(input_sequence) + len(predictions)))
        
        # Create data points
        historical_data = list(input_sequence) + [np.nan] * len(predictions)
        future_data = [np.nan] * len(input_sequence) + list(predictions)
        
        # Plot historical data with clear markers
        self.pred_ax.plot(time_steps[:len(input_sequence)], historical_data[:len(input_sequence)], 
                         'o-', color=COLORS["primary"], linewidth=2, markersize=6,
                         label='Historical Data')
        
        # Plot predicted data with distinctive style
        self.pred_ax.plot(time_steps[len(input_sequence)-1:], [historical_data[len(input_sequence)-1]] + list(predictions), 
                         's--', color=COLORS["danger"], linewidth=2, markersize=6,
                         label='Predicted Data')
        
        # Mark the current time point with a vertical line
        self.pred_ax.axvline(x=len(input_sequence)-1, color='black', linestyle='--', alpha=0.7)
        self.pred_ax.annotate('Now', xy=(len(input_sequence)-1, 0), xytext=(len(input_sequence)-1, -5),
                            ha='center', va='top')
        
        # Add value labels to the plot points
        for i, value in enumerate(input_sequence):
            self.pred_ax.annotate(f"{value:.1f}", 
                                xy=(i, value), 
                                xytext=(0, 10),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8)
        
        for i, value in enumerate(predictions):
            self.pred_ax.annotate(f"{value:.1f}", 
                                xy=(len(input_sequence) + i, value), 
                                xytext=(0, -10),
                                textcoords="offset points",
                                ha='center', va='top',
                                fontsize=8, 
                                color=COLORS["danger"])
        
        # Set labels and title
        self.pred_ax.set_title('Traffic Flow Prediction')
        self.pred_ax.set_xlabel('Time Step')
        self.pred_ax.set_ylabel('Vehicle Count')
        self.pred_ax.grid(True, linestyle='--', alpha=0.7)
        self.pred_ax.legend(loc='best')
        
        # Add clear separation between historical and predicted
        self.pred_ax.set_xticks(time_steps)
        labels = [f"t-{len(input_sequence)-i-1}" for i in range(len(input_sequence))] + \
                 [f"t+{i+1}" for i in range(len(predictions))]
        self.pred_ax.set_xticklabels(labels, rotation=45)
        
        # Update the canvas
        self.pred_canvas.draw()
        
        # Also create a table with the numerical values
        self.display_prediction_table(result)

    def display_prediction_table(self, result):
        """Display a table of prediction values"""
        # Create a frame for the table below the chart
        for widget in self.prediction_results.winfo_children():
            if widget != self.pred_canvas.get_tk_widget() and not (hasattr(widget, 'input_sequence_tag') and widget.input_sequence_tag):
                widget.destroy()
        
        table_frame = tk.Frame(self.prediction_results, bg="white")
        table_frame.pack(fill="x", padx=10, pady=10)
        
        # Create a title for the table
        tk.Label(table_frame, text="Prediction Results", 
                font=("Segoe UI", 11, "bold"), bg="white", fg=COLORS["text"]).pack(anchor="w")
        
        # Create the table headers
        headers = ["Time Step", "Predicted Value"]
        
        # Create a subframe for the table
        table = tk.Frame(table_frame, bg="white")
        table.pack(fill="x", pady=5)
        
        # Create header row
        for i, header in enumerate(headers):
            tk.Label(table, text=header, font=("Segoe UI", 10, "bold"), 
                    bg=COLORS["light"], fg=COLORS["dark"], 
                    padx=10, pady=5, borderwidth=1, relief="solid").grid(row=0, column=i, sticky="nsew")
        
        # Add the data rows
        for i, (timestamp, prediction) in enumerate(zip(
                result["timestamps"], 
                result["predictions"]
            )):
            
            # Add row
            row_color = "white" if i % 2 == 0 else "#f8f8f8"
            
            tk.Label(table, text=timestamp, font=("Segoe UI", 9), 
                    bg=row_color, padx=10, pady=5, borderwidth=1, relief="solid").grid(row=i+1, column=0, sticky="nsew")
            
            tk.Label(table, text=f"{prediction:.2f}", font=("Segoe UI", 9), 
                    bg=row_color, fg=COLORS["danger"], padx=10, pady=5, borderwidth=1, relief="solid").grid(row=i+1, column=1, sticky="nsew")
        
        # Add export button
        export_frame = tk.Frame(table_frame, bg="white")
        export_frame.pack(anchor="e", pady=10)
        
        StylishButton(export_frame, text="Export Predictions", color="primary",
                     command=lambda: self.export_predictions(result)).pack(side="right")

    def export_predictions(self, result):
        """Export prediction data to CSV"""
        try:
            # Ask for save location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save Prediction Results"
            )
            
            if not file_path:
                return
            
            # Create a DataFrame
            data = {
                "Timestamp": result["timestamps"],
                "Predicted_Value": result["predictions"]
            }
            
            df = pd.DataFrame(data)
            
            # Save to CSV
            df.to_csv(file_path, index=False)
            
            messagebox.showinfo("Export Successful", 
                              f"Prediction results saved to {file_path}")
        
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")
            print(f"Export error: {e}")
            
    def update(self):
        """Update tab content if needed"""
        pass  # This tab mostly updates on demand