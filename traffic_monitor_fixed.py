import tkinter as tk
from tkinter import ttk
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add the predict directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
predict_dir = os.path.join(current_dir, "Pems_sahil", "predict")
sys.path.append(predict_dir)

# Ensure the reload directory exists and create a default reload file
reload_dir = os.path.join(predict_dir, 'reload')
os.makedirs(reload_dir, exist_ok=True)
reload_file = os.path.join(reload_dir, 'lstm.csv')
pd.DataFrame([["False"]], columns=['reload']).to_csv(reload_file, index=False)

from lstm_predict import predict

def get_last_complete_interval(current_time=None):
    """Get the most recent completed 5-minute interval."""
    if current_time is None:
        current_time = datetime.now()
    
    # Round down to the nearest 5-minute interval
    minutes = current_time.minute
    last_interval_minute = (minutes // 5) * 5
    
    last_interval = current_time.replace(
        minute=last_interval_minute,
        second=0,
        microsecond=0
    )
    
    return last_interval

class TrafficMonitorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Traffic Monitor and Prediction")
        self.root.geometry("800x600")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create control frame
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Refresh Data", 
                  command=self.refresh_data).pack(side=tk.LEFT, padx=5)
        
        self.auto_refresh_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Auto Refresh (30s)", 
                       variable=self.auto_refresh_var,
                       command=self.toggle_auto_refresh).pack(side=tk.LEFT, padx=5)
        
        # Create display areas
        display_frame = ttk.Frame(self.main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Current Data Frame (Left)
        current_frame = ttk.LabelFrame(display_frame, text="Current Traffic Data", padding="10")
        current_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.current_data_text = tk.Text(current_frame, height=20, width=40)
        self.current_data_text.pack(fill=tk.BOTH, expand=True)
        
        # Predictions Frame (Right)
        prediction_frame = ttk.LabelFrame(display_frame, text="Traffic Predictions", padding="10")
        prediction_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.prediction_text = tk.Text(prediction_frame, height=20, width=40)
        self.prediction_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.main_frame, textvariable=self.status_var)
        status_bar.pack(fill=tk.X, pady=5)
        
        # Initialize auto-refresh
        self.auto_refresh_id = None

    def fetch_realtime_data(self):
        """Fetch real-time data from the API endpoint."""
        try:
            self.status_var.set("Fetching data...")
            response = requests.get('https://vtsvcnode1.xyz/api/get-data', timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    self.status_var.set("Data fetched successfully")
                    return data.get('data', [])
            self.status_var.set("Failed to fetch data")
            return []
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            return []

    def process_data(self, data):
        """Process the API data into a pandas DataFrame aligned to 5-minute intervals."""
        if not data:
            return pd.DataFrame(columns=['Timestamp', 'Vehicle_count'])
        
        # Create DataFrame and convert timestamps
        df = pd.DataFrame(data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Vehicle_count'] = pd.to_numeric(df['Vehicle_count'], errors='coerce').fillna(0).astype(int)
        
        # Round timestamps to 5-minute intervals
        df['Timestamp'] = df['Timestamp'].apply(lambda x: x.replace(
            minute=(x.minute // 5) * 5,
            second=0,
            microsecond=0
        ))
        
        # Get the latest complete 5-minute interval
        last_interval = get_last_complete_interval()
        
        # Group by 5-minute intervals and get the latest 10 intervals
        df = df.groupby('Timestamp')['Vehicle_count'].sum().reset_index()
        df = df[df['Timestamp'] <= last_interval]
        df = df.sort_values('Timestamp', ascending=True).tail(10)
        
        return df

    def get_last_n_timesteps(self, df, n=10):
        """Get the last n timesteps from the data."""
        last_n = df.tail(n)
        if len(last_n) < n:
            # Calculate the start time for padding
            last_time = last_n['Timestamp'].min() if not last_n.empty else get_last_complete_interval()
            padding_start = last_time - timedelta(minutes=5 * (n - len(last_n)))
            
            padding = pd.DataFrame({
                'Vehicle_count': [0] * (n - len(last_n)),
                'Timestamp': [padding_start + timedelta(minutes=5*i) for i in range(n - len(last_n))]
            })
            last_n = pd.concat([padding, last_n]).reset_index(drop=True)
        
        return last_n['Vehicle_count'].values.reshape(1, n, 1)

    def update_display(self):
        """Update both current data and predictions display."""
        # Clear displays
        self.current_data_text.delete(1.0, tk.END)
        self.prediction_text.delete(1.0, tk.END)
        
        # Fetch and process data
        raw_data = self.fetch_realtime_data()
        df = self.process_data(raw_data)
        
        if df.empty:
            self.current_data_text.insert(tk.END, "No data available\n")
            return
            
        input_sequence = self.get_last_n_timesteps(df, 10)
        
        # Update current data display
        self.current_data_text.insert(tk.END, "Last 10 Vehicle Counts:\n\n")
        for _, row in df.iterrows():
            self.current_data_text.insert(tk.END, 
                f"{row['Timestamp'].strftime('%H:%M:%S')}: {int(row['Vehicle_count'])} vehicles\n")
        
        # Make and display predictions
        self.prediction_text.insert(tk.END, "Predictions for next 5 timesteps:\n\n")
        predictions = predict(input_sequence)
        
        if predictions is not None:
            last_time = df['Timestamp'].max()
            for i, pred in enumerate(predictions, 1):
                future_time = last_time + timedelta(minutes=5*i)
                self.prediction_text.insert(tk.END, 
                    f"{future_time.strftime('%H:%M:%S')}: {int(pred)} vehicles\n")
        else:
            self.prediction_text.insert(tk.END, "Prediction failed\n")

    def refresh_data(self):
        """Manually refresh the data display."""
        self.update_display()

    def toggle_auto_refresh(self):
        """Toggle automatic refresh on/off."""
        if self.auto_refresh_var.get():
            self.start_auto_refresh()
        else:
            self.stop_auto_refresh()

    def start_auto_refresh(self):
        """Start automatic refresh every 30 seconds."""
        def refresh():
            if self.auto_refresh_var.get():
                self.update_display()
                self.auto_refresh_id = self.root.after(30000, refresh)
        
        self.auto_refresh_id = self.root.after(30000, refresh)
        self.status_var.set("Auto-refresh enabled")

    def stop_auto_refresh(self):
        """Stop automatic refresh."""
        if self.auto_refresh_id:
            self.root.after_cancel(self.auto_refresh_id)
            self.auto_refresh_id = None
        self.status_var.set("Auto-refresh disabled")

    def run(self):
        """Start the GUI application."""
        self.update_display()  # Initial display
        self.root.mainloop()

if __name__ == "__main__":
    app = TrafficMonitorGUI()
    app.run()