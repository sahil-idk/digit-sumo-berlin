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

# Modify predict function import to avoid reload file issues
def custom_predict(input_sequence):
    """Wrapper for the predict function to handle model loading."""
    import torch
    from lstm_predict import LSTMModel
    
    # Load model and make prediction
    model_path = os.path.join(current_dir, 'models', 'lstm', 'lstm_model.pth')
    scaler_path = os.path.join(current_dir, 'models', 'lstm', 'lstm_scaler.joblib')
    
    try:
        # Load model
        model = LSTMModel(1, 128, 1, 5)  # input_dim, hidden_dim, output_dim, output_window
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Load scaler
        import joblib
        scaler = joblib.load(scaler_path)
        
        # Transform input
        input_scaled = scaler.transform(input_sequence.reshape(-1, 1))
        input_tensor = torch.tensor(input_scaled.reshape(1, 10, 1), dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            predictions = model(input_tensor)
            predictions_np = predictions.squeeze(0).numpy()
            predictions_np = predictions_np.reshape(-1, 1)
            predictions_np = scaler.inverse_transform(predictions_np).flatten()
            
        return predictions_np
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

class TrafficMonitorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Traffic Monitor and Prediction")
        self.root.geometry("800x600")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add refresh button
        ttk.Button(self.main_frame, text="Refresh Data", 
                  command=self.refresh_data).pack(pady=5)
        
        # Create display areas
        self.current_data_text = tk.Text(self.main_frame, height=15, width=40)
        self.current_data_text.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.BOTH)
        
        self.prediction_text = tk.Text(self.main_frame, height=15, width=40)
        self.prediction_text.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.BOTH)

        """Fetch and process data from API."""
        try:
            response = requests.get('https://vtsvcnode1.xyz/api/get-data', timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    df = pd.DataFrame(data.get('data', []))
                    if not df.empty:
                        # Convert timestamps and sort
                        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                        df['Vehicle_count'] = pd.to_numeric(df['Vehicle_count'], errors='coerce').fillna(0)
                        df = df.sort_values('Timestamp')
                        
                        # Get current time and round down to nearest 5 minutes
                        now = datetime.now()
                        current_interval = now.replace(
                            minute=(now.minute // 5) * 5,
                            second=0,
                            microsecond=0
                        )
                        
                        # Filter and get last 10 entries
                        df = df[df['Timestamp'] <= current_interval].tail(10)
                        return df
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def refresh_data(self):
        """Update the display with fresh data."""
        # Clear current displays
        self.current_data_text.delete(1.0, tk.END)
        self.prediction_text.delete(1.0, tk.END)
        
        # Fetch new data
        df = self.fetch_and_process_data()
        
        if df.empty:
            self.current_data_text.insert(tk.END, "No data available")
            return
        
        # Display current data
        self.current_data_text.insert(tk.END, "Last 10 Vehicle Counts:\n\n")
        for _, row in df.iterrows():
            self.current_data_text.insert(tk.END, 
                f"{row['Timestamp'].strftime('%H:%M:%S')}: {int(row['Vehicle_count'])} vehicles\n")
        
        # Make predictions
        input_sequence = df['Vehicle_count'].values
        predictions = custom_predict(input_sequence)
        
        if predictions is not None:
            self.prediction_text.insert(tk.END, "Predictions for next 5 timesteps:\n\n")
            last_time = df['Timestamp'].max()
            for i, pred in enumerate(predictions, 1):
                future_time = last_time + timedelta(minutes=5*i)
                self.prediction_text.insert(tk.END, 
                    f"{future_time.strftime('%H:%M:%S')}: {int(pred)} vehicles\n")
        else:
            self.prediction_text.insert(tk.END, "Prediction failed")

    def run(self):
        """Start the application."""
        self.refresh_data()  # Initial data load
        self.root.mainloop()

if __name__ == "__main__":
    app = TrafficMonitorGUI()
    app.run()