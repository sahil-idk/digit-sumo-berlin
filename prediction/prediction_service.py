"""
Prediction service for traffic data analysis
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictionService:
    """Service for making traffic predictions based on historical or simulation data"""
    
    def __init__(self):
        """Initialize the prediction service"""
        self.input_sequence = []
        self.last_predictions = None
        self.last_prediction_time = None
        
        # Initialize with default model parameters
        self.model_loaded = False
        self.try_load_model()
    
    def try_load_model(self):
        """Attempt to load the ML model"""
        try:
            # Here we would normally load a trained ML model
            # For demonstration purposes, we'll use a simple forecasting approach
            # In a real system, you might use something like:
            #   from tensorflow.keras.models import load_model
            #   self.model = load_model('models/traffic_lstm.h5')
            
            self.model_loaded = True
            logger.info("Prediction service initialized (using simple forecasting model)")
            return True
        except Exception as e:
            logger.error(f"Failed to load prediction model: {e}")
            return False
    
    def preprocess_data(self, data, seq_length):
        """Preprocess the input data for prediction"""
        if len(data) < seq_length:
            # Pad with zeros if not enough data
            padding = [0] * (seq_length - len(data))
            padded_data = padding + data
            logger.warning(f"Input data too short, padded with zeros ({len(padding)} elements)")
            return padded_data[-seq_length:]
        else:
            # Use the last seq_length data points
            return data[-seq_length:]
    
    def get_timestamps(self, horizon, interval_minutes=5):
        """Generate future timestamps for predictions"""
        timestamps = []
        now = datetime.now()
        
        for i in range(1, horizon + 1):
            future_time = now + timedelta(minutes=i * interval_minutes)
            timestamps.append(future_time.strftime("%H:%M:%S"))
        
        return timestamps
    
    def simple_forecast(self, data, horizon=5):
        """Simple forecasting method using moving average and trend"""
        if len(data) < 3:
            return [data[-1]] * horizon
        
        # Calculate the average
        avg = np.mean(data)
        
        # Calculate the overall trend (slope)
        if len(data) >= 5:
            x = np.arange(len(data))
            y = np.array(data)
            trend = np.polyfit(x, y, 1)[0]
        else:
            # Simple trend calculation for smaller datasets
            trend = (data[-1] - data[0]) / (len(data) - 1)
        
        # Predict future values based on the last value, trend, and reversion to mean
        predictions = []
        last_value = data[-1]
        
        for i in range(horizon):
            # More weight on trend for short-term, more weight on average for long-term
            mean_weight = min(0.1 * (i + 1), 0.5)  # Gradually increase mean reversion
            trend_weight = max(1.0 - 0.1 * i, 0.4)  # Gradually decrease trend influence
            
            # Predict the next value
            next_value = last_value + (trend * trend_weight)
            # Apply mean reversion
            next_value = next_value * (1 - mean_weight) + avg * mean_weight
            
            # Ensure non-negative values for vehicle counts
            next_value = max(0, next_value)
            
            predictions.append(next_value)
            last_value = next_value
        
        return predictions
    
    def run_prediction(self, source_type, data, seq_length=10, horizon=5):
        """Run the prediction model on the provided data"""
        try:
            # Process data based on source type
            if source_type == "simulation":
                # Data is a list of vehicle counts
                if not data or len(data) == 0:
                    return {
                        "status": "error",
                        "error": "No simulation data available"
                    }
                
                # Preprocess data
                self.input_sequence = self.preprocess_data(data, seq_length)
                
            elif source_type == "csv":
                # Data is a path to a CSV file
                if not os.path.exists(data):
                    return {
                        "status": "error",
                        "error": f"CSV file not found: {data}"
                    }
                
                try:
                    # Load and preprocess CSV data
                    df = pd.read_csv(data)
                    
                    # Try to find the vehicle count column
                    vehicle_col = None
                    for col in df.columns:
                        if "flow" in col.lower() or "count" in col.lower() or "vehicle" in col.lower():
                            vehicle_col = col
                            break
                    
                    if vehicle_col is None and 'Flow (Veh/5 Minutes)' in df.columns:
                        vehicle_col = 'Flow (Veh/5 Minutes)'
                    
                    if vehicle_col is None and len(df.columns) > 1:
                        # Assume the second column might contain vehicle counts
                        vehicle_col = df.columns[1]
                    
                    if vehicle_col:
                        vehicle_counts = df[vehicle_col].values.tolist()
                        self.input_sequence = self.preprocess_data(vehicle_counts, seq_length)
                    else:
                        return {
                            "status": "error",
                            "error": "Could not identify vehicle count column in CSV"
                        }
                    
                except Exception as e:
                    return {
                        "status": "error",
                        "error": f"Error processing CSV file: {e}"
                    }
            else:
                return {
                    "status": "error",
                    "error": f"Unknown data source type: {source_type}"
                }
            
            # Make predictions
            if self.model_loaded:
                # For a real ML model, you would use something like:
                # input_tensor = np.reshape(self.input_sequence, (1, seq_length, 1))
                # predictions = self.model.predict(input_tensor)[0].tolist()
                
                # Instead, use our simple forecasting method
                predictions = self.simple_forecast(self.input_sequence, horizon)
                
                # Generate timestamps
                timestamps = self.get_timestamps(horizon)
                
                # Save the results
                self.last_predictions = {
                    "input_sequence": self.input_sequence,
                    "predictions": predictions,
                    "timestamps": timestamps,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                self.last_prediction_time = time.time()
                
                return {
                    "status": "success",
                    "input_sequence": self.input_sequence,
                    "predictions": predictions,
                    "timestamps": timestamps
                }
            else:
                return {
                    "status": "error",
                    "error": "Prediction model not loaded"
                }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": f"Prediction error: {str(e)}"
            }
    
    def get_latest_predictions(self):
        """Get the most recent prediction results"""
        if self.last_predictions and self.last_prediction_time:
            # Check if predictions are recent (less than 5 minutes old)
            if time.time() - self.last_prediction_time < 300:
                return self.last_predictions
        
        return None