"""
Traffic prediction functionality for the simulation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TrafficPredictor:
    """Simple traffic prediction module that acts as a bridge to ML models"""
    def __init__(self):
        self.prediction_data = {
            "historical": [],  # List of historical vehicle counts
            "predictions": [],  # List of predicted vehicle counts
            "timestamps": []   # List of timestamps for predictions
        }
    
    def run_prediction(self, source_type, data, seq_length=10, horizon=5):
        """
        Run traffic prediction using historical data
        
        Args:
            source_type: Either "simulation" or "csv"
            data: Either a list of vehicle counts or path to CSV file
            seq_length: Length of input sequence for prediction
            horizon: Number of future time steps to predict
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Process data based on source type
            if source_type == "simulation":
                # Data is already a list of vehicle counts
                vehicle_counts = data
                
                # Ensure we have enough data
                if len(vehicle_counts) < seq_length:
                    return {
                        "status": "error", 
                        "error": f"Not enough data points. Need {seq_length}, got {len(vehicle_counts)}"
                    }
                
                # Use the last seq_length points as input
                input_sequence = vehicle_counts[-seq_length:]
                
            elif source_type == "csv":
                # Data is a path to CSV file
                try:
                    df = pd.read_csv(data)
                    
                    # Try to find flow column
                    flow_columns = [col for col in df.columns if 'Flow' in col]
                    if flow_columns:
                        flow_column = "Flow (Veh/5 Minutes)" if "Flow (Veh/5 Minutes)" in flow_columns else flow_columns[0]
                        input_sequence = df[flow_column].values[-seq_length:].tolist()
                    else:
                        # If no flow column found, try to use third column (index 2) which often contains vehicle counts
                        if df.shape[1] > 2:
                            input_sequence = df.iloc[-seq_length:, 2].values.tolist()
                        else:
                            return {
                                "status": "error", 
                                "error": "Could not find vehicle count data in CSV"
                            }
                except Exception as e:
                    return {"status": "error", "error": f"Error processing CSV: {str(e)}"}
            
            else:
                return {"status": "error", "error": f"Unknown source type: {source_type}"}
            
            # Store the input sequence
            self.prediction_data["historical"] = input_sequence
            
            # Generate simple predictions - in a real system, this would call an ML model
            predictions = self.simple_prediction_model(input_sequence, horizon)
            
            # Generate timestamps for predictions (every 5 minutes into the future)
            current_time = datetime.now()
            timestamps = []
            for i in range(horizon):
                future_time = current_time + timedelta(minutes=5*(i+1))
                timestamps.append(future_time.strftime("%H:%M:%S"))
            
            # Store the predictions and timestamps
            self.prediction_data["predictions"] = predictions
            self.prediction_data["timestamps"] = timestamps
            
            return {
                "status": "success",
                "predictions": predictions,
                "timestamps": timestamps
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}
    
    def simple_prediction_model(self, input_sequence, horizon):
        """
        A simple model to generate predictions based on historical data
        In a real system, this would be replaced with an actual ML model
        
        Args:
            input_sequence: List of historical vehicle counts
            horizon: Number of future time steps to predict
            
        Returns:
            List of predicted vehicle counts
        """
        # Convert input to numpy array
        input_array = np.array(input_sequence)
        
        # Calculate basic statistics
        mean_value = np.mean(input_array)
        std_value = np.std(input_array)
        last_value = input_array[-1]
        
        # Calculate trend
        if len(input_array) > 1:
            trend = (input_array[-1] - input_array[0]) / len(input_array)
        else:
            trend = 0
            
        # Generate predictions with trend and some randomness
        predictions = []
        for i in range(horizon):
            # Base prediction: last value + trend
            pred = last_value + trend * (i + 1)
            
            # Add some randomness based on the standard deviation
            random_factor = np.random.normal(0, std_value * 0.1)
            pred += random_factor
            
            # Ensure prediction is not negative
            pred = max(0, pred)
            
            predictions.append(pred)
        
        return predictions
    
    def get_latest_predictions(self):
        """Get the latest prediction data"""
        return self.prediction_data

# Create a singleton instance
traffic_predictor = TrafficPredictor()

def run_prediction(source_type, data, seq_length=10, horizon=5):
    """Wrapper function to call the traffic predictor"""
    return traffic_predictor.run_prediction(source_type, data, seq_length, horizon)

def get_latest_predictions():
    """Wrapper function to get the latest predictions"""
    return traffic_predictor.get_latest_predictions()