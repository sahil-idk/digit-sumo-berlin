import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from sklearn.preprocessing import MinMaxScaler
import torch
from typing import List, Dict
import json
from torch.serialization import add_safe_globals
from sklearn.preprocessing._data import MinMaxScaler

# Add MinMaxScaler to PyTorch's safe globals
add_safe_globals([MinMaxScaler])

# Import your prediction functions
from predict.lstm_predict import predict as lstm_predict
from predict.gcn_lstm_predict import predict as gcn_lstm_predict
from predict.linear_predict import predict as linear_predict
from predict.Digut import predict as digit_predict

class RealTimePredictor:
    def __init__(self, api_url: str, window_size: int = 10):
        self.api_url = api_url
        self.window_size = window_size
        self.data_buffer = []
        
        # Load the saved scaler with weights_only=False
        try:
            self.scaler = torch.load('./models/linear/scaler.pth', weights_only=False)
        except Exception as e:
            print(f"Error loading scaler: {e}")
            # Create a new scaler if loading fails
            self.scaler = MinMaxScaler()
        
    def fetch_api_data(self) -> List[Dict]:
        """Fetch data from the API"""
        try:
            response = requests.get(self.api_url)
            response.raise_for_status()
            return response.json()['data']
        except Exception as e:
            print(f"Error fetching API data: {e}")
            return None

    def process_api_data(self, api_data: List[Dict]) -> pd.DataFrame:
        """Process API data into a DataFrame"""
        df = pd.DataFrame(api_data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp')
        
        # Get the most recent window_size records
        df = df.tail(self.window_size)
        
        return df[['Vehicle_count', 'Timestamp']]

    def prepare_model_input(self, df: pd.DataFrame):
        """Prepare input data for model prediction"""
        # Extract vehicle counts
        vehicle_counts = df['Vehicle_count'].values.reshape(-1, 1)
        
        # Fit and transform if scaler is new
        if not hasattr(self.scaler, 'n_samples_seen_') or self.scaler.n_samples_seen_ is None:
            self.scaler.fit(vehicle_counts)
        
        # Scale the data
        scaled_data = self.scaler.transform(vehicle_counts)
        
        # Prepare data for different model types
        lstm_data = scaled_data
        gcn_lstm_data = np.expand_dims(scaled_data, axis=1)  # Add node dimension
        
        return lstm_data, gcn_lstm_data

    def get_model_type(self) -> str:
        """Get current model type from model.csv"""
        try:
            model_df = pd.read_csv('csv_files/model.csv', header=None)
            return model_df.iloc[0, 0]
        except Exception as e:
            print(f"Error reading model type: {e}")
            return 'lstm'  # Default to LSTM if there's an error

    def make_prediction(self, lstm_data, gcn_lstm_data):
        """Make prediction using the specified model"""
        model_type = self.get_model_type()
        
        # Select appropriate data and prediction function based on model type
        if model_type in ['gcn_lstm', 'digit']:
            data = gcn_lstm_data
            predict_function = gcn_lstm_predict if model_type == 'gcn_lstm' else digit_predict
        else:
            data = lstm_data
            predict_function = lstm_predict if model_type == 'lstm' else linear_predict
        
        try:
            # Make prediction
            predictions = predict_function(data)
            
            # Get 5th timestep prediction
            prediction_5th = predictions[4] if len(predictions) >= 5 else predictions[-1]
            
            # Inverse transform the prediction
            prediction_5th = self.scaler.inverse_transform([[prediction_5th]])[0][0]
            
            return prediction_5th, model_type
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None, model_type

    def run_prediction_loop(self):
        """Main prediction loop"""
        print("Starting prediction loop...")
        print(f"API URL: {self.api_url}")
        print(f"Window size: {self.window_size}")
        
        while True:
            try:
                # Fetch and process API data
                print("Fetching data from API...")
                api_data = self.fetch_api_data()
                if not api_data:
                    print("No data received from API")
                    time.sleep(60)  # Wait a minute before retrying
                    continue
                
                # Process the data
                print("Processing data...")
                df = self.process_api_data(api_data)
                
                # Check if we have enough data points
                if len(df) < self.window_size:
                    print(f"Not enough data points. Have {len(df)}, need {self.window_size}")
                    time.sleep(60)
                    continue
                
                # Prepare input data
                lstm_data, gcn_lstm_data = self.prepare_model_input(df)
                
                # Make prediction
                prediction, model_type = self.make_prediction(lstm_data, gcn_lstm_data)
                
                if prediction is not None:
                    # Get current timestamp
                    current_time = datetime.now()
                    
                    # Store result
                    result = {
                        'timestamp': current_time,
                        'model_type': model_type,
                        'predicted_count': prediction,
                        'actual_count': df['Vehicle_count'].iloc[-1],  # Last known count
                        'time_for_pred': time.time()
                    }
                    
                    # Save prediction to CSV
                    pd.DataFrame([result]).to_csv(
                        'Knowledge/prediction_results.csv',
                        mode='a',
                        header=False,
                        index=False
                    )
                    
                    print(f"\nPrediction Results:")
                    print(f"Timestamp: {current_time}")
                    print(f"Model Type: {model_type}")
                    print(f"Predicted Vehicle Count: {prediction:.2f}")
                    print(f"Actual Vehicle Count: {df['Vehicle_count'].iloc[-1]}")
                    print("-" * 50)
                
                # Wait for 5 minutes before next prediction
                print("\nWaiting 5 minutes before next prediction...")
                time.sleep(300)
                
            except Exception as e:
                print(f"Error in prediction loop: {e}")
                print("Retrying in 60 seconds...")
                time.sleep(60)

def main():
    api_url = "https://vtsvcnode1.xyz/api/get-data"
    predictor = RealTimePredictor(api_url, window_size=10)
    predictor.run_prediction_loop()

if __name__ == "__main__":
    main()