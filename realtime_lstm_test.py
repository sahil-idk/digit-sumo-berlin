import requests
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys
import json
from typing import List, Dict

# Add the predict directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
predict_dir = os.path.join(current_dir, "Pems_sahil", "predict")
sys.path.append(predict_dir)

from lstm_predict import predict

def fetch_realtime_data() -> List[Dict]:
    """Fetch real-time data from the API endpoint."""
    try:
        response = requests.get('https://vtsvcnode1.xyz/api/get-data')
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                return data.get('data', [])
        return []
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def process_data(data: List[Dict]) -> pd.DataFrame:
    """Process the API data into a pandas DataFrame."""
    df = pd.DataFrame(data)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')
    df = df.drop_duplicates(subset=['Timestamp'], keep='last')
    return df

def get_last_n_timesteps(df: pd.DataFrame, n: int = 10) -> np.ndarray:
    """Get the last n timesteps from the data."""
    last_n = df.tail(n)
    if len(last_n) < n:
        print(f"Warning: Only {len(last_n)} timesteps available, padding with zeros")
        # Pad with zeros if we don't have enough data
        padding = pd.DataFrame({
            'Vehicle_count': [0] * (n - len(last_n)),
            'Timestamp': pd.date_range(
                start=df['Timestamp'].min() - pd.Timedelta(minutes=5 * (n - len(last_n))),
                periods=(n - len(last_n)),
                freq='5min'
            )
        })
        last_n = pd.concat([padding, last_n]).reset_index(drop=True)
    
    return last_n['Vehicle_count'].values.reshape(1, n, 1)

def main():
    print("Fetching real-time data...")
    raw_data = fetch_realtime_data()
    
    if not raw_data:
        print("No data received from API")
        return
    
    print("\nProcessing data...")
    df = process_data(raw_data)
    print(f"Total records: {len(df)}")
    
    print("\nGetting last 10 timesteps for prediction...")
    input_sequence = get_last_n_timesteps(df, 10)
    print("Input sequence shape:", input_sequence.shape)
    print("\nInput sequence values:")
    print(input_sequence.reshape(10, 1))
    
    print("\nMaking predictions...")
    predictions = predict(input_sequence)
    
    if predictions is not None:
        print("\nPredictions for next 5 timesteps:")
        current_time = df['Timestamp'].max()
        for i, pred in enumerate(predictions, 1):
            future_time = current_time + pd.Timedelta(minutes=5*i)
            print(f"{future_time.strftime('%Y-%m-%d %H:%M:%S')}: {pred:.2f} vehicles")
    else:
        print("\nPrediction failed. Check error messages above.")

if __name__ == "__main__":
    main()