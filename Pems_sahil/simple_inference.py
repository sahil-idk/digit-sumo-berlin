import requests
import numpy as np
import pandas as pd
from datetime import datetime
import time
from predict.lstm_predict import predict

def fetch_api_data(api_url):
    """Fetch data from the API"""
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()['data']
    except Exception as e:
        print(f"Error fetching API data: {e}")
        return None

def process_api_data(api_data, window_size=10):
    """Process API data into sequence"""
    # Convert to DataFrame
    df = pd.DataFrame(api_data)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Sort by timestamp and get latest window_size records
    df = df.sort_values('Timestamp').tail(window_size)
    
    # Extract vehicle counts as sequence
    sequence = df['Vehicle_count'].values.reshape(-1, 1)
    return sequence, df['Timestamp'].iloc[-1]

def main():
    api_url = "https://vtsvcnode1.xyz/api/get-data"
    window_size = 10  # Size of input sequence needed for LSTM
    
    print("Starting prediction loop...")
    print(f"API URL: {api_url}")
    
    while True:
        try:
            # 1. Fetch data from API
            print("\nFetching data from API...")
            api_data = fetch_api_data(api_url)
            
            if api_data:
                # 2. Process data into sequence
                sequence, last_timestamp = process_api_data(api_data, window_size)
                
                if len(sequence) == window_size:
                    # 3. Make prediction
                    predictions = predict(sequence)
                    
                    if predictions is not None:
                        print("\nPrediction Results:")
                        print(f"Timestamp: {last_timestamp}")
                        print("Predictions for next 5 timesteps:")
                        for i, pred in enumerate(predictions, 1):
                            print(f"t+{i}: {pred:.2f} vehicles")
                        print("-" * 50)
                    else:
                        print("Prediction failed")
                else:
                    print(f"Not enough data points. Have {len(sequence)}, need {window_size}")
            else:
                print("No data received from API")
            
            # Wait for 5 minutes before next prediction
            print("\nWaiting 5 minutes before next prediction...")
            time.sleep(300)
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            print("Retrying in 60 seconds...")
            time.sleep(60)

if __name__ == "__main__":
    main()