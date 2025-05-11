import os
import pandas as pd
import numpy as np
from predict.lstm_predict import predict as lstm_predict
from sklearn.preprocessing import MinMaxScaler
import time
import torch



global_start_time = time.time()
# Define input data paths
input_folder_402214 = 'C:/Users/sahil/Sumo/2024-10-20-18-27-19/Pems_sahil/Data/402214'
input_folder_402510 = 'C:/Users/sahil/Sumo/2024-10-20-18-27-19/Pems_sahil/Data/402510'
input_folder_402835 = 'C:/Users/sahil/Sumo/2024-10-20-18-27-19/Pems_sahil/Data/402835'
input_folder_414025 = 'C:/Users/sahil/Sumo/2024-10-20-18-27-19/Pems_sahil/Data/414025'

# Load the saved scaler
scaler_path = 'C:/Users/sahil/Sumo/2024-10-20-18-27-19/Pems_sahil/models/lstm/lstm_scaler.pth'
scaler = torch.load(scaler_path, weights_only=False)


# Define result path
result_path = 'Results'
def safe_read_csv(file_path):
    try:
        return pd.read_csv(file_path, header=None)
    except pd.errors.EmptyDataError:
        print(f"Warning: The file {file_path} is empty. Using default model settings.")
        return pd.DataFrame(data=[["lstm"]])

# -------- Data Loading Function --------
def load_data(folder_path):
    all_data = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            if '5 Minutes' in df.columns and 'Flow (Veh/5 Minutes)' in df.columns:
                df['5 Minutes'] = pd.to_datetime(df['5 Minutes'], errors='coerce')
                df = df.set_index('5 Minutes')
                all_data.append(df[['Flow (Veh/5 Minutes)']])
    
    combined_data = pd.concat(all_data)
    combined_data = combined_data.sort_index()
    combined_data = combined_data.dropna()
    
    return combined_data

# Preprocess input sequence for GCN-LSTM model
def prepare_gcn_lstm_data(data_list):
    node_data = np.stack([data.values for data in data_list], axis=0)
    scalers = [MinMaxScaler() for _ in range(node_data.shape[0])]
    for i in range(node_data.shape[0]):
        node_data[i] = scalers[i].fit_transform(node_data[i].reshape(-1, 1))
    return np.transpose(node_data, (1, 0, 2))  # Shape: (num_timesteps, num_nodes, num_features)

# Preprocess input sequence for LSTM model
def prepare_lstm_data(data_list):
    node_data = data_list[0].values
    scaler = MinMaxScaler()
    node_data = scaler.fit_transform(node_data.reshape(-1, 1))
    return node_data

# Function to create input sequences for both models
def create_input_sequences(folder_paths):
    data_list = []
    for folder in folder_paths:
        data_list.append(load_data(folder))
    
    common_start = max(data.index.min() for data in data_list)
    common_end = min(data.index.max() for data in data_list)
    common_index = pd.date_range(start=common_start, end=common_end, freq='5T')

    for i in range(len(data_list)):
        data_list[i] = data_list[i].reindex(common_index).interpolate()

    lstm_data = prepare_lstm_data(data_list)
    gcn_lstm_data = prepare_gcn_lstm_data(data_list)
    
    return lstm_data, gcn_lstm_data, common_index

# Main function to process and predict dynamically
def process_and_predict(folder_paths, window_size=10):
    # Prepare data for both models at the start
    lstm_data, gcn_lstm_data, common_index = create_input_sequences(folder_paths)

    num_timesteps = lstm_data.shape[0]  # Assuming both datasets have the same number of timesteps
    results = []  # List to store results for saving later

    while True:
        # Iterate over data using a sliding window
        for i in range(num_timesteps - window_size + 1):
            # Load the model type from the model.csv for each window
            model = safe_read_csv('C:/Users/sahil/Sumo/2024-10-20-18-27-19/Pems_sahil/csv_files/model.csv')
            model_type = model.iloc[0, 0]

            # Choose the right data and prediction function based on the model type
            if model_type == 'lstm':
                data = lstm_data
            else:
                print("Invalid model")
                continue

            start_time = time.time()
            predict_function = globals()[f"{model_type}_predict"]
            
            # Extract the current window
            data_window = data[i:i + window_size]
            predictions = predict_function(data_window)

            each_time = time.time() - start_time
            global_end_time = time.time()

            # Extract the 5th timestep prediction
            if len(predictions) >= 5:
                prediction_5th = predictions[4]  # Index 4 for the 5th timestep

                # Get the timestamp for the 5th timestep
                timestamp = common_index[i + 4]  # Index for 5th timestep in the current window

                # Inverse transform the actual data
                if model_type == 'lstm' or model_type == 'linear':
                    actual_data = scaler.inverse_transform(data_window)
                    actual_vehicle_count = actual_data[4, 0]  # 5th timestep, first feature
                else:  # For gcn_lstm
                    
                    actual_data = scaler.inverse_transform(data_window[:, 1, :])
                    # print(actual_data.shape)
                    actual_vehicle_count = actual_data[4, 0]  # 5th timestep, first feature

                # Store results in the list
                results.append({
                    'timestamp': timestamp,
                    'model_type': model_type,
                    'predicted_count': prediction_5th,
                    'actual_count': actual_vehicle_count,
                    'time_for_pred': each_time,
                    'global_time': global_end_time - global_start_time
                })

                print(f"Timestamp: {timestamp}, Model: {model_type}, Predicted: {prediction_5th}, Actual: {actual_vehicle_count}")
                pd.DataFrame([results[-1]]).to_csv('C:/Users/sahil/Sumo/2024-10-20-18-27-19/Pems_sahil/Knowledge/prediction_results.csv', index=False, mode='a', header=False)

            time.sleep(1)  # Simulating some processing delay

        # Clear results for the next cycle
        results.clear()

        # Continue to the next iteration without waiting

# Define input data paths
folder_paths = [input_folder_402214, input_folder_402510, input_folder_402835, input_folder_414025]

# Start the process and predict in 10-timestep batches
process_and_predict(folder_paths, window_size=10)

