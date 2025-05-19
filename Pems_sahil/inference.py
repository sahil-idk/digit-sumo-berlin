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
    """
    Process data from multiple folders and make predictions with proper evaluation metrics
    
    Args:
        folder_paths: List of folders containing traffic data files
        window_size: Size of the sliding window for input sequences
        
    Returns:
        None (results are saved to CSV)
    """
    # Prepare data for all models at the start
    lstm_data, gcn_lstm_data, common_index = create_input_sequences(folder_paths)

    num_timesteps = lstm_data.shape[0]  # Assuming both datasets have the same number of timesteps
    results = []  # List to store results for saving later
    
    # Create directory for results if it doesn't exist
    os.makedirs('Knowledge', exist_ok=True)
    
    # Initialize metrics tracking
    all_metrics = {
        'lstm': {'MAE': [], 'RMSE': [], 'MAPE': []},
        'gcn_lstm': {'MAE': [], 'RMSE': [], 'MAPE': []},
        'linear': {'MAE': [], 'RMSE': [], 'MAPE': []}
    }
    
    # Iterate over data using a sliding window
    for i in range(num_timesteps - window_size + 1):
        # Load the model type from the model.csv for each window
        model = safe_read_csv('C:/Users/sahil/Sumo/2024-10-20-18-27-19/Pems_sahil/csv_files/model.csv')
        model_type = model.iloc[0, 0]

        # Choose the right data based on the model type
        if model_type == 'lstm':
            data = lstm_data
            scaler_obj = scaler  # Use the global scaler
        elif model_type == 'gcn_lstm':
            data = gcn_lstm_data
            scaler_obj = scaler  # Use the global scaler for simplicity
        elif model_type == 'linear':
            data = lstm_data  # Linear model uses same format as LSTM
            scaler_obj = scaler
        else:
            print(f"Invalid model type: {model_type}")
            continue

        start_time = time.time()
        
        # Get the prediction function for the current model
        predict_function = None
        if model_type == 'lstm':
            from predict.lstm_predict import predict as lstm_predict
            predict_function = lstm_predict
        elif model_type == 'gcn_lstm':
            from predict.gcn_lstm_predict import predict as gcn_lstm_predict
            predict_function = gcn_lstm_predict
        elif model_type == 'linear':
            from predict.linear_predict import predict as linear_predict
            predict_function = linear_predict
        
        if predict_function is None:
            print(f"Could not load prediction function for model type: {model_type}")
            continue
            
        # Extract the current window with proper formatting
        # Ensure consistent data format across all models
        data_window = data[i:i + window_size]
        
        # For LSTM and Linear models, the input needs to be (batch_size, sequence_length, features)
        if model_type in ['lstm', 'linear']:
            # Reshape to (1, window_size, 1) for LSTM/Linear
            input_data = data_window.reshape(1, window_size, 1)
        else:  # For GCN-LSTM
            # GCN-LSTM expects a different format
            input_data = data_window

        # Make predictions with proper error handling
        try:
            predictions = predict_function(input_data)
        except Exception as e:
            print(f"Error making prediction with {model_type} model: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Calculate time taken
        each_time = time.time() - start_time
        global_end_time = time.time() - global_start_time

        # Extract the prediction for the specified timestep (5th)
        prediction_idx = 4  # Index 4 for the 5th timestep
        if len(predictions) > prediction_idx:
            prediction_value = predictions[prediction_idx]

            # Get the timestamp for this prediction
            prediction_timestamp = common_index[i + prediction_idx] if i + prediction_idx < len(common_index) else None
            
            # Get the actual data for this timestep (for evaluation)
            actual_value = None
            
            # The actual value should be at position i + window_size (the point after the window)
            if i + window_size < len(data):
                # Get the actual value and inverse transform it if needed
                if model_type in ['lstm', 'linear']:
                    actual_data = scaler_obj.inverse_transform(data[i + window_size].reshape(-1, 1))
                    actual_value = actual_data[0, 0]
                elif model_type == 'gcn_lstm':
                    # For GCN-LSTM, we need the node index (using 1 as in original code)
                    node_idx = 1  # Use the second node as in original code
                    actual_data = scaler_obj.inverse_transform(data[i + window_size, node_idx].reshape(-1, 1))
                    actual_value = actual_data[0, 0]
            
            # Calculate evaluation metrics if we have actual data
            metrics = {}
            if actual_value is not None:
                # Calculate metrics
                mae = abs(prediction_value - actual_value)
                rmse = (prediction_value - actual_value) ** 2
                
                # MAPE calculation with safeguard against division by zero
                mape = 0 if actual_value == 0 else 100 * abs((prediction_value - actual_value) / actual_value)
                
                # Store the metrics
                metrics = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape
                }
                
                # Add to the aggregate metrics
                all_metrics[model_type]['MAE'].append(mae)
                all_metrics[model_type]['RMSE'].append(rmse)
                all_metrics[model_type]['MAPE'].append(mape)
            
            # Store all information in the results
            result_entry = {
                'timestamp': prediction_timestamp,
                'model_type': model_type,
                'predicted_value': prediction_value,
                'actual_value': actual_value if actual_value is not None else 'N/A',
                'time_for_pred': each_time,
                'global_time': global_end_time
            }
            
            # Add metrics if available
            if metrics:
                for metric, value in metrics.items():
                    result_entry[metric] = value
            
            # Add to results list
            results.append(result_entry)
            
            # Print the prediction
            metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()]) if metrics else "No metrics"
            print(f"Timestamp: {prediction_timestamp}, Model: {model_type}, "
                  f"Predicted: {prediction_value:.2f}, Actual: {actual_value if actual_value is not None else 'N/A'}, "
                  f"Metrics: {metrics_str}")
            
            # Save prediction results to CSV
            # Create a dictionary with all the prediction details
            csv_row = {
                'timestamp': prediction_timestamp,
                'model_type': model_type,
                'predicted_value': prediction_value,
                'actual_value': actual_value if actual_value is not None else '',
                'time_for_pred': each_time,
                'global_time': global_end_time
            }
            
            # Add metrics to the CSV if available
            if metrics:
                for metric, value in metrics.items():
                    csv_row[metric] = value
            
            # Convert to DataFrame for easy CSV writing
            pd.DataFrame([csv_row]).to_csv(
                'C:/Users/sahil/Sumo/2024-10-20-18-27-19/Pems_sahil/Knowledge/prediction_results.csv', 
                index=False, 
                mode='a', 
                header=False
            )
        
        # Sleep to avoid overwhelming the system
        time.sleep(0.1)  # Reduced sleep time for faster processing
    
    # Calculate and print the aggregate metrics at the end
    print("\nAggregate Metrics:")
    for model_type, metrics in all_metrics.items():
        if all(len(values) > 0 for values in metrics.values()):
            avg_mae = sum(metrics['MAE']) / len(metrics['MAE'])
            avg_rmse = (sum(metrics['RMSE']) / len(metrics['RMSE'])) ** 0.5  # Square root for RMSE
            avg_mape = sum(metrics['MAPE']) / len(metrics['MAPE'])
            
            print(f"{model_type.upper()} Model:")
            print(f"  Average MAE: {avg_mae:.4f}")
            print(f"  Average RMSE: {avg_rmse:.4f}")
            print(f"  Average MAPE: {avg_mape:.4f}%")
            
            # Save the aggregate metrics to a separate CSV file
            metrics_df = pd.DataFrame({
                'model_type': [model_type],
                'avg_mae': [avg_mae],
                'avg_rmse': [avg_rmse],
                'avg_mape': [avg_mape],
                'num_predictions': [len(metrics['MAE'])]
            })
            
            metrics_df.to_csv(
                'C:/Users/sahil/Sumo/2024-10-20-18-27-19/Pems_sahil/Knowledge/aggregate_metrics.csv',
                mode='a',
                header=not os.path.exists('C:/Users/sahil/Sumo/2024-10-20-18-27-19/Pems_sahil/Knowledge/aggregate_metrics.csv'),
                index=False
            )

# Define input data paths
folder_paths = [input_folder_402214, input_folder_402510, input_folder_402835, input_folder_414025]

# Start the process and predict in 10-timestep batches
process_and_predict(folder_paths, window_size=10)

