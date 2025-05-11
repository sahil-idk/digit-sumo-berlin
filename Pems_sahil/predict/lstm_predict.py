import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import torch  # For saving/loading the scaler

# Get the base directory
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Construct absolute paths
model_path = os.path.join("C:/Users/sahil/Sumo/2024-10-20-18-27-19/Pems_sahil/models/lstm/lstm_model.pth")
scaler_path = os.path.join("C:/Users/sahil/Sumo/2024-10-20-18-27-19/Pems_sahil/models/lstm/lstm_scaler.pth")  # Changed extension

print(f"Looking for model at: {model_path}")
print(f"Looking for scaler at: {scaler_path}")

# Create models directory if it doesn't exist
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Load or create the scaler
try:
    scaler = torch.load(scaler_path, weights_only=False)  # Use joblib to load
    print("Scaler loaded successfully")
except FileNotFoundError:
    print(f"Scaler file not found at {scaler_path}")
    print("Creating new scaler...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # Fit the scaler with some dummy data
    dummy_data = np.random.rand(100, 1) * 100  # Random values between 0 and 100
    scaler.fit(dummy_data)
    torch.dump(scaler, scaler_path)  # Use joblib to save
    print("New scaler created and saved")

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, output_window):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim * output_window)
        self.output_window = output_window
        self.output_dim = output_dim

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        output = output.view(-1, self.output_window, self.output_dim)
        return output

# Configuration
input_window = 10
output_window = 5
input_dim = 1
hidden_dim = 128
output_dim = 1

# Initialize and load/save the LSTM model
model = LSTMModel(input_dim, hidden_dim, output_dim, output_window)
try:
    model.load_state_dict(torch.load(model_path))
    print("Model loaded successfully")
except FileNotFoundError:
    print(f"Model file not found at {model_path}")
    print("Initializing new model...")
    # Save the initialized model
    torch.save(model.state_dict(), model_path)
    print("New model created and saved")

model.eval()

def predict(input_sequence):
    try:
        # Create a reload flag file path
        reload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reload')
        os.makedirs(reload_dir, exist_ok=True)
        reload_file = os.path.join(reload_dir, 'lstm.csv')

        # Check for model reload
        if os.path.exists(reload_file):
            try:
                reload_df = pd.read_csv(reload_file)
                if not reload_df.empty and reload_df.iloc[0, 0].strip().lower() == 'true':
                    model.load_state_dict(torch.load(model_path))
                    model.eval()
                    # Write false to the reload file
                    pd.DataFrame([["False"]], columns=['reload']).to_csv(reload_file, index=False)
            except Exception as e:
                print(f"Error checking reload file: {e}")

        # Transform input data
        input_data = input_sequence.reshape(-1, 1)
        input_scaled = scaler.transform(input_data)
        input_sequence_scaled = input_scaled.reshape(1, input_window, 1)

        # Convert to torch tensor
        input_tensor = torch.tensor(input_sequence_scaled, dtype=torch.float32)
        
        # Make predictions
        with torch.no_grad():
            predictions = model(input_tensor)

        # Convert predictions to numpy array
        predictions_np = predictions.squeeze(0).numpy()
        predictions_np = predictions_np.reshape(-1, output_dim)
        
        # Inverse transform the predictions
        predictions_np = scaler.inverse_transform(predictions_np).flatten()

        return predictions_np

    except Exception as e:
        print(f"Prediction error: {e}")
        print(f"Input sequence shape: {input_sequence.shape}")
        print(f"Input sequence type: {type(input_sequence)}")
        return None

if __name__ == "__main__":
    # Test the prediction with dummy data
    dummy_input = np.random.rand(1, 10, 1)
    print("\nTesting prediction with dummy data...")
    result = predict(dummy_input)
    print("Prediction result:", result)