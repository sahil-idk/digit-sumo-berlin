import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
model_path = '../models/lstm/lstm_model.pth'

# Load the saved scaler
scaler_path = '../models/lstm/lstm_scaler.pth'
scaler = torch.load(scaler_path)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, output_window):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim * output_window)
        self.output_window = output_window
        self.output_dim = output_dim

    def forward(self, x):
        # x shape: (batch_size, input_window, input_dim)
        lstm_out, _ = self.lstm(x)  # LSTM output
        lstm_out = lstm_out[:, -1, :]  # Take last LSTM output
        output = self.fc(lstm_out)  # (batch_size, output_dim * output_window)
        output = output.view(-1, self.output_window, self.output_dim)  # Reshape to (batch_size, output_window, output_dim)
        return output

# Configuration
input_window = 10  # Sequence length
output_window = 5  # How many steps ahead to predict
input_dim = 1  # Number of features
hidden_dim = 128  # Hidden units in LSTM
output_dim = 1  # Number of features to predict

# Initialize and load the LSTM model
model = LSTMModel(input_dim, hidden_dim, output_dim, output_window)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set model to evaluation mode

# Prediction function
def predict(input_sequence):
    try:
        model_reload = pd.read_csv('./predict/reload/lstm.csv').iloc[0, 0]
        if model_reload:
            model.load_state_dict(torch.load(model_path))
            model.eval()  # Set model to evaluation mode
            #write in csv false
            model_reload = pd.DataFrame(data=[["False"]])
        # Ensure input is a PyTorch tensor and has the correct shape
        input_sequence = torch.tensor(input_sequence, dtype=torch.float32)
        if input_sequence.ndim == 2:  # If input is [window_size, num_features]
            input_sequence = input_sequence.unsqueeze(0)  # Add batch dimension
        
        # Make predictions
        with torch.no_grad():
            predictions = model(input_sequence)

        # Convert predictions to numpy array
        predictions_np = predictions.squeeze(0).numpy()

        # Reshape predictions to (n_samples, n_features)
        predictions_np = predictions_np.reshape(-1, output_dim)

        # Inverse transform the predictions
        predictions_np = scaler.inverse_transform(predictions_np).flatten()

        return np.array(predictions_np)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None
