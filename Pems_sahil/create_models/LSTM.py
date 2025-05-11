# %%
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import os

# -------- Data Loading for a Single Node --------
def load_single_node_data(folder_path):
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
    combined_data = combined_data.dropna()  # Remove missing data
    return combined_data

# Load data for one node (you can change folder path accordingly)
folder_path = "C:/Users/sahil/Sumo/2024-10-20-18-27-19\Pems_sahil\Data/402835"
data = load_single_node_data(folder_path)

# Normalize the data
scaler = MinMaxScaler()
data['Flow (Veh/5 Minutes)'] = scaler.fit_transform(data[['Flow (Veh/5 Minutes)']])


# Function to save only the flow and timestamp from train data
def save_flow_and_timestamp(raw_data, folder_path='../models/lstm/'):
    # Select the first column (timestamp) and flow column
    df = raw_data[['Flow (Veh/5 Minutes)']].copy()
    df.index.name = 'Timestamp'
    
    # Inverse transform the flow column
    df['Flow (Veh/5 Minutes)'] = scaler.inverse_transform(df[['Flow (Veh/5 Minutes)']])
    
    # Create directory if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save DataFrame to CSV
    csv_file = os.path.join(folder_path, 'flow_and_timestamp.csv')
    df.to_csv(csv_file)
    print(f"Flow and timestamp saved to {csv_file}")

# Prepare the data for LSTM
def prepare_lstm_data(data, input_window=10, output_window=5):
    X, Y = [], []
    data_values = data.values
    for i in range(len(data_values) - input_window - output_window + 1):
        X.append(data_values[i:i+input_window])
        Y.append(data_values[i+input_window:i+input_window+output_window])
    
    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)
    return X, Y

input_window = 10
output_window = 5

# Split data into train, validation, and test sets
train_size = int(0.5 * len(data))
val_size = int(0.25 * len(data))
test_size = len(data) - train_size - val_size

train_data, val_data, test_data = np.split(data, [train_size, train_size + val_size])

# Save the raw training data with timestamp and flow (50% of the original data)
save_flow_and_timestamp(train_data)

X_train, Y_train = prepare_lstm_data(train_data, input_window, output_window)
X_val, Y_val = prepare_lstm_data(val_data, input_window, output_window)
X_test, Y_test = prepare_lstm_data(test_data, input_window, output_window)

# Create DataLoader for train, val, and test sets
def create_dataloader(X, Y, batch_size):
    dataset = TensorDataset(X, Y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

batch_size = 32
train_loader = create_dataloader(X_train, Y_train, batch_size)
val_loader = create_dataloader(X_val, Y_val, batch_size)
test_loader = create_dataloader(X_test, Y_test, batch_size)

# -------- LSTM Model --------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, output_window):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim * output_window)
        self.output_window = output_window
        self.output_dim = output_dim

    def forward(self, x):
        # x shape: (batch_size, input_window, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch_size, input_window, hidden_dim)
        lstm_out = lstm_out[:, -1, :]  # Take only the last output
        output = self.fc(lstm_out)  # (batch_size, output_dim * output_window)
        output = output.view(-1, self.output_window, self.output_dim)  # Reshape
        return output

# Model parameters
input_dim = 1  # Since we're using only the flow values
hidden_dim = 128
output_dim = 1  # We are predicting a single time series (flow)
output_window = 5

# Initialize the model
model = LSTMModel(input_dim, hidden_dim, output_dim, output_window)
optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = nn.MSELoss()

# -------- Train the Model --------
def train_model(model, train_loader, val_loader, num_epochs=10, patience=5):
    best_val_loss = float('inf')
    best_model = None
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, Y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for X_val, Y_val in val_loader:
                val_predictions = model(X_val)
                val_loss += criterion(val_predictions, Y_val).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            epochs_no_improve = 0
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve == patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break
    
    return best_model

# Train the model
best_model = train_model(model, train_loader, val_loader, num_epochs=10)

# Load the best model
model.load_state_dict(best_model)

# Save the model
torch.save(model.state_dict(), 'C:/Users/sahil/Sumo/2024-10-20-18-27-19/Pems_sahil/models/lstm/lstm_model.pth')
torch.save(scaler, 'C:/Users/sahil/Sumo/2024-10-20-18-27-19/Pems_sahil/models/lstm/lstm_scaler.pth')
print("Model and scaler saved successfully.")

# Store baseline predictions
def store_baseline_predictions(model, data_loader, scaler):
    model.eval()
    all_predictions = []
    all_actuals = []
    with torch.no_grad():
        for X_batch, Y_batch in data_loader:
            predictions = model(X_batch)
            all_predictions.append(predictions.cpu().numpy())
            all_actuals.append(Y_batch.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0).reshape(-1, 1)
    all_actuals = np.concatenate(all_actuals, axis=0).reshape(-1, 1)
    
    # Inverse transform
    all_predictions = scaler.inverse_transform(all_predictions)
    all_actuals = scaler.inverse_transform(all_actuals)
    
    # Create DataFrame
    df = pd.DataFrame({
        'actuals': all_actuals.flatten(),
        'predicted': all_predictions.flatten()
    })
    
    # Save to CSV
    df.to_csv('../models/lstm/baseline_predictions_lstm.csv', index=False)
    print("Baseline predictions saved to 'baseline_predictions_lstm.csv'")

# Store baseline predictions
store_baseline_predictions(model, train_loader, scaler)

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time
import torch
import seaborn as sns

def visualize_predictions(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    start = time.time()
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            preds = model(X_batch).cpu().numpy()
            predictions.append(preds)
            actuals.append(Y_batch.cpu().numpy())
    end = time.time()

    # Concatenate predictions and actuals
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    # Reshape predictions and actuals if necessary
    if predictions.ndim > 2:
        predictions = predictions.reshape(predictions.shape[0], -1)
    if actuals.ndim > 2:
        actuals = actuals.reshape(actuals.shape[0], -1)

    # Inverse transform the predictions and actuals
    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(actuals)

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)

    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Time taken: {end - start:.2f} seconds")

# Visualize the predictions
visualize_predictions(model, test_loader)


# %%

