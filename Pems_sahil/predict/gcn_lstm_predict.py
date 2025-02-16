import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import pickle
# Load the saved model
model_path = './models/gcn_lstm/gcn_lstm.pth'
scaler_path = './models/gcn_lstm/scalers_gcn_lstm.pkl'

# Load the scaler
with open(scaler_path, 'rb') as f:
    scalers = pickle.load(f)

# Load the adjacency matrix (connectivity between nodes)
adj_matrix = pd.read_csv('./Data/adj.csv', header=None).values

# Normalize the adjacency matrix
def normalize_adjacency_matrix(adj):
    D = np.diag(np.sum(adj, axis=1))
    D_inv = np.linalg.inv(D)
    adj_normalized = np.dot(D_inv, adj)
    return adj_normalized

adj_normalized = normalize_adjacency_matrix(adj_matrix)
# Ensure adj_normalized is a PyTorch tensor with correct shape
adj_normalized = adj_normalized[:4, :4]  # Keep only the first 4x4 submatrix
adj_normalized = torch.tensor(adj_normalized, dtype=torch.float32)

# GCN Layer
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x shape: (batch_size, num_nodes, in_features)
        # adj shape: (num_nodes, num_nodes)

        # Multiply x with adjacency matrix on the node dimension
        out = torch.einsum('bni,nj->bni', x, adj)
        out = self.fc(out)
        return torch.relu(out)


# GCN Model
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)

    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        return x

# Combined GNN + LSTM Model
class GNN_LSTM_Model(nn.Module):
    def __init__(self, num_nodes, input_dim, gnn_hidden_dim, lstm_hidden_dim, output_dim, output_window):
        super(GNN_LSTM_Model, self).__init__()
        self.gcn = GCN(input_dim, gnn_hidden_dim, gnn_hidden_dim)
        self.lstm = nn.LSTM(gnn_hidden_dim * num_nodes, lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, num_nodes * output_dim * output_window)
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.output_window = output_window

    def forward(self, x, adj):
        # x shape: (batch_size, input_window, num_nodes, input_dim)
        batch_size, input_window, num_nodes, input_dim = x.shape
        
        gnn_out = []
        for t in range(input_window):
            gnn_output = self.gcn(x[:, t, :, :], adj)  # (batch_size, num_nodes, gnn_hidden_dim)
            gnn_out.append(gnn_output)
        
        # Stack GNN outputs
        gnn_out = torch.stack(gnn_out, dim=1)  # (batch_size, input_window, num_nodes, gnn_hidden_dim)
        
        # Reshape for LSTM input
        lstm_in = gnn_out.view(batch_size, input_window, -1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(lstm_in)  # (batch_size, input_window, lstm_hidden_dim)
        
        # Use only the last output from LSTM
        fc_in = lstm_out[:, -1, :]
        
        # Pass through fully connected layer
        fc_out = self.fc(fc_in)  # (batch_size, num_nodes * output_dim * output_window)
        
        # Reshape output to (batch_size, output_window, num_nodes, output_dim)
        output = fc_out.view(batch_size, self.output_window, self.num_nodes, self.output_dim)
        
        return output


# Model instantiation
num_nodes = 4
input_dim = 1
gnn_hidden_dim = 128
lstm_hidden_dim = 256
output_dim = 1
output_window = 5

model = GNN_LSTM_Model(num_nodes, input_dim, gnn_hidden_dim, lstm_hidden_dim, output_dim, output_window)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set model to evaluation mode

# Prediction function
def predict(scaled_input_sequence):
    try:
        
        model_reload = pd.read_csv('./predict/reload/gcn_lstm.csv').iloc[0, 0]
        if model_reload:
            model.load_state_dict(torch.load(model_path))
            model.eval()  # Set model to evaluation mode
            #write in csv false
            model_reload = pd.DataFrame(data=[["False"]])


        # Ensure input is a PyTorch tensor
        scaled_input_sequence = torch.tensor(scaled_input_sequence, dtype=torch.float32)
        
        # Reshape input for the model: (batch_size, input_window, num_nodes, input_dim)
        batch_size = 1  # Since we're predicting a single sequence
        input_window = scaled_input_sequence.shape[0]
        scaled_input_sequence = scaled_input_sequence.view(batch_size, input_window, num_nodes, input_dim)

        # Make predictions
        with torch.no_grad():
            predictions = model(scaled_input_sequence, adj_normalized)

        # Convert predictions to numpy array
        predictions_np = predictions.numpy()

        # Reshape predictions to (batch_size, output_window, num_nodes, output_dim)
        predictions_np = predictions_np.reshape(batch_size, output_window, num_nodes, output_dim)

        # Inverse transform the predictions for each node
        predictions_inverse = []
        for i in range(num_nodes):
            pred = predictions_np[0, :, i, 0]  # Get predictions for the i-th node
            inverse_pred = scalers[i].inverse_transform(pred.reshape(-1, 1))  # Inverse transform
            predictions_inverse.append(inverse_pred)

        # Extract the second row (for the second node)
        second_node_predictions = predictions_inverse[1].squeeze()  # Get the second node's predictions

        return second_node_predictions



    except Exception as e:
        print(adj_matrix.shape)  # Should output: torch.Size([4, 4])

        print(f"Prediction error: {e}")
        return None

# Example usage with a dummy scaled input sequence
# scaled_input_sequence shape: [input_window, num_nodes, input_dim]
input_window = 5  # Adjust based on your input sequence length
scaled_input_sequence = np.random.rand(input_window, num_nodes, input_dim)  # Random scaled input

# Predict future values
predicted_values = predict(scaled_input_sequence)

if predicted_values is not None:
    print("Predicted values (inverse transformed):", predicted_values)
