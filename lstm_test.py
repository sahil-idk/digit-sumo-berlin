import numpy as np
import os
import sys

# Add the predict directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
predict_dir = os.path.join(current_dir, "Pems_sahil", "predict")
sys.path.append(predict_dir)

from lstm_predict import predict

def test_lstm_prediction():
    try:
        # Create a dummy input sequence
        # Shape should be (1, 10, 1) - (batch_size, sequence_length, features)
        input_sequence = np.random.rand(1, 10, 1) * 50  # Random values between 0 and 50

        print("Input sequence shape:", input_sequence.shape)
        print("\nInput sequence values:")
        print(input_sequence.reshape(10, 1))  # Reshape for better visualization

        # Make prediction
        predictions = predict(input_sequence)

        if predictions is not None:
            print("\nPredictions (next 5 time steps):")
            print(predictions)
        else:
            print("\nPrediction failed. Check error message above.")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Predict directory path: {predict_dir}")

if __name__ == "__main__":
    test_lstm_prediction()