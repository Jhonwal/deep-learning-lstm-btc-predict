
# Sample code for making predictions with the trained model
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load the model and scalers
model = load_model('final_model.h5')
feature_scaler = joblib.load('feature_scaler.gz')
target_scaler = joblib.load('target_scaler.gz')

# Prepare your input data (should have the same features as during training)
# Get the last 60 minutes of data
latest_data = get_latest_60_minutes_data()  # Replace with your data collection function

# Scale the input
scaled_input = feature_scaler.transform(latest_data)

# Reshape for LSTM input [samples, time steps, features]
model_input = scaled_input.reshape(1, 60, scaled_input.shape[1])

# Make prediction
scaled_prediction = model.predict(model_input)

# Inverse transform to get actual price
predicted_price = target_scaler.inverse_transform(scaled_prediction)[0][0]

print(f"Predicted BTC price in 1 hour: ${predicted_price:.2f}")
