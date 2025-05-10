import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Constants
LOOK_BACK = 48
FORECAST_HORIZON = 24

# Load model and related data
def load_model_artifacts():
    model = load_model('models/bitcoin_hourly_model.h5')
    
    with open('models/full_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('models/close_scaler.pkl', 'rb') as f:
        close_scaler = pickle.load(f)
    
    with open('models/feature_list.pkl', 'rb') as f:
        features = pickle.load(f)
    
    with open('models/historical_data.pkl', 'rb') as f:
        historical_data = pickle.load(f)
    
    with open('models/hourly_training_results.json', 'r') as f:
        training_results = json.load(f)
    
    return model, scaler, close_scaler, features, historical_data, training_results

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Provide information about the model architecture and performance"""
    try:
        _, _, _, _, historical_data, training_results = load_model_artifacts()
        
        # Extract key metrics for frontend display
        metrics = {
            'short_term': {
                'rmse': training_results['metrics']['1_hour']['test']['rmse'],
                'mape': training_results['metrics']['1_hour']['test']['mape'],
                'directional_accuracy': training_results['metrics']['1_hour']['test']['directional_accuracy']
            },
            'long_term': {
                'rmse': training_results['metrics']['24_hour']['test']['rmse'] if '24_hour' in training_results['metrics'] else None,
                'mape': training_results['metrics']['24_hour']['test']['mape'] if '24_hour' in training_results['metrics'] else None,
                'directional_accuracy': training_results['metrics']['24_hour']['test']['directional_accuracy'] if '24_hour' in training_results['metrics'] else None
            }
        }
        
        # Create simplified model info for frontend
        model_info = {
            'name': 'Bitcoin Price Predictor',
            'architecture': training_results['model_config']['architecture'],
            'forecast_horizon': training_results['model_config']['forecast_horizon'],
            'look_back': training_results['model_config']['look_back_window'],
            'features_used': training_results['model_config']['features'],
            'training_date': training_results.get('training_date', historical_data['last_updated']),
            'last_data_timestamp': historical_data['last_timestamp'],
            'metrics': metrics,
            'train_history': {
                'loss': training_results['history']['loss'][-20:],  # Last 20 epochs for visualization
                'val_loss': training_results['history']['val_loss'][-20:]
            }
        }
        
        return jsonify(model_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Generate predictions for the next 24 hours"""
    try:
        model, scaler, close_scaler, features, historical_data, _ = load_model_artifacts()
        
        # Use the historical data to make predictions
        scaled_input = np.array(historical_data['scaled_data'])
        
        # Reshape for model input
        model_input = np.array([scaled_input])
        
        # Make prediction
        prediction = model.predict(model_input)
        
        # Convert the prediction to actual prices
        predicted_prices = close_scaler.inverse_transform(prediction[0].reshape(-1, 1)).flatten()
        
        # Parse the last timestamp from historical data
        last_timestamp = datetime.strptime(historical_data['last_timestamp'], '%Y-%m-%d %H:%M:%S')
        
        # Create timestamps for predictions
        future_timestamps = [(last_timestamp + timedelta(hours=i+1)).strftime('%Y-%m-%d %H:%M:%S') 
                             for i in range(FORECAST_HORIZON)]
        
        # Calculate confidence intervals (simple approach)
        lower_bound = predicted_prices * 0.98  # 2% lower bound
        upper_bound = predicted_prices * 1.02  # 2% upper bound
        
        # Prepare prediction data for frontend
        predictions = [{
            'timestamp': ts,
            'predicted_price': float(price),
            'lower_bound': float(lb),
            'upper_bound': float(ub),
            'hour': i+1
        } for i, (ts, price, lb, ub) in enumerate(zip(future_timestamps, predicted_prices, lower_bound, upper_bound))]
        
        return jsonify({
            'last_timestamp': historical_data['last_timestamp'],
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluation', methods=['GET'])
def get_evaluation():
    """Provide detailed evaluation metrics"""
    try:
        _, _, _, _, _, training_results = load_model_artifacts()
        
        # Extract evaluation metrics for different time horizons
        horizons = {}
        for horizon_key, horizon_data in training_results['metrics'].items():
            hours = horizon_key.split('_')[0]
            horizons[hours] = {
                'train': horizon_data['train'],
                'validation': horizon_data['val'],
                'test': horizon_data['test']
            }
        
        evaluation = {
            'horizons': horizons,
            'overfitting_ratios': training_results['overfitting_ratios']
        }
        
        return jsonify(evaluation)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)