from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
import json
import os
import pickle
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# Global variables
model = None
scaler = None
LOOK_BACK = 60  # Number of previous time steps to use as input features
MODEL_PATH = 'models/bitcoin_lstm_model.h5'
SCALER_PATH = 'models/scaler.pkl'
RESULTS_PATH = 'models/training_results.json'  # Path for training results

# Load model and scaler on startup if they exist
def load_saved_model():
    global model, scaler
    
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            model = load_model(MODEL_PATH)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            print("Model and scaler loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model or scaler: {str(e)}")
    return False

# Initialize by loading the model if it exists
load_saved_model()

def preprocess_data(df):
    """Preprocess the Bitcoin price data"""
    global scaler
    
    # Create a new scaler if none exists
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        
    # Extract relevant features
    data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    
    # Scale the data
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data

def create_sequences(data, look_back=LOOK_BACK):
    """Create sequences for LSTM model"""
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])
        y.append(data[i, 3])  # Close price
    return np.array(X), np.array(y)

def build_model(input_shape):
    """Build and compile LSTM model"""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the LSTM model with uploaded dataset and provide comprehensive evaluation"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    try:
        # Read and preprocess data
        df = pd.read_csv(file)
        df['Open time'] = pd.to_datetime(df['Open time'])
        df = df.sort_values('Open time')
        
        # Preprocess data
        scaled_data = preprocess_data(df)
        
        # Create sequences
        X, y = create_sequences(scaled_data)
        
        # Split data
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Build and train model
        global model
        model = build_model((X_train.shape[1], X_train.shape[2]))
        
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Save the model and scaler
        if not os.path.exists('models'):
            os.makedirs('models')
        
        model.save(MODEL_PATH)
        
        # Save the scaler as well
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Generate detailed training metrics
        train_metrics = {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
        
        # Make predictions for visualization and detailed evaluation
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        
        # Inverse transform predictions
        temp_train = np.zeros((len(train_predict), 5))
        temp_train[:, 3] = train_predict.flatten()
        temp_test = np.zeros((len(test_predict), 5))
        temp_test[:, 3] = test_predict.flatten()
        
        train_predict = scaler.inverse_transform(temp_train)[:, 3]
        test_predict = scaler.inverse_transform(temp_test)[:, 3]
        
        # Get original values
        temp_y_train = np.zeros((len(y_train), 5))
        temp_y_train[:, 3] = y_train
        temp_y_test = np.zeros((len(y_test), 5))
        temp_y_test[:, 3] = y_test
        
        y_train_orig = scaler.inverse_transform(temp_y_train)[:, 3]
        y_test_orig = scaler.inverse_transform(temp_y_test)[:, 3]
        
        # Comprehensive Evaluation Metrics
        def calculate_metrics(actual, predicted):
            from sklearn.metrics import (
                mean_squared_error, 
                mean_absolute_error, 
                mean_absolute_percentage_error, 
                r2_score
            )
            
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual, predicted)
            mape = mean_absolute_percentage_error(actual, predicted)
            r2 = r2_score(actual, predicted)
            
            # Directional Accuracy (predicting price movement)
            actual_changes = np.sign(np.diff(actual))
            predicted_changes = np.sign(np.diff(predicted))
            directional_accuracy = np.mean(actual_changes == predicted_changes)
            
            return {
                'MSE': float(mse),
                'RMSE': float(rmse),
                'MAE': float(mae),
                'MAPE': float(mape),
                'R2_Score': float(r2),
                'Directional_Accuracy': float(directional_accuracy)
            }
        
        # Calculate metrics for training and test sets
        train_evaluation = calculate_metrics(y_train_orig, train_predict)
        test_evaluation = calculate_metrics(y_test_orig, test_predict)
        
        # Prepare dates for visualization
        dates = df['Open time'].iloc[LOOK_BACK:].reset_index(drop=True)
        train_dates = dates[:split].dt.strftime('%Y-%m-%d').tolist()
        test_dates = dates[split:].dt.strftime('%Y-%m-%d').tolist()
        
        # Prepare detailed results
        training_results = {
            'message': 'Model trained successfully',
            'metrics': {
                'training_metrics': train_evaluation,
                'test_metrics': test_evaluation
            },
            'history': train_metrics,
            'visualization': {
                'train': {
                    'dates': train_dates,
                    'actual': y_train_orig.tolist(),
                    'predicted': train_predict.tolist()
                },
                'test': {
                    'dates': test_dates,
                    'actual': y_test_orig.tolist(),
                    'predicted': test_predict.tolist()
                }
            },
            'model_config': {
                'epochs': 20,
                'batch_size': 32,
                'look_back_window': LOOK_BACK,
                'features': ['Open', 'High', 'Low', 'Close', 'Volume']
            }
        }
        
        # Save training results to file
        with open(RESULTS_PATH, 'w') as f:
            json.dump(training_results, f)
        
        return jsonify(training_results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training-results', methods=['GET'])
def get_training_results():
    """Get the saved training results"""
    if not os.path.exists(RESULTS_PATH):
        return jsonify({'error': 'No training results available'}), 404
    
    try:
        with open(RESULTS_PATH, 'r') as f:
            training_results = json.load(f)
        return jsonify(training_results)
    except Exception as e:
        return jsonify({'error': f'Error loading training results: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions using the trained model"""
    global model, scaler
    
    # Check if model is trained
    if model is None or scaler is None:
        if not load_saved_model():
            return jsonify({'error': 'Model not trained yet. Please train the model first.'}), 400
    
    try:
        # Get input data
        data = request.json.get('data')
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Invalid input data format. Expected a list of price data.'}), 400
        
        # Convert to numpy array
        input_data = np.array(data)
        
        # Scale the data
        scaled_data = scaler.transform(input_data)
        
        # Create sequence
        if len(scaled_data) < LOOK_BACK:
            return jsonify({'error': f'Not enough data points. Need at least {LOOK_BACK} data points.'}), 400
        
        # Take the last LOOK_BACK points
        input_sequence = scaled_data[-LOOK_BACK:].reshape(1, LOOK_BACK, 5)
        
        # Make prediction
        predicted_scaled = model.predict(input_sequence)
        
        # Inverse transform the prediction
        temp = np.zeros((1, 5))
        temp[0, 3] = predicted_scaled[0, 0]
        predicted_price = scaler.inverse_transform(temp)[0, 3]
        
        return jsonify({
            'predicted_price': float(predicted_price)
        })
        
    except Exception as e:
        return jsonify({'error': f'Error making prediction: {str(e)}'}), 500

@app.route('/api/upload_prediction_data', methods=['POST'])
def upload_prediction_data():
    """Process CSV file for prediction data"""
    if 'csv_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['csv_file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and file.filename.endswith('.csv'):
        try:
            # Read and parse CSV data
            csv_content = file.read().decode('utf-8')
            csv_data = []
            dates = []
            
            for i, line in enumerate(csv_content.splitlines()):
                if i == 0:  # Skip header
                    continue
                    
                parts = line.split(',')
                if len(parts) < 6:
                    return jsonify({'error': f'Invalid CSV format at line {i+1}. Expected at least 6 columns'}), 400
                
                # Extract date and price data
                try:
                    date = parts[0].strip()
                    dates.append(date)
                    
                    open_price = float(parts[1].strip())
                    high_price = float(parts[2].strip())
                    low_price = float(parts[3].strip())
                    close_price = float(parts[4].strip())
                    volume = float(parts[5].strip())
                    
                    csv_data.append([open_price, high_price, low_price, close_price, volume])
                except (ValueError, IndexError) as e:
                    return jsonify({'error': f'Error parsing data at line {i+1}: {str(e)}'}), 400
            
            # Ensure minimum data points
            if len(csv_data) < LOOK_BACK:
                return jsonify({'error': f'CSV must contain at least {LOOK_BACK} data points (excluding header)'}), 400
            
            # Format date (assuming dates are in chronological order)
            last_date = None
            if dates:
                try:
                    # Try to parse the date in various formats
                    for date_format in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']:
                        try:
                            last_date = datetime.strptime(dates[-1], date_format).strftime('%Y-%m-%d')
                            break
                        except ValueError:
                            continue
                    
                    if not last_date:
                        raise ValueError("Could not parse date")
                except Exception as e:
                    return jsonify({'error': f'Error parsing date: {str(e)}'}), 400
            
            return jsonify({
                'data': csv_data,
                'last_date': last_date
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing CSV: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file format. Please upload a CSV file'}), 400

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the trained model"""
    global model, scaler
    
    # Try to load the model if it's not loaded yet
    if model is None:
        if not load_saved_model():
            return jsonify({'trained': False})
    
    return jsonify({
        'trained': True,
        'architecture': [
            {'type': 'LSTM', 'units': 50, 'return_sequences': True},
            {'type': 'Dropout', 'rate': 0.2},
            {'type': 'LSTM', 'units': 50, 'return_sequences': True},
            {'type': 'Dropout', 'rate': 0.2},
            {'type': 'LSTM', 'units': 50},
            {'type': 'Dropout', 'rate': 0.2},
            {'type': 'Dense', 'units': 1}
        ],
        'look_back': LOOK_BACK,
        'features': ['Open', 'High', 'Low', 'Close', 'Volume'],
        'model_path': MODEL_PATH,
        'scaler_path': SCALER_PATH
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)