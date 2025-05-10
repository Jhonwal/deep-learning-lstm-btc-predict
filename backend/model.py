import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime, timedelta

# Parameters
LOOK_BACK = 48  # 2 days of hourly data as input
FORECAST_HORIZON = 24  # Predict prices for the next 24 hours

def preprocess_data(df):
    """Preprocess the hourly Bitcoin price data"""
    # Create a copy to avoid modifying the original
    processed_df = df.copy()

    # Ensure proper datetime format for timestamp
    if 'timestamp' in processed_df.columns:
        processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])
        processed_df.set_index('timestamp', inplace=True)

    # Drop any missing values
    processed_df.dropna(inplace=True)

    # Define features to use in the model
    price_features = ['open', 'high', 'low', 'close', 'volume']
    time_features = ['hour', 'day_of_week']
    technical_features = ['price_change', 'volatility', 'ma_5', 'ma_10', 'ma_20', 'rsi']

    # Combine all features
    all_features = price_features + time_features + technical_features

    # Make sure all required features are in the dataframe
    missing_features = [f for f in all_features if f not in processed_df.columns]
    if missing_features:
        # Generate missing time features if needed
        if 'hour' in missing_features and 'timestamp' in processed_df.index.names:
            processed_df['hour'] = processed_df.index.hour
            missing_features.remove('hour')

        if 'day_of_week' in missing_features and 'timestamp' in processed_df.index.names:
            processed_df['day_of_week'] = processed_df.index.dayofweek
            missing_features.remove('day_of_week')

        # Calculate missing technical indicators if needed
        if 'price_change' in missing_features:
            processed_df['price_change'] = processed_df['close'].pct_change()
            missing_features.remove('price_change')

        if 'volatility' in missing_features:
            # Calculate hourly volatility as (high-low)/close
            processed_df['volatility'] = (processed_df['high'] - processed_df['low']) / processed_df['close']
            missing_features.remove('volatility')

        if 'ma_5' in missing_features:
            processed_df['ma_5'] = processed_df['close'].rolling(window=5).mean()
            missing_features.remove('ma_5')

        if 'ma_10' in missing_features:
            processed_df['ma_10'] = processed_df['close'].rolling(window=10).mean()
            missing_features.remove('ma_10')

        if 'ma_20' in missing_features:
            processed_df['ma_20'] = processed_df['close'].rolling(window=20).mean()
            missing_features.remove('ma_20')

        if 'rsi' in missing_features:
            # Calculate RSI
            delta = processed_df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            processed_df['rsi'] = 100 - (100 / (1 + rs))
            missing_features.remove('rsi')

    # Drop any rows with NaN due to rolling calculations
    processed_df.dropna(inplace=True)

    # Scale features to [0,1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(processed_df[all_features])

    # Create a separate scaler for close price only (will be needed for inverse transform)
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler.fit_transform(processed_df[['close']])

    return scaled_data, scaler, close_scaler, processed_df, all_features

def create_sequences_multistep(data, look_back=LOOK_BACK, forecast_horizon=FORECAST_HORIZON):
    """Create sequences for multi-step prediction"""
    X, y = [], []
    for i in range(look_back, len(data) - forecast_horizon + 1):
        # Input sequence
        X.append(data[i-look_back:i])

        # Output sequence - multiple future close prices
        future_closes = [data[i+j, 3] for j in range(forecast_horizon)]  # Close price is at index 3
        y.append(future_closes)

    return np.array(X), np.array(y)

def build_hybrid_model(input_shape, output_steps):
    """Build a hybrid LSTM-GRU model optimized for hourly data and multistep prediction"""
    model = Sequential()

    # First Bidirectional LSTM layer with explicit input_shape
    model.add(Bidirectional(
        LSTM(
            units=64,
            return_sequences=True,
            kernel_regularizer=l1_l2(l1=0.00001, l2=0.00001),
            recurrent_regularizer=l1_l2(l2=0.00001),
            recurrent_dropout=0.1
        ),
        input_shape=input_shape
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Second LSTM layer
    model.add(LSTM(
        units=32,
        return_sequences=True,
        kernel_regularizer=l1_l2(l2=0.00001),
        recurrent_regularizer=l1_l2(l2=0.00001),
        recurrent_dropout=0.1
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # GRU layer
    model.add(GRU(
        units=32,
        return_sequences=False,
        kernel_regularizer=l1_l2(l2=0.00001),
        recurrent_regularizer=l1_l2(l2=0.00001)
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Output layer with multiple units for multi-step prediction
    model.add(Dense(
        units=output_steps,
        kernel_regularizer=l1_l2(l2=0.00001)
    ))

    # Using Huber loss for robustness against outliers
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber()
    )

    return model

def train_model(df):
    """Train and evaluate the LSTM-GRU model for hourly data"""
    # Preprocess data
    scaled_data, scaler, close_scaler, processed_df, all_features = preprocess_data(df)

    # Create sequences for multi-step prediction
    X, y = create_sequences_multistep(scaled_data, LOOK_BACK, FORECAST_HORIZON)

    # Split data into training, validation, and test sets (chronological order)
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # Build hybrid model for multi-step prediction
    model = build_hybrid_model((X_train.shape[1], X_train.shape[2]), FORECAST_HORIZON)

    # Build the model explicitly to see parameters
    dummy_input = np.zeros((1, X_train.shape[1], X_train.shape[2]))
    _ = model.predict(dummy_input)
    model.summary()

    # Define callbacks for better training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=0.00001,
            verbose=1
        )
    ]

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate model - FIXED: passing history to evaluate_model
    results = evaluate_model(model, close_scaler, X_train, y_train, X_val, y_val, X_test, y_test,
                            processed_df, all_features, history)  # Added history parameter

    # Save model and scalers
    save_results(model, scaler, close_scaler, results, history, processed_df, all_features)

    return model, scaler, close_scaler, results, history

def evaluate_model(model, close_scaler, X_train, y_train, X_val, y_val, X_test, y_test, processed_df, all_features, history):  # Added history parameter
    """Evaluate model performance with detailed metrics for multi-step prediction"""

    # Make predictions
    train_predict = model.predict(X_train)
    val_predict = model.predict(X_val)
    test_predict = model.predict(X_test)

    # We'll evaluate metrics for different forecast horizons
    horizons = [1, 6, 12, 24]  # 1 hour, 6 hours, 12 hours, 24 hours
    metrics = {}

    for horizon in horizons:
        if horizon > FORECAST_HORIZON:
            continue

        # Get predictions for the specific horizon
        train_h = train_predict[:, horizon-1].reshape(-1, 1)
        val_h = val_predict[:, horizon-1].reshape(-1, 1)
        test_h = test_predict[:, horizon-1].reshape(-1, 1)

        # Get actual values for the specific horizon
        y_train_h = y_train[:, horizon-1].reshape(-1, 1)
        y_val_h = y_val[:, horizon-1].reshape(-1, 1)
        y_test_h = y_test[:, horizon-1].reshape(-1, 1)

        # Inverse transform to get actual prices
        train_predict_prices = close_scaler.inverse_transform(train_h)
        val_predict_prices = close_scaler.inverse_transform(val_h)
        test_predict_prices = close_scaler.inverse_transform(test_h)

        y_train_prices = close_scaler.inverse_transform(y_train_h)
        y_val_prices = close_scaler.inverse_transform(y_val_h)
        y_test_prices = close_scaler.inverse_transform(y_test_h)

        # Calculate metrics
        train_metrics = calculate_metrics(y_train_prices, train_predict_prices)
        val_metrics = calculate_metrics(y_val_prices, val_predict_prices)
        test_metrics = calculate_metrics(y_test_prices, test_predict_prices)

        metrics[f'{horizon}_hour'] = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }

    # Check for overfitting
    overfitting_ratio_train_test = metrics['1_hour']['train']['rmse'] / metrics['1_hour']['test']['rmse']
    overfitting_ratio_train_val = metrics['1_hour']['train']['rmse'] / metrics['1_hour']['val']['rmse']

    results = {
        'metrics': metrics,
        'overfitting_ratios': {
            'train_to_val': float(overfitting_ratio_train_val),
            'train_to_test': float(overfitting_ratio_train_test),
        },
        'training_date': datetime.now().strftime('%Y-%m-%d')
    }

    # Plot results
    plt.figure(figsize=(15, 10))

    # Plot 1-hour prediction
    plt.subplot(2, 2, 1)
    plt.plot(y_test_prices[:100], label='Actual')
    plt.plot(test_predict_prices[:100], label='Predicted (1h)')
    plt.title('Bitcoin Price 1-Hour Prediction - Test Set')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()

    # Plot 24-hour prediction if available
    if '24_hour' in metrics:
        plt.subplot(2, 2, 2)
        test_24h = test_predict[:, 23].reshape(-1, 1)
        test_predict_24h = close_scaler.inverse_transform(test_24h)
        y_test_24h = y_test[:, 23].reshape(-1, 1)
        y_test_prices_24h = close_scaler.inverse_transform(y_test_24h)

        plt.plot(y_test_prices_24h[:100], label='Actual')
        plt.plot(test_predict_24h[:100], label='Predicted (24h)')
        plt.title('Bitcoin Price 24-Hour Prediction - Test Set')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend()

    # Plot training loss - Now history is properly defined
    plt.subplot(2, 2, 3)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot an example of full 24-hour prediction
    plt.subplot(2, 2, 4)
    example_idx = np.random.randint(0, len(test_predict))
    example_pred = test_predict[example_idx]
    example_actual = y_test[example_idx]

    example_pred_prices = close_scaler.inverse_transform(example_pred.reshape(-1, 1)).flatten()
    example_actual_prices = close_scaler.inverse_transform(example_actual.reshape(-1, 1)).flatten()

    plt.plot(range(1, FORECAST_HORIZON+1), example_actual_prices, label='Actual')
    plt.plot(range(1, FORECAST_HORIZON+1), example_pred_prices, label='Predicted')
    plt.title('Example 24-Hour Forecast')
    plt.xlabel('Hours Ahead')
    plt.ylabel('Price')
    plt.legend()

    plt.tight_layout()
    plt.savefig('hourly_model_performance.png')

    return results

def calculate_metrics(actual, predicted):
    """Calculate comprehensive evaluation metrics"""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    # Directional Accuracy
    if len(actual) > 1:
        actual_changes = np.sign(np.diff(actual, axis=0))
        predicted_changes = np.sign(np.diff(predicted, axis=0))
        directional_accuracy = np.mean(actual_changes == predicted_changes)
    else:
        directional_accuracy = 0

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(r2),
        'directional_accuracy': float(directional_accuracy)
    }

def save_results(model, scaler, close_scaler, results, history, processed_df, all_features):
    """Save model, scalers, and training results"""
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save model
    model.save('models/bitcoin_hourly_model.h5')

    # Save scalers
    with open('models/full_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open('models/close_scaler.pkl', 'wb') as f:
        pickle.dump(close_scaler, f)

    # Save feature list
    with open('models/feature_list.pkl', 'wb') as f:
        pickle.dump(all_features, f)

    # Save historical data
    last_sequence = processed_df[all_features].values[-LOOK_BACK:]

    historical_data = {
        'scaled_data': last_sequence.tolist(),
        'original_data': processed_df[all_features].values[-LOOK_BACK:].tolist(),
        'last_timestamp': processed_df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open('models/historical_data.pkl', 'wb') as f:
        pickle.dump(historical_data, f)

    # Prepare training history
    train_history = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }

    # Combine all results
    training_results = {
        'message': 'Model trained successfully for hourly Bitcoin price prediction',
        'metrics': results['metrics'],
        'overfitting_ratios': results['overfitting_ratios'],
        'history': train_history,
        'model_config': {
            'architecture': 'Bidirectional LSTM + LSTM + GRU with regularization',
            'epochs': len(history.history['loss']),
            'early_stopping': True,
            'batch_size': 32,
            'look_back_window': LOOK_BACK,
            'forecast_horizon': FORECAST_HORIZON,
            'features': all_features
        }
    }

    # Save training results
    with open('models/hourly_training_results.json', 'w') as f:
        import json
        json.dump(training_results, f, indent=4)

    print("Model, scalers, and results saved successfully")

def make_future_prediction(df, model, scaler, close_scaler, all_features):
    """Make predictions for the next 24 hours"""
    latest_data = df.copy()

    if 'timestamp' in latest_data.columns:
        latest_data['timestamp'] = pd.to_datetime(latest_data['timestamp'])
        latest_data.set_index('timestamp', inplace=True)

    # Generate missing features
    for feature in all_features:
        if feature not in latest_data.columns:
            if feature == 'hour':
                latest_data['hour'] = latest_data.index.hour
            elif feature == 'day_of_week':
                latest_data['day_of_week'] = latest_data.index.dayofweek
            elif feature == 'price_change':
                latest_data['price_change'] = latest_data['close'].pct_change()
            elif feature == 'volatility':
                latest_data['volatility'] = (latest_data['high'] - latest_data['low']) / latest_data['close']
            elif feature == 'ma_5':
                latest_data['ma_5'] = latest_data['close'].rolling(window=5).mean()
            elif feature == 'ma_10':
                latest_data['ma_10'] = latest_data['close'].rolling(window=10).mean()
            elif feature == 'ma_20':
                latest_data['ma_20'] = latest_data['close'].rolling(window=20).mean()
            elif feature == 'rsi':
                delta = latest_data['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                latest_data['rsi'] = 100 - (100 / (1 + rs))

    latest_data.dropna(inplace=True)

    # Get the last LOOK_BACK timestamps of data
    input_data = latest_data[all_features].values[-LOOK_BACK:]

    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Reshape for model input
    model_input = np.array([scaled_input])

    # Make prediction
    prediction = model.predict(model_input)

    # Convert the prediction to actual prices
    predicted_prices = close_scaler.inverse_transform(prediction[0].reshape(-1, 1)).flatten()

    # Create a DataFrame with the predictions
    last_timestamp = latest_data.index[-1]
    future_timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(FORECAST_HORIZON)]

    forecast_df = pd.DataFrame({
        'timestamp': future_timestamps,
        'predicted_close': predicted_prices
    })

    forecast_df.set_index('timestamp', inplace=True)

    return forecast_df

def main():
    # Load data
    file_path = './btc_hourly_data.csv'
    df = pd.read_csv(file_path)

    # Check required columns
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(col.lower() in [c.lower() for c in df.columns] for col in required_columns):
        missing = [col for col in required_columns if col.lower() not in [c.lower() for c in df.columns]]
        print(f"Missing required columns: {missing}")
        return

    # Standardize column names
    df.columns = [col.lower() for col in df.columns]

    # Train model
    model, scaler, close_scaler, results, history = train_model(df)

    print("\nTraining complete!")
    print(f"Test RMSE (1 hour ahead): {results['metrics']['1_hour']['test']['rmse']}")
    print(f"Test MAPE (1 hour ahead): {results['metrics']['1_hour']['test']['mape']}")
    print(f"Directional accuracy (1 hour ahead): {results['metrics']['1_hour']['test']['directional_accuracy']}")

    if '24_hour' in results['metrics']:
        print(f"\nTest RMSE (24 hours ahead): {results['metrics']['24_hour']['test']['rmse']}")
        print(f"Test MAPE (24 hours ahead): {results['metrics']['24_hour']['test']['mape']}")
        print(f"Directional accuracy (24 hours ahead): {results['metrics']['24_hour']['test']['directional_accuracy']}")

    print(f"\nOverfitting ratio (train to test): {results['overfitting_ratios']['train_to_test']}")

    # Make prediction for next 24 hours
    all_features = []
    with open('models/feature_list.pkl', 'rb') as f:
        all_features = pickle.load(f)

    future_prediction = make_future_prediction(df, model, scaler, close_scaler, all_features)
    print("\nPredictions for the next 24 hours:")
    print(future_prediction)

if __name__ == "__main__":
    main()