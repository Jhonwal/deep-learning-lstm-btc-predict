import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Dense, Dropout, BatchNormalization, 
    Bidirectional, MultiHeadAttention, LayerNormalization
)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import CyclicalLearningRate
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime, timedelta
import optuna
import json
from pathlib import Path

# Add to parameters
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

def save_artifacts(model, scalers, features, results, history, processed_df):
    """Save all required artifacts for deployment and analysis"""
    # Save model
    model.save(MODEL_DIR / 'bitcoin_transformer.keras')
    
    # Save scalers
    with open(MODEL_DIR / 'full_scaler.pkl', 'wb') as f:
        pickle.dump(scalers['feature'], f)
    with open(MODEL_DIR / 'close_scaler.pkl', 'wb') as f:
        pickle.dump(scalers['close'], f)
    
    # Save features
    with open(MODEL_DIR / 'feature_list.pkl', 'wb') as f:
        pickle.dump(features, f)
    
    # Save historical data for future predictions
    last_sequence = processed_df[features].values[-LOOK_BACK:]
    historical_data = {
        'scaled_data': last_sequence.tolist(),
        'original_data': processed_df[features].values[-LOOK_BACK:].tolist(),
        'last_timestamp': processed_df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(MODEL_DIR / 'historical_data.pkl', 'wb') as f:
        pickle.dump(historical_data, f)
    
    # Save training results
    training_results = {
        'metrics': results,
        'history': {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        },
        'model_config': {
            'architecture': 'Transformer-LSTM',
            'look_back': LOOK_BACK,
            'forecast_horizon': FORECAST_HORIZON,
            'features': features,
            'training_date': datetime.now().strftime('%Y-%m-%d')
        }
    }
    with open(MODEL_DIR / 'training_results.json', 'w') as f:
        json.dump(training_results, f, indent=4)

def make_future_prediction(model, scalers, features, df):
    """Make predictions for the next 24 hours (updated for Flask app)"""
    # Generate required features
    processed_df = df.copy()
    processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])
    processed_df.set_index('timestamp', inplace=True)
    
    # Generate temporal features
    processed_df['hour'] = processed_df.index.hour
    processed_df['day_of_week'] = processed_df.index.dayofweek
    processed_df['month'] = processed_df.index.month
    processed_df['is_weekend'] = (processed_df.index.dayofweek >= 5).astype(int)
    processed_df['is_halving'] = processed_df.index.isin(pd.to_datetime(BITCOIN_HALVING_DATES)).astype(int)
    
    # Generate technical features
    processed_df['price_change'] = processed_df['close'].pct_change()
    processed_df['volatility'] = (processed_df['high'] - processed_df['low']) / processed_df['close']
    processed_df.dropna(inplace=True)
    
    # Get and scale input data
    input_data = processed_df[features].values[-LOOK_BACK:]
    scaled_input = scalers['feature'].transform(input_data)
    
    # Make prediction
    prediction = model.predict(np.array([scaled_input]))
    predicted_prices = scalers['close'].inverse_transform(prediction.reshape(-1, 1)).flatten()
    
    # Generate future timestamps
    last_timestamp = processed_df.index[-1]
    return pd.DataFrame({
        'timestamp': [last_timestamp + timedelta(hours=i+1) for i in range(FORECAST_HORIZON)],
        'predicted_close': predicted_prices
    }).set_index('timestamp')
# Parameters
LOOK_BACK = 48  # 2 days of hourly data
FORECAST_HORIZON = 24  # Predict next 24 hours
BITCOIN_HALVING_DATES = ['2020-11-28', '2022-07-09', '2024-05-11']  # Example dates

def preprocess_data(df):
    """Preprocess data with time-aware splitting and feature engineering"""
    processed_df = df.copy()
    processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])
    processed_df.set_index('timestamp', inplace=True)
    
    # Generate advanced features
    processed_df['hour'] = processed_df.index.hour
    processed_df['day_of_week'] = processed_df.index.dayofweek
    processed_df['month'] = processed_df.index.month
    processed_df['is_weekend'] = (processed_df.index.dayofweek >= 5).astype(int)
    processed_df['is_halving'] = processed_df.index.isin(pd.to_datetime(BITCOIN_HALVING_DATES)).astype(int)
    
    # Technical features
    processed_df['price_change'] = processed_df['close'].pct_change()
    processed_df['volatility'] = (processed_df['high'] - processed_df['low']) / processed_df['close']
    
    # Remove highly correlated features (example)
    processed_df.drop(['ma_5', 'ma_10'], axis=1, inplace=True, errors='ignore')
    
    # Drop NA after feature generation
    processed_df.dropna(inplace=True)
    
    # Define final features
    features = ['open', 'high', 'low', 'close', 'volume', 'hour', 
               'day_of_week', 'month', 'is_weekend', 'is_halving',
               'price_change', 'volatility', 'ma_20', 'rsi']
    
    # Time-based split (no shuffle)
    train_size = int(len(processed_df) * 0.7)
    train_data = processed_df.iloc[:train_size]
    val_test_data = processed_df.iloc[train_size:]
    val_size = int(len(val_test_data) * 0.5)
    val_data = val_test_data.iloc[:val_size]
    test_data = val_test_data.iloc[val_size:]
    
    # Scale using training data only
    scaler = MinMaxScaler().fit(train_data[features])
    close_scaler = MinMaxScaler().fit(train_data[['close']])
    
    return {
        'train': scaler.transform(train_data[features]),
        'val': scaler.transform(val_data[features]),
        'test': scaler.transform(test_data[features]),
        'scaler': scaler,
        'close_scaler': close_scaler,
        'features': features
    }

def create_sequences(data, look_back, forecast_horizon):
    """Create time-series sequences with walk-forward validation"""
    X, y = [], []
    for i in range(look_back, len(data) - forecast_horizon):
        X.append(data[i-look_back:i])
        y.append(data[i:i+forecast_horizon, 3])  # Close price at index 3
    return np.array(X), np.array(y)

def build_transformer_model(input_shape, forecast_horizon):
    """Transformer-based architecture with temporal attention"""
    inputs = Input(shape=input_shape)
    
    # Bi-LSTM for initial feature extraction
    x = Bidirectional(LSTM(64, return_sequences=True, 
                          recurrent_dropout=0.2))(inputs)
    x = LayerNormalization()(x)
    
    # Multi-head attention
    attn_output = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = LayerNormalization()(x + attn_output)  # Residual connection
    
    # GRU for sequence processing
    x = GRU(32, recurrent_dropout=0.2)(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(forecast_horizon)(x)
    
    model = Model(inputs, outputs)
    
    # Cyclical learning rate
    clr = CyclicalLearningRate(
        base_lr=1e-5, max_lr=1e-3, step_size=2000
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(clr),
        loss=tf.keras.losses.Huber(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    
    return model

def calculate_trading_metrics(y_true, y_pred):
    """Calculate trading-specific performance metrics"""
    returns = y_pred[1:] / y_pred[:-1] - 1
    actual_returns = y_true[1:] / y_true[:-1] - 1
    
    # Sharpe Ratio
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(24*365)
    
    # Directional accuracy
    correct_directions = np.sign(returns) == np.sign(actual_returns)
    accuracy = np.mean(correct_directions)
    
    return sharpe, accuracy

def train_model(X_train, y_train, X_val, y_val):
    """Train model with enhanced callbacks"""
    model = build_transformer_model(ava
        (X_train.shape[1], X_train.shape[2]), FORECAST_HORIZON
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def optimize_hyperparameters(trial):
    """Hyperparameter tuning with Optuna"""
    units = trial.suggest_categorical('units', [32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    
    model = build_transformer_model((LOOK_BACK, len(features)), FORECAST_HORIZON)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.Huber()
    )
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        verbose=0
    )
    
    return model.evaluate(X_val, y_val, verbose=0)[0]

def main():
    # Load data
    df = pd.read_csv('./btc_hourly_data.csv')
    df.columns = [col.lower() for col in df.columns]
    
    # Preprocess
    processed = preprocess_data(df)
    
    # Create sequences
    X_train, y_train = create_sequences(processed['train'], LOOK_BACK, FORECAST_HORIZON)
    X_val, y_val = create_sequences(processed['val'], LOOK_BACK, FORECAST_HORIZON)
    X_test, y_test = create_sequences(processed['test'], LOOK_BACK, FORECAST_HORIZON)
    
    # Train
    model, history = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate
    test_metrics = {
        'rmse': mean_squared_error(test_actual_prices, test_pred_prices, squared=False),
        'mape': mean_absolute_percentage_error(test_actual_prices, test_pred_prices),
        'sharpe_ratio': sharpe,
        'directional_accuracy': dir_acc
    }
    
    # Save all artifacts
    save_artifacts(
        model=model,
        scalers={
            'feature': processed['scaler'],
            'close': processed['close_scaler']
        },
        features=processed['features'],
        results=test_metrics,
        history=history,
        processed_df=pd.concat([
            pd.DataFrame(processed['train'], columns=processed['features']),
            pd.DataFrame(processed['val'], columns=processed['features']),
            pd.DataFrame(processed['test'], columns=processed['features'])
        ])
    )
    
    # Print results
    print(f"\nSaved artifacts in {MODEL_DIR} directory:")
    print(list(MODEL_DIR.glob('*')))
    
    # Test prediction function
    future_df = make_future_prediction(
        model=model,
        scalers={
            'feature': processed['scaler'],
            'close': processed['close_scaler']
        },
        features=processed['features'],
        df=df.copy()
    )
    print("\nSample prediction:", future_df.head())

if __name__ == "__main__":
    main()