import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, GRU, Bidirectional
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import os
import datetime
import json

# Keep your existing code for mounting drive and paths

# Feature engineering improvements
def add_technical_indicators(df):
    """Add more technical indicators to improve model performance"""
    # Add more moving averages with different windows
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    
    # Add exponential moving averages
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Add price momentum indicators
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    
    # Add volatility indicators
    df['volatility_5'] = df['close'].rolling(window=5).std()
    df['volatility_10'] = df['close'].rolling(window=10).std()
    
    # Add price changes at different intervals
    df['price_change_1'] = df['close'].pct_change(1)
    df['price_change_5'] = df['close'].pct_change(5)
    
    # Add MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Fill NaN values
    df.fillna(method='bfill', inplace=True)
    return df

# Improved data preprocessing
def preprocess_data(df):
    """Preprocess data with better normalization techniques"""
    # Convert timestamp to datetime features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    
    # Create sine and cosine features for cyclical time data
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute']/60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute']/60)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    return df

# Improved model architecture with less complexity
def build_improved_model(input_shape):
    """Build a less complex but more robust model to prevent overfitting"""
    model = Sequential()
    
    # Use Bidirectional LSTM for better pattern recognition
    model.add(Bidirectional(
        LSTM(
            units=40,
            return_sequences=True,
            input_shape=input_shape,
            kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
            recurrent_regularizer=l1_l2(l1=0.0001, l2=0.0001)
        )
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))  # Increased dropout
    
    # Single GRU layer
    model.add(GRU(
        units=30,
        kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
        recurrent_regularizer=l1_l2(l1=0.0001, l2=0.0001)
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))  # Increased dropout
    
    # Output layer
    model.add(Dense(
        units=1,
        kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
        activation='linear'
    ))
    
    # Use a learning rate scheduler
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='huber_loss'  # Huber loss is less sensitive to outliers
    )
    return model

# Improved data preparation
def prepare_training_data(df, sequence_length=60, prediction_horizon=60):
    """Prepare training data with improved scaling and sequence creation"""
    # Select features
    features = ['open', 'high', 'low', 'close', 'volume',
                'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'day_sin', 'day_cos',
                'ma_5', 'ma_10', 'ma_20', 'ma_50', 'ema_5', 'ema_10', 'ema_20',
                'momentum_5', 'momentum_10', 'volatility_5', 'volatility_10',
                'price_change_1', 'price_change_5', 'macd', 'macd_signal', 'macd_hist']
    
    # Use all available features
    X = df[features].values
    
    # Target: next hour's closing price
    df['target'] = df['close'].shift(-prediction_horizon)
    df.dropna(inplace=True)
    y = df['target'].values
    
    # Use RobustScaler for better handling of outliers
    feature_scaler = RobustScaler()
    X_scaled = feature_scaler.fit_transform(X[:len(df)])
    
    target_scaler = RobustScaler()
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1))
    
    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)
    
    # Split with more training data
    X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.25, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, shuffle=False)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_scaler, target_scaler, features

# Enhanced training function with learning rate scheduling
def train_model(X_train, y_train, X_val, y_val, input_shape, model_save_path):
    """Train model with improved techniques to prevent overfitting"""
    model = build_improved_model(input_shape)
    
    # Enhanced callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=f"{model_save_path}/best_model.h5",
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=8,
        min_lr=0.00001,
        verbose=1
    )
    
    # Use more epochs but with early stopping
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=64,  # Larger batch size for better generalization
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1,
        shuffle=True  # Enable shuffling for better generalization
    )
    
    return model, history

# Main training workflow
def main():
    # Read the data
    df = pd.read_csv('/content/drive/MyDrive/btc_minute_data3.csv')
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Prepare data for training
    X_train, X_val, X_test, y_train, y_val, y_test, feature_scaler, target_scaler, features = prepare_training_data(df)
    
    # Train model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model, history = train_model(X_train, y_train, X_val, y_val, input_shape, model_save_path)
    
    # Save all necessary artifacts
    model.save(f"{model_save_path}/final_model.h5")
    joblib.dump(feature_scaler, f"{model_save_path}/feature_scaler.gz")
    joblib.dump(target_scaler, f"{model_save_path}/target_scaler.gz")
    
    # Save feature list
    with open(f"{model_save_path}/feature_list.json", 'w') as f:
        json.dump(features, f)
    
    # Evaluate and visualize results
    evaluate_model(model, X_test, y_test, target_scaler, model_save_path, history)
    
    # Export for TensorFlow.js
    tfjs_path = f"{model_save_path}/tfjs_model"
    os.makedirs(tfjs_path, exist_ok=True)
    tfjs.converters.save_keras_model(model, tfjs_path)
    
    print("Training completed and all artifacts saved!")

if __name__ == "__main__":
    main()
