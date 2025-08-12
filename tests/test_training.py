#!/usr/bin/env python3
"""
Test script for train_rnn_model.py

This script tests the functionality of the training script to ensure it's working as expected.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from importlib import reload

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Import the training script as a module
import train_rnn_model as trm

# Reload the module in case of changes
reload(trm)

def test_data_loading():
    """Test the data loading and preprocessing function."""
    print("Testing data loading...")
    data_path = 'merged_player_defense.csv'  # Update this path if needed
    df, features = trm.load_and_preprocess_data(data_path)
    
    # Basic checks
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Number of features: {len(features)}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    return df, features

def test_sequence_creation(df, features):
    """Test the sequence creation function."""
    print("\nTesting sequence creation...")
    
    # Test with a single player
    test_player = df['PLAYER_NAME'].value_counts().index[0]
    player_data = df[df['PLAYER_NAME'] == test_player].sort_values('GAME_DATE_EST')
    
    print(f"Creating sequences for {test_player} with {len(player_data)} games")
    
    # Create sequences
    sequence_length = 10
    X, y = trm.create_sequences(
        df=player_data,
        target_column='PTS',
        sequence_length=sequence_length,
        features=features,
        date_column='GAME_DATE_EST'
    )
    
    print(f"Created {len(X)} sequences with shape {X[0].shape}")
    print(f"First sequence features (shape: {X[0].shape}):")
    print(X[0])
    print(f"\nCorresponding target: {y[0]}")
    
    return X, y

def test_model_training(X, y, features):
    """Test training a small model."""
    print("\nTesting model training...")
    
    # Convert X and y to numpy arrays if they aren't already
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and validation
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Print shapes for debugging
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape: {y_val.shape}")
    
    # Train a small model
    print(f"\nTraining on {len(X_train)} sequences, validating on {len(X_val)}")
    
    # For testing, we'll create a simple LSTM model directly
    # instead of using the full training function
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    
    # Define model
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(16),
        BatchNormalization(),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=5,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")
    
    return model, history

def main():
    """Run all tests."""
    print("Starting tests for train_rnn_model.py\n")
    
    # Test data loading
    df, features = test_data_loading()
    
    # Test sequence creation
    X, y = test_sequence_creation(df, features)
    
    # Test model training
    if len(X) > 10:  # Only test training if we have enough sequences
        model, history = test_model_training(X, y, features)
    else:
        print("Not enough sequences for training test.")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
