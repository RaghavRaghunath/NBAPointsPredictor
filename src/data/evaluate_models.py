import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from train_rnn_model import load_and_preprocess_data, create_sequences

def evaluate_model(model_path, X_test, y_test, scaler):
    # Load the model
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Scale the test data
    X_test_scaled = scaler.transform(X_test)
    
    # Create sequences
    X_test_seq = create_sequences(X_test_scaled, sequence_length=10)
    
    # Make predictions
    y_pred = model.predict(X_test_seq)
    
    # Inverse transform the predictions
    y_pred_actual = y_pred * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1]
    y_test_actual = y_test * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1]
    
    # Calculate MAE
    mae = np.mean(np.abs(y_pred_actual - y_test_actual))
    print(f"Model: {model_path}")
    print(f"MAE: {mae:.4f}")
    print("-" * 50)
    return mae

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    df, features = load_and_preprocess_data('merged_player_defense.csv', 'PTS')
    
    # Split data into train and test
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Prepare features and target
    X_train = train_df[features].values
    y_train = train_df['PTS'].values.reshape(-1, 1)
    X_test = test_df[features].values
    y_test = test_df['PTS'].values.reshape(-1, 1)
    
    # Scale features
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train)
    
    # Evaluate all models
    model_files = [
        'nba_player_performance_rnn.h5',
        'rnn_fold_1.h5',
        'rnn_fold_2.h5',
        'rnn_fold_3.h5',
        'rnn_fold_4.h5',
        'rnn_fold_5.h5'
    ]
    
    results = {}
    for model_file in model_files:
        try:
            mae = evaluate_model(model_file, X_test, y_test, y_scaler)
            results[model_file] = mae
        except Exception as e:
            print(f"Error evaluating {model_file}: {str(e)}")
    
    # Print results
    print("\nEvaluation Results:")
    for model, mae in sorted(results.items(), key=lambda x: x[1]):
        print(f"{model}: {mae:.4f} MAE")
