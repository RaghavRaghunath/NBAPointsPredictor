import pandas as pd
import numpy as np
import os
import joblib
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Layer, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
import json
from typing import Tuple, List, Dict, Optional, Any
import logging
import sys
from sklearn.model_selection import GroupKFold
from optuna import Trial
from tensorflow.keras.layers import BatchNormalization

# Get the project root directory (3 levels up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# Add the src directory to the path so we can import from utils
sys.path.append(PROJECT_ROOT)
from src.utils.feature_engineering import calculate_all_emas_and_derived_features

# Set random seeds for reproducibility
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

set_seeds(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set TensorFlow to only log errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 1 = no info, 2 = no warnings, 3 = no errors
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name = 'attention_weight',
                                shape=(input_shape[-1], 1), 
                                initializer='random_normal',
                                trainable=True
                                )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # x represents the shape (batch_size, time_steps, and lstm_units)
        e = tf.tanh(tf.matmul(x, self.W))
        # self.W is a learnable weight matrix that has the shape (lstm_units, 1)
        # .dot() is a function for matrix multiplication, and it produces an attention score for each game
        # the final result, e, has a shape (batch_size, time_steps, lstm_units) that represents raw attention scores for each player
        a = tf.nn.softmax(e, axis=1)
        # Convert the raw attention score for each game into a probability function using softmax
        # Result, a, is a tensor with shape (batch_size, 10, 1).
        context = tf.reduce_sum(x * a, axis=1)
        # Element-wise multiplication
        last_step = x[:, -1, :]
        return context * last_step
        # Above line is the sum over time steps

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        # No custom parameters to add here, but this method is required for serialization
        return config


def calculate_double_doubles_triple_doubles(df):
    """Calculates Double-Doubles (DD2) and Triple-Doubles (TD3) for each game."""
    df['DD2'] = 0
    df['TD3'] = 0

    # Ensure relevant columns are numeric and fill NaNs with 0 for calculation
    for col in ['PTS', 'REB', 'AST', 'STL', 'BLK']:
        if col in df.columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce').fillna(0) # Use .loc to avoid SettingWithCopyWarning
        else:
            warnings.warn(f"Column '{col}' not found in DataFrame for DD2/TD3 calculation. Assuming 0 for missing values.")
            df.loc[:, col] = 0 # Create the column with default 0 if missing, using .loc

    for index, row in df.iterrows():
        double_digit_stats = 0
        if row['PTS'] >= 10:
            double_digit_stats += 1
        if row['REB'] >= 10:
            double_digit_stats += 1
        if row['AST'] >= 10:
            double_digit_stats += 1
        if row['STL'] >= 10:
            double_digit_stats += 1
        if row['BLK'] >= 10:
            double_digit_stats += 1

        if double_digit_stats >= 2:
            df.loc[index, 'DD2'] = 1
        if double_digit_stats >= 3:
            df.loc[index, 'TD3'] = 1
    return df

def load_and_preprocess_data(csv_filename: str, target_stat: str = 'PTS') -> Tuple[pd.DataFrame, List[str]]:
    """Load and preprocess the NBA player data."""
    # Construct the full path to the data file
    data_dir = os.path.join(PROJECT_ROOT, 'data/raw')
    csv_path = os.path.join(data_dir, csv_filename)
    
    print(f"Loading data from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    print("Initial columns:", df.columns.tolist())
    
    # Clean the data: 
    if 'GAME_DATE' in df.columns and 'GAME_DATE_EST' not in df.columns:
        df.rename(columns={'GAME_DATE': 'GAME_DATE_EST'}, inplace=True)
    if 'GAME_DATE_EST' in df.columns:
        df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'], errors='coerce')
    
    # Sort by player and date (important for EMA calculation later)
    if 'PLAYER_NAME' in df.columns and 'GAME_DATE_EST' in df.columns:
        df = df.sort_values(['PLAYER_NAME', 'GAME_DATE_EST'])
    
    # Drop columns with more than 50% missing values
    missing = df.columns[df.isnull().mean() > 0.5]
    print("Columns to drop (due to missing):", missing.tolist())
    if not missing.empty:
        df.drop(columns=missing, inplace=True)
    
    # Fill numeric columns with the mean of the column
    numbers = df.select_dtypes(include = ['float64', 'int64']).columns
    for col in numbers: 
        if df[col].isnull().any():
            df.loc[:, col] = df[col].fillna(df[col].mean()) # Use .loc

    # --- NEW: Calculate all EMA features and derived stats here ---
    df = calculate_all_emas_and_derived_features(df)
    # --- END NEW ---
    
    # Columns that are not feature columns (metadata, target, etc.)
    exclude = [
        'Player_ID', 'GAME_ID', 'SEASON_ID', 'GAME_DATE_EST', 
        'MIN', 'VIDEO_AVAILABLE', 'PLAYER_NAME', 'TEAM_ABBREVIATION',
        'MATCHUP', 'WL', 'MIN_STR', 'PLUS_MINUS', 'GAME_DATE', # PLUS_MINUS is now handled via 'PM'
        # 'IS_HOME', 'DAYS_REST', 'IS_BACK_TO_BACK', # These are direct features, NOT to be excluded from final feature list
        'IS_INJURED', 'IS_PLAYOFF_PUSH', 'POSITION', 'POSITION_GROUP', 'TEAM_PACE',
        'TEAM_OFF_RTG', 'TEAM_DEF_RTG', '_merge', 'ROLLING_AVG_PTS', 'PTS_DIFF',
        'VS_BEST_DEF_PG', 'VS_WORST_DEF_PG', 'VS_BEST_DEF_SG', 'VS_WORST_DEF_SG',
        'VS_BEST_DEF_SF', 'VS_WORST_DEF_SF', 'VS_BEST_DEF_PF', 'VS_WORST_DEF_PF',
        'VS_BEST_DEF_C', 'VS_WORST_DEF_C', 'DD2', 'TD3' # DD2/TD3 are base for EMA, not direct features
    ]
    
    # All columns that could potentially be features after preprocessing
    # This list will now include the newly calculated EMA features
    potential_features = [col for col in df.columns if col not in exclude and col != target_stat]
    
    # Ensure we don't have duplicate features
    features = list(dict.fromkeys(potential_features))
    
    print(f"Selected {len(features)} features for modeling (pre-filter)")
    
    # Remove remaining rows with missing values that are critical for features or target
    # This is important after EMA calculation, as initial rows might have NaNs for EMAs
    df = df.dropna(subset=features + [target_stat])
    
    # Drop any duplicate rows
    df = df.drop_duplicates()
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Number of unique players: {df['PLAYER_NAME'].nunique()}")
    
    return df, features

def create_sequences(df, target_column, sequence_length, features=None, date_column='GAME_DATE_EST'):
    """
    Create sequences of features and targets for time series prediction.
    
    Args:
        df: DataFrame containing the data
        target_column: Name of the target column
        sequence_length: Number of time steps to include in each sequence
        features: List of feature columns to include (if None, use all columns except target)
        date_column: Name of the date column to sort by
    
    Returns:
        X: Numpy array of sequences (n_samples, sequence_length, n_features)
        y: Numpy array of target values (n_samples,)
    """
    sequences = []
    targets = []
    
    # If features not provided, use all columns except target and metadata
    if features is None:
        features = [col for col in df.columns 
                   if col not in [target_column, 'PLAYER_NAME', 'PLAYER_ID', 
                                date_column, 'GAME_ID', 'SEASON_ID']]
    
    # Check if required columns exist
    required_columns = features + [target_column]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in data: {missing_cols}")
    
    # Group by player if player column exists, otherwise treat as single group
    if 'PLAYER_NAME' in df.columns:
        grouped = df.groupby('PLAYER_NAME')
    elif 'PLAYER_ID' in df.columns:
        grouped = df.groupby('PLAYER_ID')
    else:
        grouped = [('all', df)]
    
    # Create sequences for each group
    for group_name, group in grouped:
        # Sort by date if available, otherwise use existing order
        if date_column in group.columns and not group[date_column].isna().all():
            group = group.sort_values(date_column)
        else:
            print(f"Warning: Date column '{date_column}' not found or all NaNs for group {group_name}. Using existing order.")
        
        # Skip groups that are too small
        if len(group) <= sequence_length:
            continue
            
        # Get features and targets
        group_features = group[features].values
        group_targets = group[target_column].values
        
        # Create sequences
        for i in range(len(group) - sequence_length):
            sequences.append(group_features[i:(i + sequence_length)])
            targets.append(group_targets[i + sequence_length])
    
    # Check if any sequences were created
    if len(sequences) == 0:
        raise ValueError("No sequences were created. Check your data and sequence length.")
    
    # Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(targets)
    
    print(f"Created {len(X)} sequences with {len(features)} features each")
    print(f"X shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y


def build_model(input_shape, lstm_units1=128, lstm_units2=64, dense_units=64, 
               dropout_rate=0.3, learning_rate=0.001, optimizer_name='adam'):
    model_input = Input(shape=input_shape)

    # First LSTM layer
    lstm1 = LSTM(
        units=lstm_units1,
        return_sequences=True,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate*0.5
    )(model_input)
    bn1 = BatchNormalization()(lstm1)
    act1 = tf.keras.layers.Activation('tanh')(bn1) # Using string 'tanh'
    drop1 = Dropout(dropout_rate)(act1)
    
    # Second LSTM layer
    lstm2 = LSTM(
        units=lstm_units2,
        return_sequences=True, # Keep return_sequences=True for AttentionLayer
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate*0.5
    )(drop1)
    bn2 = BatchNormalization()(lstm2)
    act2 = tf.keras.layers.Activation('tanh')(bn2) # Using string 'tanh'
    drop2 = Dropout(dropout_rate)(act2)

    # Attention Layer
    attention_output = AttentionLayer()(drop2)
    
    # Dense layers for final prediction
    dense1 = Dense(dense_units, activation='relu')(attention_output) # 'relu' is already a string
    dense_bn1 = BatchNormalization()(dense1)
    dense_act1 = tf.keras.layers.Activation('relu')(dense_bn1) # Using string 'relu'
    dense_drop1 = Dropout(dropout_rate)(dense_act1)
    
    output = Dense(1)(dense_drop1) # Linear activation for regression

    model = tf.keras.Model(inputs=model_input, outputs=output)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model(X_train_df, y_train_series, X_val_df, y_val_series, features, sequence_length=10, 
               lstm_units1=128, lstm_units2=64, dense_units=32, dropout_rate=0.2, 
               learning_rate=0.001, batch_size=32, epochs=100, patience=10, 
               min_delta=0.001, fold_number=None, target_stat='PTS'):
    """
    Train the RNN model with the given parameters and return the trained model and training history.
    
    Args:
        X_train_df: Training features DataFrame
        y_train_series: Training target values (Series)
        X_val_df: Validation features DataFrame
        y_val_series: Validation target values (Series)
        features: List of feature columns to use
        sequence_length: Number of time steps to include in each sequence
        lstm_units1: Number of units in first LSTM layer
        lstm_units2: Number of units in second LSTM layer
        dense_units: Number of units in dense layer
        dropout_rate: Dropout rate
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        patience: Patience for early stopping
        min_delta: Minimum change to qualify as improvement
        fold_number: Fold number for cross-validation (or 'final' for final model)
        target_stat: Target statistic for model (e.g., 'PTS', 'REB', etc.)
    
    Returns:
        model: The trained Keras model.
        history: The training history object.
    """
    print("\n" + "="*50)
    print(f"TRAINING {'FINAL' if str(fold_number).lower() == 'final' else f'FOLD {fold_number}'} MODEL")
    print("="*50)

    # Set up callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,  
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Scale the features
    print("\nScaling features...")
    global scaler # Use the global scaler
    scaler = StandardScaler()
    
    # Ensure all features are numeric before scaling and handle missing columns
    # Make copies to avoid SettingWithCopyWarning
    X_train_df_copy = X_train_df.copy()
    X_val_df_copy = X_val_df.copy()

    for col in features:
        if col not in X_train_df_copy.columns:
            warnings.warn(f"Feature '{col}' not found in X_train_df. It will be treated as missing during scaling.")
            X_train_df_copy.loc[:, col] = 0.0 # Add missing column with default value
        X_train_df_copy.loc[:, col] = pd.to_numeric(X_train_df_copy[col], errors='coerce').fillna(X_train_df_copy[col].mean() if not X_train_df_copy[col].empty else 0.0)
        
        if col not in X_val_df_copy.columns:
            warnings.warn(f"Feature '{col}' not found in X_val_df. It will be treated as missing during scaling.")
            X_val_df_copy.loc[:, col] = 0.0 # Add missing column with default value
        X_val_df_copy.loc[:, col] = pd.to_numeric(X_val_df_copy[col], errors='coerce').fillna(X_val_df_copy[col].mean() if not X_val_df_copy[col].empty else 0.0)

    # Fit scaler only on training data for the specified features
    scaler.fit(X_train_df_copy[features])
    X_train_scaled = scaler.transform(X_train_df_copy[features])
    X_val_scaled = scaler.transform(X_val_df_copy[features])
    
    # Save the scaler for later use in app.py
    # Save the scaler for later use in app.py
    # Using the original path that the rest of the code expects
    scaler_path = 'feature_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Create DataFrames from scaled arrays to pass to create_sequences
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features, index=X_train_df_copy.index)
    X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=features, index=X_val_df_copy.index)

    # Add back necessary metadata for create_sequences (PLAYER_NAME, GAME_DATE_EST)
    # AND THE TARGET COLUMN (PTS)
    for col in ['PLAYER_NAME', 'GAME_DATE_EST', target_stat]: # Added target_stat here
        if col in X_train_df_copy.columns:
            X_train_scaled_df.loc[:, col] = X_train_df_copy[col] # Use .loc
        elif col == target_stat: # If target_stat was not in X_train_df_copy (e.g., if it was only in y_train_series)
            X_train_scaled_df.loc[:, col] = y_train_series.loc[X_train_scaled_df.index] # Align index

        if col in X_val_df_copy.columns:
            X_val_scaled_df.loc[:, col] = X_val_df_copy[col] # Use .loc
        elif col == target_stat: # If target_stat was not in X_val_df_copy
            X_val_scaled_df.loc[:, col] = y_val_series.loc[X_val_scaled_df.index] # Align index


    # Create sequences for training and validation
    print("\nCreating training sequences...")
    try:
        X_train_sequence, y_train_sequence = create_sequences(
            X_train_scaled_df, 
            target_column=target_stat, 
            sequence_length=sequence_length,
            features=features,
            date_column='GAME_DATE_EST'
        )
    except Exception as e:
        print(f"Error creating training sequences: {str(e)}")
        raise
    
    print("\nCreating validation sequences...")
    try:
        X_val_sequence, y_val_sequence = create_sequences(
            X_val_scaled_df,
            target_column=target_stat,
            sequence_length=sequence_length,
            features=features,
            date_column='GAME_DATE_EST'
        )
    except Exception as e:
        print(f"Error creating validation sequences: {str(e)}")
        raise
    
    # Print shapes for debugging
    print(f"\nTraining sequences: {X_train_sequence.shape}")
    print(f"Training targets: {y_train_sequence.shape}")
    print(f"Validation sequences: {X_val_sequence.shape}")
    print(f"Validation targets: {y_val_sequence.shape}")

    # Verify input shape matches model expectation
    expected_input_shape = (sequence_length, len(features))
    if X_train_sequence.shape[1:] != expected_input_shape:
        raise ValueError(f"Mismatch in X_train_sequence shape: Expected {expected_input_shape}, got {X_train_sequence.shape[1:]}. Check feature list and sequence creation.")
    
    # Build the model
    print("\nBuilding model...")
    model = build_model(
        input_shape=(X_train_sequence.shape[1], X_train_sequence.shape[2]),
        lstm_units1=lstm_units1,
        lstm_units2=lstm_units2,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
    model.summary() # Print model summary to confirm input shape

    # Train the model
    print("\nTraining model...")
    history = model.fit(
        X_train_sequence, y_train_sequence,
        validation_data=(X_val_sequence, y_val_sequence),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Ensure the models directory exists
    os.makedirs('data/models', exist_ok=True)
    
    # Save the model with the target stat in the filename
    model_filename = f"data/models/nba_player_performance_rnn_{target_stat}.h5"
    model.save(model_filename)
    print(f"\nModel saved to {model_filename}")
    
    # Also save the scaler with the target stat in the filename
    scaler_filename = f"data/models/feature_scaler_{target_stat}.pkl"
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler saved to {scaler_filename}")
    
    # Plot training history
    if history is not None:
        plt.figure(figsize=(12, 4))
        
        # Plot training & validation loss values
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'])
            plt.legend(['Train', 'Validation'], loc='upper right')
        else:
            plt.legend(['Train'], loc='upper right')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        
        # Plot training & validation MAE
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'])
        if 'val_mae' in history.history:
            plt.plot(history.history['val_mae'])
            plt.legend(['Train', 'Validation'], loc='upper right')
        else:
            plt.legend(['Train'], loc='upper right')
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        print("Training history plot saved to training_history.png")
    
    return model, history # Only return model and history


def evaluate_model(model, X_test_df, y_test_series, features, scaler, target_stat='PTS'):
    """Evaluate the model and return metrics and predictions."""
    if scaler is not None:
        X_test_df_copy = X_test_df.copy() # Make a copy to avoid SettingWithCopyWarning
        # Ensure all features are numeric before scaling
        for col in features:
            if col not in X_test_df_copy.columns:
                warnings.warn(f"Feature '{col}' not found in X_test_df during evaluation. It will be treated as missing.")
                X_test_df_copy.loc[:, col] = 0.0 # Add missing column with default value
            X_test_df_copy.loc[:, col] = pd.to_numeric(X_test_df_copy[col], errors='coerce').fillna(X_test_df_copy[col].mean() if not X_test_df_copy[col].empty else 0.0)

        X_test_scaled = scaler.transform(X_test_df_copy[features])
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features, index=X_test_df_copy.index)
    else:
        X_test_scaled_df = X_test_df.copy() # Still make a copy
    
    # Add target column for sequence creation
    test_df_for_seq = X_test_scaled_df.copy()
    test_df_for_seq.loc[:, target_stat] = y_test_series.loc[test_df_for_seq.index] # Use .loc and align index

    # Add back necessary metadata for create_sequences (PLAYER_NAME, GAME_DATE_EST)
    for col in ['PLAYER_NAME', 'GAME_DATE_EST']:
        if col in X_test_df.columns:
            test_df_for_seq.loc[:, col] = X_test_df[col] # Use .loc

    # Create sequences for evaluation
    X_test_seq, y_test_seq = create_sequences(test_df_for_seq, target_stat, 10, features=features, date_column='GAME_DATE_EST')
    
    # Evaluate model
    metrics = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    predictions = model.predict(X_test_seq, verbose=0)
    
    return metrics, predictions, y_test_seq

def analyze_features(df, features, target='PTS'):
    """Analyze features and their relationship with the target variable."""
    print("\n" + "="*50)
    print("FEATURE ANALYSIS")
    print("="*50)
    
    # Basic statistics
    print(f"\nTarget variable ({target}) statistics:")
    print(df[target].describe())
    
    # Check for high correlation with target
    print("\nTop 20 features most correlated with target:")
    # Ensure all features exist in df before calculating correlation
    features_present = [f for f in features if f in df.columns]
    if target in df.columns:
        correlations = df[features_present + [target]].corr()[target].sort_values(key=abs, ascending=False)
        print(correlations.head(20))
    else:
        print(f"Warning: Target column '{target}' not found in DataFrame. Cannot calculate correlations.")
        correlations = pd.Series([]) # Return empty series

    # Check for constant or near-constant features
    print("\nFeatures with low variance (potential candidates for removal):")
    for col in features_present:
        if df[col].nunique() == 1:
            print(f"{col}: Constant value")
        elif df[col].nunique() < 5:
            print(f"{col}: Only {df[col].nunique()} unique values")
    
    return correlations

def objective(trial, X_train_sequence, y_train_sequence, X_val_sequence, y_val_sequence):
    """Objective function for hyperparameter optimization."""
    # Define hyperparameter search space
    params = {
        'lstm_units1': trial.suggest_int('lstm_units1', 32, 256, step=32),
        'lstm_units2': trial.suggest_int('lstm_units2', 16, 128, step=16),
        'dense_units': trial.suggest_int('dense_units', 0, 128, step=8),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    }
    
    # Build model with current hyperparameters
    model = build_model(
        input_shape=X_train_sequence.shape[1:],
        lstm_units1=params['lstm_units1'],
        lstm_units2=params['lstm_units2'],
        dense_units=params['dense_units'],
        dropout_rate=params['dropout_rate'],
        learning_rate=params['learning_rate']
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_sequence, y_train_sequence,
        validation_data=(X_val_sequence, y_val_sequence),
        epochs=100,
        batch_size=params['batch_size'],
        callbacks=[early_stopping],
        verbose=0
    )

    # Get the best validation MAE
    val_mae = min(history.history['val_mae'])
    return val_mae

def optimize_hyperparameters(X_train_sequence, y_train_sequence, X_val_sequence, y_val_sequence, n_trials=50):
    print("Starting hyperparameter optimization...")
    print(f"Training data shape: {X_train_sequence.shape}, Validation data shape: {X_val_sequence.shape}")
    
    # Create a study with a pruner for early stopping unpromising trials
    study = create_study(
        direction='minimize',
        study_name='lstm_hyperparams',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, X_train_sequence, y_train_sequence, X_val_sequence, y_val_sequence),
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True  # Helps with memory management
    )
    
    # Print results
    print("\nHyperparameter optimization completed!")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (min validation loss): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Plot optimization history
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()
        
        # Plot parameter importance
        fig = optuna.visualization.plot_param_importances(study)
        fig.show()
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    return trial.params

def main(optimize_hyperparams=False, n_trials=20, target_stat='PTS'):
    set_seeds(42)
    
    print("Loading and preprocessing data...")
    # Use the dataset that previously gave best results
    df, all_features = load_and_preprocess_data('merged_player_defense.csv', target_stat)
    
    # Define the exact 91 features that the model expects and app.py uses
    # This list MUST match the features_list_order in app.py
    important_features = [
        'PLAYER_PTS_EMA_5', 'PLAYER_PTS_EMA_10', 'PLAYER_PTS_EMA_20', 'PLAYER_PTS_EMA_50',
        'PLAYER_FGA_EMA_5', 'PLAYER_FGA_EMA_10', 'PLAYER_FGA_EMA_20', 'PLAYER_FGA_EMA_50',
        'PLAYER_FGM_EMA_5', 'PLAYER_FGM_EMA_10', 'PLAYER_FGM_EMA_20', 'PLAYER_FGM_EMA_50',
        'PLAYER_FG_PCT_EMA_5', 'PLAYER_FG_PCT_EMA_10', 'PLAYER_FG_PCT_EMA_20', 'PLAYER_FG_PCT_EMA_50',
        'PLAYER_FG3M_EMA_5', 'PLAYER_FG3M_EMA_10', 'PLAYER_FG3M_EMA_20', 'PLAYER_FG3M_EMA_50',
        'PLAYER_FG3A_EMA_5', 'PLAYER_FG3A_EMA_10', 'PLAYER_FG3A_EMA_20', 'PLAYER_FG3A_EMA_50',
        'PLAYER_FG3_PCT_EMA_5', 'PLAYER_FG3_PCT_EMA_10', 'PLAYER_FG3_PCT_EMA_20', 'PLAYER_FG3_PCT_EMA_50',
        'PLAYER_FTM_EMA_5', 'PLAYER_FTM_EMA_10', 'PLAYER_FTM_EMA_20', 'PLAYER_FTM_EMA_50',
        'PLAYER_FTA_EMA_5', 'PLAYER_FTA_EMA_10', 'PLAYER_FTA_EMA_20', 'PLAYER_FTA_EMA_50',
        'PLAYER_FT_PCT_EMA_5', 'PLAYER_FT_PCT_EMA_10', 'PLAYER_FT_PCT_EMA_20', 'PLAYER_FT_PCT_EMA_50',
        'PLAYER_OREB_EMA_5', 'PLAYER_OREB_EMA_10', 'PLAYER_OREB_EMA_20', 'PLAYER_OREB_EMA_50',
        'PLAYER_DREB_EMA_5', 'PLAYER_DREB_EMA_10', 'PLAYER_DREB_EMA_20', 'PLAYER_DREB_EMA_50',
        'PLAYER_REB_EMA_5', 'PLAYER_REB_EMA_10', 'PLAYER_REB_EMA_20', 'PLAYER_REB_EMA_50',
        'PLAYER_AST_EMA_5', 'PLAYER_AST_EMA_10', 'PLAYER_AST_EMA_20', 'PLAYER_AST_EMA_50',
        'PLAYER_STL_EMA_5', 'PLAYER_STL_EMA_10', 'PLAYER_STL_EMA_20', 'PLAYER_STL_EMA_50',
        'PLAYER_BLK_EMA_5', 'PLAYER_BLK_EMA_10', 'PLAYER_BLK_EMA_20', 'PLAYER_BLK_EMA_50',
        'PLAYER_TOV_EMA_5', 'PLAYER_TOV_EMA_10', 'PLAYER_TOV_EMA_20', 'PLAYER_TOV_EMA_50',
        'PLAYER_PF_EMA_5', 'PLAYER_PF_EMA_10', 'PLAYER_PF_EMA_20', 'PLAYER_PF_EMA_50',
        'PLAYER_MIN_EMA_5', 'PLAYER_MIN_EMA_10', 'PLAYER_MIN_EMA_20', 'PLAYER_MIN_EMA_50',
        'PLAYER_PM_EMA_5', 'PLAYER_PM_EMA_10', 'PLAYER_PM_EMA_20', 'PLAYER_PM_EMA_50', # PLUS_MINUS -> PM
        'IS_HOME', 'DAYS_REST', 'IS_BACK_TO_BACK', # These are now included as direct features
        'DEF_OPP_PTS_EMA_5', 'DEF_OPP_PTS_EMA_10', 'DEF_OPP_PTS_EMA_20',
        'DEF_OPP_FG_PCT_EMA_5', 'DEF_OPP_FG_PCT_EMA_10', 'DEF_OPP_FG_PCT_EMA_20',
        'PLAYER_DD2_EMA_5', # Double-Doubles EMA 5
        'PLAYER_TD3_EMA_5'  # Triple-Doubles EMA 5
    ]
    
    # Filter features to only include those that exist in the dataframe
    # This ensures no KeyError if a feature is missing from the raw data
    features = [f for f in important_features if f in df.columns]
    print(f"Using {len(features)} features for training (after data check)")
    
    # Ensure we have the required columns for splitting and target
    required_columns = ['PLAYER_NAME', 'GAME_DATE_EST', 'PTS']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing critical columns for training: {missing_columns}. Please check your 'merged_player_defense.csv' and data loading logic.")
    
    # Sort by player and date
    df = df.sort_values(['PLAYER_NAME', 'GAME_DATE_EST'])
    
    # Analyze features before proceeding
    correlations = analyze_features(df, features, target='PTS')
    
    # IMPORTANT: Ensure these lines are commented out or removed!
    # top_n = min(50, len(features))
    # top_features = correlations.index[1:top_n+1].tolist()
    # features = [f for f in features if f in top_features]
    
    print(f"Final number of features to be used for model: {len(features)}")
    
    # Create target and features
    X = df[features] # Use the full list of 91 features
    y = df['PTS']
    groups = df['PLAYER_NAME'] # For GroupKFold
    
    # Split data into train, validation, and test sets
    print("\nSplitting data into train, validation, and test sets...")
    
    # Use GroupKFold to ensure players are not split across sets
    # First split: 80% train+val, 20% test
    gkf_initial = GroupKFold(n_splits=5) # Using 5 folds for 80/20 split (1/5 = 20%)
    train_val_idx, test_idx = next(gkf_initial.split(X, y, groups))

    X_train_val, X_test = X.iloc[train_val_idx], X.iloc[test_idx]
    y_train_val, y_test = y.iloc[train_val_idx], y.iloc[test_idx]
    groups_train_val = groups.iloc[train_val_idx]

    print(f"Training + Validation set size: {len(X_train_val)}")
    print(f"Test set size: {len(X_test)}")

    # Second split: From train+val, split into 80% train, 20% validation
    gkf_train_val = GroupKFold(n_splits=5) # Using 5 folds for 80/20 split (1/5 = 20%)
    train_idx, val_idx = next(gkf_train_val.split(X_train_val, y_train_val, groups_train_val))

    X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
    y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")

    # Hyperparameter optimization (optional)
    best_params = {
        'lstm_units1': 128, 'lstm_units2': 64, 'dense_units': 64,
        'dropout_rate': 0.3, 'learning_rate': 0.001, 'batch_size': 32
    } # Default values

    if optimize_hyperparams:
        # Before optimization, we need to create sequences from X_train, y_train, X_val, y_val
        # The scaler needs to be fitted here once for the optimization process
        temp_scaler = StandardScaler()
        numeric_features_for_scaling = [f for f in features if f in X_train.select_dtypes(include=['float64', 'int64']).columns]
        
        X_train_scaled_for_opt = X_train.copy()
        X_val_scaled_for_opt = X_val.copy()

        # Ensure all features are numeric before scaling and handle missing columns
        for col in features:
            if col not in X_train_scaled_for_opt.columns:
                X_train_scaled_for_opt.loc[:, col] = 0.0
            X_train_scaled_for_opt.loc[:, col] = pd.to_numeric(X_train_scaled_for_opt[col], errors='coerce').fillna(X_train_scaled_for_opt[col].mean() if not X_train_scaled_for_opt[col].empty else 0.0)
            
            if col not in X_val_scaled_for_opt.columns:
                X_val_scaled_for_opt.loc[:, col] = 0.0
            X_val_scaled_for_opt.loc[:, col] = pd.to_numeric(X_val_scaled_for_opt[col], errors='coerce').fillna(X_val_scaled_for_opt[col].mean() if not X_val_scaled_for_opt[col].empty else 0.0)

        if numeric_features_for_scaling:
            temp_scaler.fit(X_train_scaled_for_opt[numeric_features_for_scaling])
            X_train_scaled_for_opt.loc[:, numeric_features_for_scaling] = temp_scaler.transform(X_train_scaled_for_opt[numeric_features_for_scaling])
            X_val_scaled_for_opt.loc[:, numeric_features_for_scaling] = temp_scaler.transform(X_val_scaled_for_opt[numeric_features_for_scaling])
        else:
            print("Warning: No numeric features for scaling in optimization data.")


        # Create sequences for optimization
        # Add 'PTS' column to the scaled dataframes before creating sequences
        X_train_scaled_for_opt.loc[:, 'PTS'] = y_train.loc[X_train_scaled_for_opt.index]
        X_val_scaled_for_opt.loc[:, 'PTS'] = y_val.loc[X_val_scaled_for_opt.index]

        X_train_seq_opt, y_train_seq_opt = create_sequences(
            X_train_scaled_for_opt,
            target_column='PTS',
            sequence_length=10,
            features=features,
            date_column='GAME_DATE_EST'
        )
        X_val_seq_opt, y_val_seq_opt = create_sequences(
            X_val_scaled_for_opt,
            target_column='PTS',
            sequence_length=10,
            features=features,
            date_column='GAME_DATE_EST'
        )

        if X_train_seq_opt.shape[0] == 0 or X_val_seq_opt.shape[0] == 0:
            print("Not enough sequences for hyperparameter optimization after creation. Skipping optimization.")
        else:
            best_params = optimize_hyperparameters(X_train_seq_opt, y_train_seq_opt, X_val_seq_opt, y_val_seq_opt, n_trials=n_trials)

    # Train the final model using the best parameters (or defaults)
    # Pass X_val and y_val to train_model as its validation set
    final_model, history = train_model(
        X_train, y_train, X_val, y_val, # Pass X_val, y_val here
        features=features,
        sequence_length=10,
        lstm_units1=best_params['lstm_units1'],
        lstm_units2=best_params['lstm_units2'],
        dense_units=best_params['dense_units'],
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate'],
        batch_size=best_params['batch_size'],
        epochs=150, # Set a reasonable max epochs
        patience=20, # Increased patience for final model
        fold_number='final'
    )
    
    # Evaluate on the test set
    print("\nEvaluating final model on test set...")
    test_metrics, predictions, y_true = evaluate_model(final_model, X_test, y_test, features, scaler)
    test_mae = test_metrics[1]
    print(f"Final Test Loss: {test_metrics[0]:.4f}, Final Test MAE: {test_mae:.4f}")

    # Save predictions and true values for analysis
    results_df = pd.DataFrame({
        'true_pts': y_true.flatten(),
        'predicted_pts': predictions.flatten()
    })
    
    # Add back player and game information to results_df if available
    # Align indices to merge correctly
    test_df_for_results = df.loc[y_test.index].copy() # Get original rows from df using y_test's index
    
    # Need to align predictions/y_true with the original game info
    # The 'create_sequences' function shifts the target, so the original game info
    # for y_test_sequence[i] is X_test_sequence[i] (the game at the end of the sequence)
    # This is complex to re-align perfectly without tracking original indices during sequence creation.
    # For simplicity, we'll just save the predictions and true values.
    
    results_df['abs_error'] = np.abs(results_df['true_pts'] - results_df['predicted_pts'])
    
    results_df.to_csv('model_predictions.csv', index=False)
    print("\nDetailed predictions saved to 'model_predictions.csv'")

    # Removed calculate_feature_importance as requested
    # feature_importances = calculate_feature_importance(final_model, X_test, y_test, features, scaler)
    # print("\nComplete feature importance saved to rnn_feature_importance.csv")
    
    defensive_features = [col for col in df.columns if 'DEF_OPP_' in col]
    if defensive_features:
        print("\nAnalyzing defensive features correlations with target:")
        def_corrs = df[defensive_features + ['PTS']].corr()['PTS'].sort_values(ascending=False)
        print(def_corrs)

if __name__ == '__main__':
    # Set optimize_hyperparams to True to run hyperparameter tuning
    # Set n_trials to a higher number (e.g., 50-100) for more robust tuning
    main(optimize_hyperparams=False, n_trials=5)
