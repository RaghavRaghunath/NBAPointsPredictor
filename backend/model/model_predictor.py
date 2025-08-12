import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Dense, Dropout, MultiHeadAttention, LayerNormalization
from tensorflow.keras import backend as K
from typing import Dict, List, Tuple, Optional, Any
import joblib
from datetime import datetime, timedelta
import logging
import warnings
import sys

# Add the src directory to the path so we can import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.feature_engineering import calculate_all_emas_and_derived_features

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionLayer(Layer):
    """
    Attention mechanism layer for the RNN model.
    This is a simplified version that should match the saved model's expectations.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        # Create trainable weights
        self.attention_weights = self.add_weight(
            name='attention_weights',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x, mask=None):
        # Calculate attention scores
        attention_scores = tf.tensordot(x, self.attention_weights, axes=1)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores += (1.0 - tf.cast(mask, tf.float32)) * -1e9
            
        # Calculate attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        
        # Apply attention weights
        context_vector = tf.reduce_sum(x * attention_weights, axis=1)
        return context_vector
    
    def compute_mask(self, inputs, mask=None):
        return None
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config

class ModelPredictor:
    def __init__(self):
        # Define the exact features expected by the model and scaler
        self.feature_columns = [
            # Base stats (20)
            'PTS', 'FGA', 'FGM', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
            'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 
            'BLK', 'TOV', 'PF', 'MIN', 'PLUS_MINUS',
            
            # Game context (6)
            'IS_HOME', 'DAYS_REST', 'IS_BACK_TO_BACK',
            'TEAM_PACE', 'TEAM_OFF_RTG', 'TEAM_DEF_RTG',
            
            # Advanced metrics (2)
            'TS_PCT', 'USG_PCT',
            
            # Derived stats (3)
            'PTS_PER_MIN', 'AST_PER_MIN', 'REB_PER_MIN',
            
            # Player EMAs (5, 10, 20 game windows) - 14 stats * 3 windows = 42
            *[f'PLAYER_{stat}_EMA_{window}' 
              for stat in [
                  'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 
                  'FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM', 'MIN', 'PM'
              ]
              for window in [5, 10, 20]],
            
            # Defense EMAs - 2 stats * 3 windows = 6
            *[f'DEF_OPP_{stat}_EMA_{window}'
              for stat in ['PTS', 'FG_PCT']
              for window in [5, 10, 20]],
            
            # Special EMAs - 2
            'PLAYER_DD2_EMA_5', 'PLAYER_TD3_EMA_5',
            
            # Additional features to reach 91 (10 features)
            'PLAYER_DD2_EMA_10', 'PLAYER_DD2_EMA_20',
            'PLAYER_TD3_EMA_10', 'PLAYER_TD3_EMA_20',
            'DEF_OPP_FG3_PCT_EMA_5', 'DEF_OPP_FG3_PCT_EMA_10', 'DEF_OPP_FG3_PCT_EMA_20',
            'PLAYER_PF_EMA_5', 'PLAYER_PF_EMA_10', 'PLAYER_PF_EMA_20'
        ]
        
        # Log the feature columns for debugging
        print("\n=== FEATURE COLUMNS ===")
        print(f"Total features: {len(self.feature_columns)}")
        print("\nFeature list:")
        for i, col in enumerate(self.feature_columns, 1):
            print(f"{i}. {col}")
            
        # Ensure we have exactly 91 features to match the scaler
        assert len(self.feature_columns) == 91, \
            f"Expected 91 features but got {len(self.feature_columns)}. " \
            f"Please update feature_columns to match the scaler's expectations."
        self.sequence_length = 10
        # Define paths relative to the project root (go up two levels from backend/model/ to reach project root)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.models_dir = os.path.join(project_root, 'data', 'models')
        self.default_csv = os.path.join(project_root, 'data', 'raw', 'merged_player_defense.csv')
        
        # Define model and scaler files (points only)
        self.model_file = 'nba_player_performance_rnn_PTS.h5'
        self.scaler_file = 'feature_scaler_PTS.pkl'
        
        # Log the paths for debugging
        print(f"Models directory: {self.models_dir}")
        print(f"CSV path: {self.default_csv}")
        
        # Initialize model and scaler caches
        self.models = {}
        self.scalers = {}
        self._load_models()
    
    def _load_models(self):
        """Load the pre-trained model and scaler for points prediction."""
        print("\n=== Starting model loading process ===")
        
        # Load the scaler
        scaler_path = os.path.join(self.models_dir, self.scaler_file)
        print(f"\nLoading scaler from: {scaler_path}")
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at: {scaler_path}")
        
        try:
            self.scaler = joblib.load(scaler_path)
            print("✅ Scaler loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading scaler: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # Load the model
        model_path = os.path.join(self.models_dir, self.model_file)
        print(f"\nLoading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        # Try different approaches to load the model
        loaded = False
        
        # Attempt 1: Load with custom_objects
        if not loaded:
            try:
                print("Attempt 1: Loading with custom_objects...")
                self.model = load_model(
                    model_path, 
                    custom_objects={'AttentionLayer': AttentionLayer},
                    compile=False
                )
                print("✅ Model loaded successfully!")
                loaded = True
            except Exception as e:
                print(f"❌ Attempt 1 failed: {str(e)}")
        
        # Attempt 2: Load with custom_objects_scope
        if not loaded:
            try:
                print("Attempt 2: Loading with custom_objects_scope...")
                with tf.keras.utils.custom_object_scope({'AttentionLayer': AttentionLayer}):
                    self.model = load_model(model_path, compile=False)
                print("✅ Model loaded successfully!")
                loaded = True
            except Exception as e:
                print(f"❌ Attempt 2 failed: {str(e)}")
        
        # Attempt 3: Create model and load weights
        if not loaded:
            try:
                print("Attempt 3: Creating model and loading weights...")
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM, Dense, Dropout
                
                self.model = Sequential([
                    LSTM(64, return_sequences=True, input_shape=(10, len(self.feature_columns))),
                    AttentionLayer(),
                    Dense(32, activation='relu'),
                    Dropout(0.2),
                    Dense(1)
                ])
                
                # Load weights
                self.model.load_weights(model_path)
                print("✅ Model loaded with weights!")
                loaded = True
            except Exception as e:
                print(f"❌ Attempt 3 failed: {str(e)}")
        
        if not loaded:
            raise RuntimeError("Failed to load model after all attempts")
        
        print("\n=== Model loading process completed ===\n")

    def build_sequence(
            self, 
            player_name: str,
            as_of_date: Optional[str] = None,
            csv_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Build a sequence of features for the given player as of a specific date.
        
        Args:
            player_name (str): Name of the player.
            as_of_date (Optional[str]): Date to filter the data. If None, uses the latest data.
            csv_path (Optional[str]): Path to the CSV file containing player data. Defaults to DEFAULT_CSV.
        
        Returns:
            np.ndarray: A sequence of features for the player.
        """
        path = csv_path or self.default_csv
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found at {path}")
            
        # Load the data
        df = pd.read_csv(path, low_memory=False)
        
        # Filter for the specific player
        pdf = df[df['PLAYER_NAME'] == player_name].copy()
        if pdf.empty:
            raise ValueError(f"No data found for player: {player_name}")
        
        # Log initial columns before EMA calculation
        print("\n=== BEFORE EMA CALCULATION ===")
        print(f"Initial columns in player data: {len(pdf.columns)}")
        print(f"Columns: {sorted(pdf.columns.tolist())}")
        
        # Ensure we have the required columns for EMA calculations
        required_cols = ['PTS', 'FGA', 'FGM', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
                        'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 
                        'BLK', 'TOV', 'PF', 'MIN', 'PLUS_MINUS', 'DD2', 'TD3']
        
        for col in required_cols:
            if col not in pdf.columns:
                print(f"Adding missing required column: {col}")
                pdf[col] = 0.0
        
        # Calculate EMA features (same as during training)
        try:
            print("\n=== CALCULATING EMA FEATURES ===")
            pdf = calculate_all_emas_and_derived_features(pdf)
            print(f"After EMA calculation, columns: {len(pdf.columns)}")
            # Print EMA columns that were added
            ema_cols = [col for col in pdf.columns if 'EMA_' in col]
            print(f"EMA columns added: {len(ema_cols)}")
            print(f"Total columns after EMA: {len(pdf.columns)}")
        except Exception as e:
            print(f"Error calculating EMA features: {str(e)}")
            raise
        
        # Convert GAME_DATE to datetime and sort
        date_col = 'GAME_DATE' if 'GAME_DATE' in pdf.columns else 'GAME_DATE_EST'
        pdf[date_col] = pd.to_datetime(pdf[date_col])
        pdf = pdf.sort_values(date_col)
        
        # Filter by date if provided
        if as_of_date:
            as_of_date = pd.to_datetime(as_of_date)
            pdf = pdf[pdf[date_col] <= as_of_date]
        
        # Take the most recent sequence_length games
        tail = pdf.tail(self.sequence_length)
        if len(tail) < self.sequence_length:
            raise ValueError(f"Not enough data for player {player_name} to build a sequence of length {self.sequence_length}")
        
        # Ensure all feature columns are present (fill missing with zeros)
        for col in self.feature_columns:
            if col not in tail.columns:
                print(f"⚠️ Column {col} not found in data, filling with zeros")
                tail[col] = 0.0
        
        # Ensure all columns are present and in the correct order
        X2d = tail[self.feature_columns].copy()
        
        # Log the first row of features for debugging
        print("\n=== SAMPLE FEATURE VALUES ===")
        for i, (col, val) in enumerate(zip(self.feature_columns, X2d.iloc[0])):
            if i < 10:  # Only show first 10 features to avoid cluttering the output
                print(f"{col}: {val:.4f}")
        
        # Convert to numpy array
        X2d = X2d.to_numpy(dtype=float)
        
        print(f"\n=== SCALING FEATURES ===")
        print(f"Input shape: {X2d.shape}")
        if self.scaler is not None:
            print(f"Scaler expects {self.scaler.n_features_in_} features")
            try:
                X2d = self.scaler.transform(X2d)
                print("✅ Scaling successful")
            except Exception as e:
                print(f"❌ Error in scaling: {e}")
                print(f"Input shape: {X2d.shape}, Expected features: {self.scaler.n_features_in_}")
                # Try to identify which features might be causing issues
                if hasattr(self.scaler, 'feature_names_in_'):
                    print("Scaler was fit on features:", self.scaler.feature_names_in_)
                raise
        else:
            print("⚠️ No scaler available, using raw features")
        
        # Reshape for the model (batch_size, sequence_length, num_features)
        X2d = X2d.reshape((1, self.sequence_length, len(self.feature_columns)))
        
        return X2d

    def preprocess_input(self, player_data: Dict[str, Any]) -> np.ndarray:
        """
        Normalize inputs into a 3D tensor (1, sequence_length, n_features)
        with features aligned to self.feature_columns and scaled by self.scaler.
        """
        import numpy as np
        import pandas as pd
        import warnings

        exp_feat = len(self.feature_columns)
        seq_len = self.sequence_length
        scaler_features = getattr(self.scaler, "n_features_in_", exp_feat)

        # --- helpers ---
        def align_df(df: pd.DataFrame) -> pd.DataFrame:
            """Add missing cols (0.0), drop extras, reorder to self.feature_columns."""
            missing = [c for c in self.feature_columns if c not in df.columns]
            for c in missing:
                df[c] = 0.0
            df = df[self.feature_columns]  # drop extras + reorder
            df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            return df

        def ensure_seq_len(X2d: np.ndarray) -> np.ndarray:
            """Trim or pad (repeat last row) to seq_len rows."""
            nrows = X2d.shape[0]
            if nrows > seq_len:
                return X2d[-seq_len:, :]
            if nrows < seq_len:
                if nrows == 0:
                    # no data: fill all zeros
                    return np.zeros((seq_len, X2d.shape[1]), dtype=float)
                last = X2d[-1:, :]
                pad_rows = np.repeat(last, seq_len - nrows, axis=0)
                return np.vstack([X2d, pad_rows])
            return X2d

        def ensure_feat_count(X2d: np.ndarray, n_features: int) -> np.ndarray:
            """Pad with zeros or truncate columns to n_features."""
            cur = X2d.shape[1]
            if cur == n_features:
                return X2d
            if cur > n_features:
                return X2d[:, :n_features]
            # pad zeros on the right
            pad = np.zeros((X2d.shape[0], n_features - cur), dtype=float)
            return np.hstack([X2d, pad])

        # --- build a 2D matrix (seq_len, exp_feat) ---
        if "feature_sequence" in player_data:
            fs = player_data["feature_sequence"]
            if not isinstance(fs, pd.DataFrame):
                raise ValueError("feature_sequence must be a pandas DataFrame")

            # Keep only last seq_len rows (assume already time-ordered; if not, caller must sort)
            fs_aligned = align_df(fs)
            X2d = fs_aligned.to_numpy(dtype=float)
            X2d = ensure_seq_len(X2d)                   # (seq_len, exp_feat)

        elif "raw_sequence" in player_data:
            # Expect list[list[feature]], length ≈ seq_len
            X2d = np.array(player_data["raw_sequence"], dtype=float)
            if X2d.ndim != 2:
                raise ValueError("raw_sequence must be a 2D array-like (timesteps x features)")
            # first normalize feature count to expected model features list
            X2d = ensure_feat_count(X2d, exp_feat)
            X2d = ensure_seq_len(X2d)

        elif "player_name" in player_data:
            # Server-side sequence builder
            as_of_date = player_data.get("game_date")
            X = self.build_sequence(
                player_name=player_data["player_name"],
                as_of_date=as_of_date,
                csv_path=self.default_csv
            )
            # build_sequence might return 3D or 2D; normalize to 2D
            if isinstance(X, np.ndarray) and X.ndim == 3:
                # (1, seq_len, n_features) -> (seq_len, n_features)
                X2d = X[0]
            else:
                # assume 2D already
                X2d = np.array(X, dtype=float)

            # Safety: enforce expected dims
            X2d = ensure_feat_count(X2d, exp_feat)
            X2d = ensure_seq_len(X2d)

        else:
            raise ValueError("Provide 'player_name', 'feature_sequence' (DataFrame), or 'raw_sequence'.")

        # --- scale (2D only!) ---
        if self.scaler is not None:
            # Ensure scaler and features agree
            if exp_feat != scaler_features:
                # If this triggers, your features list doesn't match the scaler’s fit.
                # Prefer fixing training to save a features_list.json and load it here.
                warnings.warn(
                    f"Mismatch: feature list has {exp_feat}, scaler expects {scaler_features}. "
                    "Adjusting features to scaler size."
                )
            # Match scaler feature count for transform
            X2d_for_scaler = ensure_feat_count(X2d, scaler_features)
            X2d_scaled = self.scaler.transform(X2d_for_scaler)
            # If scaler had different size, pad/truncate back to model expected feature count
            X2d_scaled = ensure_feat_count(X2d_scaled, exp_feat)
        else:
            warnings.warn("No scaler found, using unscaled features")
            X2d_scaled = X2d.astype(np.float32, copy=False)

        # Final shape: (1, seq_len, exp_feat)
        X3d = X2d_scaled.reshape(1, seq_len, exp_feat)
        return X3d

    
    def _get_stat_units(self, stat: str) -> str:
        """Get the units for a given statistic."""
        units = {
            'PTS': 'points',
            'AST': 'assists',
            'REB': 'rebounds',
            'STL': 'steals',
            'BLK': 'blocks'
        }
        return units.get(stat, '')

    def predict(self, player_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the next game points for a player based on their historical data.
        
        Args:
            player_data (Dict[str, Any]): Dictionary containing player data.
        
        Returns:
            Dict[str, Any]: Dictionary containing the predicted points and metadata.
        """
        print("\n=== Starting points prediction ===")
        print(f"Input data: {player_data}")
        
        try:
            # Preprocess input data
            print("\nPreprocessing input data...")
            X = self.preprocess_input(player_data)
            
            # Make prediction
            print("\nMaking prediction...")
            prediction = self.model.predict(X)
            predicted_points = float(prediction[0][0])
            
            # Ensure the prediction is non-negative (since points can't be negative)
            predicted_points = max(0, predicted_points)
            
            print(f"✅ Points prediction successful. Predicted points: {predicted_points:.1f}")
            
            return {
                "success": True,
                "prediction": predicted_points,
                "player_name": player_data.get("player_name", ""),
                "stat": "PTS",
                "units": "points",
                "confidence": 0.7  # Could be made dynamic based on model confidence
            }
            
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "success": False, 
                "error": error_msg
            }

def get_predictor() -> ModelPredictor:
    """
    Get a singleton instance of the ModelPredictor.
    
    Returns:
        ModelPredictor: The singleton instance of ModelPredictor.
    """
    if not hasattr(get_predictor, "_instance"):
        get_predictor._instance = ModelPredictor()
    return get_predictor._instance
