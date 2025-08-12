import tensorflow as tf
import joblib
import os
import sys # Import sys for path modification

# Custom Keras Layer (copied from train_rnn_model.py)
# This is necessary for loading the model if it contains this custom layer
class AttentionLayer(tf.keras.layers.Layer):
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
        e = tf.tanh(tf.matmul(x, self.W))
        a = tf.nn.softmax(e, axis=1)
        context = tf.reduce_sum(x * a, axis=1)
        last_step = x[:, -1, :]
        return context * last_step
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config


def inspect_model_and_scaler():
    # Determine the project root directory
    # This assumes the script is located in the project root (e.g., SportsParlayMLIdea/)
    # If it's run from Deployment/, it will still correctly find the parent.
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If inspect_model.py is in Deployment/, go up one level.
    # Otherwise, assume it's already in the project root.
    if os.path.basename(current_script_dir).lower() == 'deployment':
        project_root_dir = os.path.dirname(current_script_dir)
    else:
        project_root_dir = current_script_dir

    # Add project_root_dir to sys.path to ensure AttentionLayer can be imported
    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)

    # Inspect the model
    model_path = os.path.join(project_root_dir, 'nba_player_performance_rnn.h5')
    print(f"\n=== Model Information ===")
    print(f"Model path: {model_path}")
    
    try:
        # Load model with custom objects - ONLY include truly custom layers
        custom_objects = {
            'AttentionLayer': AttentionLayer
            # Removed 'tanh' and 'relu' as they are standard Keras activations
        }
        model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
        print("Model loaded successfully.")
        print("Model input shape:", model.input_shape)
        print("Model output shape:", model.output_shape)
        print("Model summary:")
        model.summary()
    except Exception as e:
        print(f"Error loading model: {e}")
    
    # Inspect the scaler
    scaler_path = os.path.join(project_root_dir, 'feature_scaler.pkl')
    print(f"\n=== Scaler Information ===")
    print(f"Scaler path: {scaler_path}")
    
    try:
        scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully.")
        print(f"Scaler type: {type(scaler)}")
        if hasattr(scaler, 'mean_'):
            print(f"Number of features in scaler: {len(scaler.mean_)}")
            print(f"First 5 means: {scaler.mean_[:5]}")
            print(f"First 5 scales: {scaler.scale_[:5]}")
        elif hasattr(scaler, 'n_features_in_'): # For newer scikit-learn versions
            print(f"Number of features in scaler: {scaler.n_features_in_}")
        else:
            print("Could not determine number of features from scaler object (missing mean_ or n_features_in_).")
    except Exception as e:
        print(f"Error loading scaler: {e}")

if __name__ == '__main__':
    inspect_model_and_scaler()
