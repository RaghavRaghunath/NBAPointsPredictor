#!/bin/bash

# Create a virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p static/css
mkdir -p templates

# Copy model files (if they exist in the parent directory)
echo "Checking for model files..."
if [ -f "../nba_player_performance_rnn_final.h5" ] && [ -f "../feature_scaler.pkl" ]; then
    echo "Found model files in parent directory."
else
    echo "WARNING: Model files not found in parent directory."
    echo "Please make sure the following files exist in the parent directory:"
    echo "- nba_player_performance_rnn_final.h5"
    echo "- feature_scaler.pkl"
fi

echo ""
echo "Setup complete!"
echo "To start the development server, run:"
echo "  source venv/bin/activate  # On Windows: .\\venv\\Scripts\\activate"
echo "  python app.py"
