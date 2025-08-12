# NBA Player Points Predictor - Deployment

This directory contains the Flask web application for serving the NBA Player Points Prediction RNN model.

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- The following files from the trained model:
  - `nba_player_performance_rnn_final.h5` - The trained RNN model
  - `feature_scaler.pkl` - The feature scaler used during training

## Setup

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <your-repository-url>
   cd SportsParlayMLIdea/Deployment
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Place model files**:
   - Copy `nba_player_performance_rnn_final.h5` and `feature_scaler.pkl` to the parent directory of the Deployment folder.
   - The app will look for these files in the parent directory by default.

## Running the Application

### Development Mode

To run the Flask development server:

```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Production Deployment

For production, it's recommended to use a production WSGI server like Gunicorn:

```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t nba-predictor .
   ```

2. Run the container:
   ```bash
   docker run -p 5000:5000 nba-predictor
   ```

## API Endpoints

### Prediction Endpoint
- **URL**: `/predict`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "playerName": "LeBron James",
    "IS_HOME": 1,
    "DAYS_REST": 2,
    "IS_BACK_TO_BACK": 0,
    "PLAYER_PTS_EMA_5": 25.5,
    "PLAYER_FGA_EMA_5": 20.3,
    "PLAYER_MIN_EMA_5": 36.2,
    "PLAYER_FG3M_EMA_5": 2.8,
    "DEF_OPP_PTS_EMA_5": 112.3,
    "DEF_OPP_FG_PCT_EMA_5": 0.475
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "prediction": 27.8,
    "confidence": 0.9
  }
  ```

### Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "status": "healthy"
  }
  ```

## Project Structure

```
Deployment/
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── static/               # Static files (CSS, JS, images)
│   └── style.css         # Stylesheet
├── templates/            # HTML templates
│   └── index.html        # Main web interface
└── README.md             # This file
```

## Troubleshooting

- **Model not found**: Ensure the model files are in the correct location (parent directory by default)
- **Dependency issues**: Make sure all dependencies are installed with the correct versions
- **Port in use**: Change the port in `app.py` if port 5000 is already in use

## License

This project is licensed under the MIT License - see the LICENSE file for details.
