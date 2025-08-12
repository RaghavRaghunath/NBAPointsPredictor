# NBA Sports Parlay ML

A machine learning project for predicting NBA game outcomes and player performances to optimize sports parlay betting strategies.

## Project Structure

```
SportsParlayMLIdea/
├── data/               # Data files (raw and processed)
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── src/                # Source code
│   ├── data/          # Data collection and processing
│   ├── models/        # Model training and evaluation
│   └── utils/         # Utility functions and helpers
├── deployment/        # Deployment code and web app
└── tests/             # Test files
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run data collection and processing:
   ```bash
   python src/data/data_processing.py
   ```
2. Train the model:
   ```bash
   python src/models/train.py
   ```
3. Run the web app:
   ```bash
   python deployment/app.py
   ```

## Data

- Raw data is stored in `data/raw/`
- Processed data is stored in `data/processed/`
- Models are saved in `data/models/`

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
