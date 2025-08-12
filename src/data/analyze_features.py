import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from typing import Tuple
from sklearn.inspection import permutation_importance

def load_data(csv: str, target: str = 'PTS') -> Tuple[pd.DataFrame, list]:
    # Loading the data
    df = pd.read_csv(csv)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    rolling_patterns = ['_ROLL', '_EMA_', '_SMA_', 'LAG_', 'rolling_', 'shift_']
    if 'GAME_DATE' in df.columns and 'GAME_DATE_EST' not in df.columns:
        df.rename(columns={'GAME_DATE': 'GAME_DATE_EST'}, inplace=True)
    if 'GAME_DATE_EST' in df.columns:
        df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'])
    if 'PLAYER_NAME' in df.columns and 'GAME_DATE_EST' in df.columns:
        df = df.sort_values(['PLAYER_NAME', 'GAME_DATE_EST'])
    missing = df.columns[df.isnull().mean() > 0.5]
    if not missing.empty:
        df.drop(columns = missing, inplace = True)
    # Fill numeric columns with the mean of the column
    numbers = df.select_dtypes(include = ['float64', 'int64']).columns
    for col in numbers: 
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
    # Columns that are not feature columns
    # Add these to your exclude list in load_data()
    exclude = [
        'PLAYER_ID', 'GAME_ID', 'SEASON_ID', 'GAME_DATE_EST', 
        'MIN', 'MIN_STR', 'VIDEO_AVAILABLE', 'PLAYER_NAME', 'TEAM_ID',
        # Add these to prevent data leakage
        'PTS', 'FGM', 'FG3M', 'FTM', 'FGA', 'FTA', 'FG3A', 'PTS_PER_MIN'
    ]
    # Extract the feature columns
    features = [col for col in numeric_cols 
        if (any(pattern in col.upper() for pattern in rolling_patterns) or
            col.endswith('_L1') or col.endswith('_L2') or  # Common lag notation
            col.endswith('_L3') or col.endswith('_L4') or
            col.endswith('_L5') or col.endswith('_L6') or
            col.endswith('_L7') or col.endswith('_L8') or
            col.endswith('_L9') or col.endswith('_L10'))
        and col not in exclude 
        and col != target]

    
    X = df[features]
    y = df[target]

    valid_index = X.notna().all(axis=1) & y.notna()
    X = X[valid_index]
    y = y[valid_index]

    return (X,y), features

(X, y), features = load_data('nba_player_rolling_stats_single_pass_enhanced.csv', 'PTS')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_jobs = -1, random_state = 42)
print(rf.fit(X_train, y_train))
print(rf.score(X_test, y_test))

from sklearn.metrics import mean_absolute_error, r2_score

# After fitting the model
y_pred = rf.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R² Score: {r2:.6f}")
print(f"Mean Absolute Error: {mae:.2f} points")
print(f"Average points in test set: {y_test.mean():.2f}")

# Check feature importances
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10).to_string(index=False))

nb_clf = GaussianNB()
print(nb_clf.fit(X_train, y_train))
print(nb_clf.score(X_test, y_test))

importances = permutation_importance(nb_clf, X_test, y_test, n_repeats=30)
importances = dict(zip(features,importances.importances_mean))
importances = {k: v for k, v in sorted(importances.items(), key = lambda x: x[1], reverse=True)}
print(importances)
print("\nTop 10 most important features:")
print(feature_importance.head(10).to_string(index=False))