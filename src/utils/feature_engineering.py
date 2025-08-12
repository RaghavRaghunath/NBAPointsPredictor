import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

def calculate_double_doubles_triple_doubles(df):
    """Calculates Double-Doubles (DD2) and Triple-Doubles (TD3) for each game."""
    df = df.copy()
    df['DD2'] = 0
    df['TD3'] = 0

    # Ensure relevant columns are numeric and fill NaNs with 0 for calculation
    for col in ['PTS', 'REB', 'AST', 'STL', 'BLK']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            warnings.warn(f"Column '{col}' not found in DataFrame for DD2/TD3 calculation. Assuming 0 for missing values.")
            df[col] = 0

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
            df.at[index, 'DD2'] = 1
        if double_digit_stats >= 3:
            df.at[index, 'TD3'] = 1
    return df

def calculate_all_emas_and_derived_features(df):
    """
    Calculates all necessary EMA features and other derived features (DD2, TD3, PM)
    for the entire DataFrame. This should be done BEFORE splitting data.
    """
    df_copy = df.copy()

    # Find date column (could be GAME_DATE or GAME_DATE_EST)
    date_col = 'GAME_DATE' if 'GAME_DATE' in df_copy.columns else 'GAME_DATE_EST'
    
    if date_col in df_copy.columns:
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        df_copy = df_copy.dropna(subset=[date_col])
        df_copy = df_copy.sort_values(['PLAYER_NAME', date_col]).reset_index(drop=True)
    else:
        warnings.warn("GAME_DATE or GAME_DATE_EST not found. EMA calculations might not be accurate.")
        return df_copy

    # Calculate DD2 and TD3 if not already present
    if 'DD2' not in df_copy.columns or 'TD3' not in df_copy.columns:
        df_copy = calculate_double_doubles_triple_doubles(df_copy)

    # Handle PLUS_MINUS to PM
    if 'PLUS_MINUS' in df_copy.columns and 'PM' not in df_copy.columns:
        df_copy['PM'] = pd.to_numeric(df_copy['PLUS_MINUS'], errors='coerce').fillna(0)
    elif 'PM' not in df_copy.columns:
        df_copy['PM'] = 0.0

    ema_stats_config = {
        'PTS': [5, 10, 20, 50], 'FGA': [5, 10, 20, 50], 'FGM': [5, 10, 20, 50], 'FG_PCT': [5, 10, 20, 50],
        'FG3M': [5, 10, 20, 50], 'FG3A': [5, 10, 20, 50], 'FG3_PCT': [5, 10, 20, 50],
        'FTM': [5, 10, 20, 50], 'FTA': [5, 10, 20, 50], 'FT_PCT': [5, 10, 20, 50],
        'OREB': [5, 10, 20, 50], 'DREB': [5, 10, 20, 50], 'REB': [5, 10, 20, 50],
        'AST': [5, 10, 20, 50], 'STL': [5, 10, 20, 50], 'BLK': [5, 10, 20, 50],
        'TOV': [5, 10, 20, 50], 'PF': [5, 10, 20, 50], 'MIN': [5, 10, 20, 50],
        'PM': [5, 10, 20, 50],
        'DD2': [5, 10, 20],  # Added more windows for DD2
        'TD3': [5, 10, 20]   # Added more windows for TD3
    }

    # Iterate through players to calculate EMAs
    for player_name, group in tqdm(df_copy.groupby('PLAYER_NAME'), desc="Calculating Player EMAs"):
        for stat, windows in ema_stats_config.items():
            if stat in group.columns:
                for window in windows:
                    ema_col_name = f'PLAYER_{stat}_EMA_{window}'
                    df_copy.loc[group.index, ema_col_name] = group[stat].shift(1).ewm(
                        span=window, min_periods=1, adjust=False
                    ).mean()
            else:
                for window in windows:
                    ema_col_name = f'PLAYER_{stat}_EMA_{window}'
                    df_copy[ema_col_name] = 0.0

    # Fill any remaining NaNs in EMA columns
    for col in df_copy.columns:
        if 'EMA_' in col and df_copy[col].isnull().any():
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean() if not df_copy[col].empty else 0.0)

    return df_copy
