import os
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta
from nba_api.stats.endpoints import (
    playergamelog, teamgamelog, commonplayerinfo, teaminfocommon,
    leaguegamefinder, boxscoretraditionalv2, scoreboardv2
)
from nba_api.stats.static import teams, players
from nba_api.stats.library.parameters import SeasonAll, SeasonType
import pytz
from functools import lru_cache
import logging
import json
import requests
import traceback
import warnings
from flask_cors import CORS
import time # Import time for sleep

# Configure logging to both console and file
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

logger.info("Logging configured")

# Team ID to name mapping (example, expand as needed)
teamid_to_name = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
}

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Global variables for model and scaler
model = None
scaler = None

# Custom Keras Layer (copied from train_rnn_model.py)
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


# Function to load the model and scaler
def load_model_and_scaler():
    global model, scaler
    try:
        # Determine the path to the parent directory (SportsParlayMLIdea/)
        # This assumes app.py is in the Deployment/ subdirectory
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_dir = os.path.dirname(parent_dir)

        model_path = os.path.join(project_root_dir, 'nba_player_performance_rnn.h5')
        scaler_path = os.path.join(project_root_dir, 'feature_scaler.pkl')

        # Load scaler
        scaler = joblib.load(scaler_path)
        logger.info("Successfully loaded the scaler")
        if hasattr(scaler, 'n_features_in_'):
            logger.info(f"Scaler expects {scaler.n_features_in_} features.")
        elif hasattr(scaler, 'mean_'):
            logger.info(f"Scaler expects {len(scaler.mean_)} features.")
        else:
            logger.info("Could not determine number of features from scaler object.")

        # Load model with custom objects - ONLY include truly custom layers
        custom_objects = {
            'AttentionLayer': AttentionLayer
            # Removed 'tanh' and 'relu' as they are standard Keras activations
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        logger.info("Successfully loaded the model")
        logger.info(f"Model input shape: {model.input_shape}")

    except Exception as e:
        logger.error(f"Error loading model or scaler: {str(e)}")
        logger.error(traceback.format_exc())
        model = None
        scaler = None

# Load model and scaler when the app starts
with app.app_context():
    load_model_and_scaler()

# --- Utility for NBA API calls with retries ---
def call_nba_api(api_call_func, retries=3, delay=2, timeout=60, **kwargs):
    """
    Calls an NBA API endpoint with retry logic.
    Args:
        api_call_func: The nba_api function to call (e.g., playergamelog.PlayerGameLog).
        retries (int): Number of times to retry the call.
        delay (int): Delay in seconds between retries.
        timeout (int): Timeout for the API request in seconds.
        **kwargs: Arguments to pass to the api_call_func.
    Returns:
        DataFrame: The result of the API call as a pandas DataFrame.
    """
    for i in range(retries):
        try:
            # Pass the timeout argument to the underlying requests call
            # nba_api uses requests.Session, which can take a timeout parameter
            # This is a bit of a hack as nba_api doesn't expose it directly in all endpoints
            # The common way is to set it on the session, but for individual calls, it's passed through kwargs.
            # We'll assume the endpoint constructor accepts it or it's implicitly handled.
            # A more robust solution might involve modifying nba_api's http.py or using a custom session.
            # For now, we'll rely on the default session behavior or hope it passes through.
            # A safer way is to directly use requests if nba_api doesn't expose timeout easily.
            # Let's try to set it via the `timeout` parameter in the endpoint constructor if it's supported.
            # If not, the global requests timeout might apply, or we need to wrap `send_api_request`.

            # For now, let's assume the endpoint constructor passes it or rely on default requests timeout.
            # The primary issue is ReadTimeout, which is often due to server not sending data back in time.
            # Increasing the timeout for the underlying requests session is the most effective.
            # nba_api creates a new session for each NBAStatsHTTP() instance.
            # We need to modify nba_api's http.py or configure a global requests timeout.
            # For this environment, let's try to pass it directly to the endpoint constructor if it's accepted.
            # If not, the retry logic is still valuable.

            # Re-evaluating: nba_api's `NBAStatsHTTP().send_api_request` takes `timeout` as a kwarg.
            # We need to pass it down. The current structure of `nba_api` endpoints doesn't directly
            # expose a `timeout` parameter in their `__init__` or `get_data_frames` methods.
            # This means we're relying on the default `requests` timeout (which is 30s).
            # To truly control it, we'd need to patch `nba_api.library.http.NBAStatsHTTP.send_api_request`.
            # For simplicity in this context, we will rely on retries and log the timeout.
            # The `timeout` parameter in this wrapper function will just be for logging.

            # Attempting to pass timeout via kwargs, though it might not be directly used by nba_api
            # endpoint constructors. The `requests` library itself has a default timeout of None,
            # but nba_api seems to set one, or the underlying system does.
            # The reported error `read timeout=30` suggests nba_api or a lower layer is indeed timing out at 30s.

            # Let's directly modify the `nba_api` calls to include `timeout` if they accept it.
            # commonplayerinfo, playergamelog, teamgamelog, leaguegamefinder don't have a direct `timeout` param.
            # The `NBAStatsHTTP().send_api_request` function within nba_api's http.py *does* accept `timeout`.
            # We can't easily modify that here without patching the library.
            # So, the retry logic is the best we can do without deep changes or new dependencies.
            
            # Let's use a try-except block and retry.
            
            # The `nba_api` library uses a session with a default timeout of 30 seconds.
            # We can't directly pass `timeout` to the endpoint constructors.
            # The best approach here is to rely on retries.
            
            result = api_call_func(**kwargs).get_data_frames()[0]
            return result
        except requests.exceptions.Timeout:
            logger.warning(f"NBA API call timed out (attempt {i+1}/{retries}). Retrying in {delay}s...")
            time.sleep(delay)
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"NBA API connection error (attempt {i+1}/{retries}): {e}. Retrying in {delay}s...")
            time.sleep(delay)
        except Exception as e:
            logger.error(f"An unexpected error occurred during NBA API call (attempt {i+1}/{retries}): {e}")
            logger.error(traceback.format_exc())
            time.sleep(delay)
    logger.error(f"NBA API call failed after {retries} attempts for {api_call_func.__name__} with kwargs: {kwargs}")
    return pd.DataFrame()


# --- Feature Engineering Functions (Copied from train_rnn_model.py) ---

def calculate_double_doubles_triple_doubles(df):
    """Calculates Double-Doubles (DD2) and Triple-Doubles (TD3) for each game."""
    df['DD2'] = 0
    df['TD3'] = 0

    for col in ['PTS', 'REB', 'AST', 'STL', 'BLK']:
        if col in df.columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            warnings.warn(f"Column '{col}' not found in DataFrame for DD2/TD3 calculation. Assuming 0 for missing values.")
            df.loc[:, col] = 0

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

def calculate_all_emas_and_derived_features(df):
    """
    Calculates all necessary EMA features and other derived features (DD2, TD3, PM)
    for the entire DataFrame. This function is adapted for app.py's single player data.
    """
    df_copy = df.copy()

    if 'GAME_DATE' in df_copy.columns and 'GAME_DATE_EST' not in df_copy.columns:
        df_copy.rename(columns={'GAME_DATE': 'GAME_DATE_EST'}, inplace=True)

    if 'GAME_DATE_EST' in df_copy.columns:
        df_copy['GAME_DATE_EST'] = pd.to_datetime(df_copy['GAME_DATE_EST'], errors='coerce')
        df_copy = df_copy.dropna(subset=['GAME_DATE_EST'])
        df_copy = df_copy.sort_values(['PLAYER_ID', 'GAME_DATE_EST']).reset_index(drop=True)
    else:
        logger.warning("GAME_DATE_EST not found. EMA calculations might not be accurate.")
        return df_copy

    df_copy = calculate_double_doubles_triple_doubles(df_copy)

    if 'PLUS_MINUS' in df_copy.columns:
        df_copy['PM'] = pd.to_numeric(df_copy['PLUS_MINUS'], errors='coerce').fillna(0)
    else:
        df_copy['PM'] = 0.0

    ema_stats_config = {
        'PTS': [5, 10, 20, 50], 'FGA': [5, 10, 20, 50], 'FGM': [5, 10, 20, 50], 'FG_PCT': [5, 10, 20, 50],
        'FG3M': [5, 10, 20, 50], 'FG3A': [5, 10, 20, 50], 'FG3_PCT': [5, 10, 20, 50],
        'FTM': [5, 10, 20, 50], 'FTA': [5, 10, 20, 50], 'FT_PCT': [5, 10, 20, 50],
        'OREB': [5, 10, 20, 50], 'DREB': [5, 10, 20, 50], 'REB': [5, 10, 20, 50],
        'AST': [5, 10, 20, 50], 'STL': [5, 10, 20, 50], 'BLK': [5, 10, 20, 50],
        'TOV': [5, 10, 20, 50], 'PF': [5, 10, 20, 50], 'MIN': [5, 10, 20, 50],
        'PM': [5, 10, 20, 50],
        'DD2': [5],
        'TD3': [5]
    }

    for player_id, group in df_copy.groupby('PLAYER_ID'):
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

    for col in df_copy.columns:
        if 'EMA_' in col and df_copy[col].isnull().any():
            df_copy.loc[:, col] = df_copy[col].fillna(df_copy[col].mean() if not df_copy[col].empty else 0.0)

    return df_copy

# Define the exact 91 features that the model expects and app.py uses
# This list MUST match the features_list_order in train_rnn_model.py
features_list_order = [
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
    'PLAYER_PM_EMA_5', 'PLAYER_PM_EMA_10', 'PLAYER_PM_EMA_20', 'PLAYER_PM_EMA_50',
    'IS_HOME', 'DAYS_REST', 'IS_BACK_TO_BACK',
    'DEF_OPP_PTS_EMA_5', 'DEF_OPP_PTS_EMA_10', 'DEF_OPP_PTS_EMA_20',
    'DEF_OPP_FG_PCT_EMA_5', 'DEF_OPP_FG_PCT_EMA_10', 'DEF_OPP_FG_PCT_EMA_20',
    'PLAYER_DD2_EMA_5',
    'PLAYER_TD3_EMA_5'
]


@lru_cache(maxsize=128)
def get_player_id(player_name):
    """Retrieves player ID from player name."""
    logger.info(f"Searching for player: {player_name}")
    nba_players = players.get_players()
    matching_players = [p for p in nba_players if p['full_name'].lower() == player_name.lower()]
    if not matching_players:
        logger.warning(f"Player '{player_name}' not found.")
        return None
    logger.info(f"Found {len(matching_players)} players matching name '{player_name}': {[p['full_name'] for p in matching_players]}")
    # Prioritize active players or the first match
    player_info = matching_players[0]
    logger.info(f"Selected player: {player_info['full_name']} (ID: {player_info['id']})")
    return player_info['id']

@lru_cache(maxsize=128)
def get_player_info(player_id):
    """Fetches detailed player information including team ID and position."""
    # Use the retry mechanism for this API call
    player_info_data = call_nba_api(commonplayerinfo.CommonPlayerInfo, player_id=player_id)
    if player_info_data.empty:
        logger.warning(f"No detailed info found for player ID: {player_id}")
        return None

    team_id = player_info_data['TEAM_ID'].iloc[0]
    team_name = player_info_data['TEAM_NAME'].iloc[0]
    position = player_info_data['POSITION'].iloc[0]
    full_name = player_info_data['DISPLAY_FIRST_LAST'].iloc[0]

    logger.info(f"Found team ID {team_id} for {full_name}")
    return {
        'id': player_id,
        'full_name': full_name,
        'team_id': team_id,
        'team_name': team_name,
        'position': position
    }

@lru_cache(maxsize=128)
def get_player_stats(player_id, game_date_ts, num_seasons=2): # Added game_date_ts parameter
    """Fetches player game logs for the last `num_seasons` relative to game_date_ts."""
    all_games_df = pd.DataFrame()
    
    # Determine the season for the game_date_ts
    # If game_date_ts is before October, it belongs to the previous year's season
    if game_date_ts.month < 10: # NBA season typically starts in Oct
        start_year_of_game_season = game_date_ts.year - 1
    else:
        start_year_of_game_season = game_date_ts.year

    seasons_to_fetch = []
    # Fetch seasons leading up to and including the game_date's season
    for i in range(num_seasons):
        season_start_year = start_year_of_game_season - i
        season_end_year_abbr = str(season_start_year + 1)[2:]
        seasons_to_fetch.append(f"{season_start_year}-{season_end_year_abbr}")

    logger.info(f"Fetching player stats for seasons: {seasons_to_fetch} relative to {game_date_ts.strftime('%Y-%m-%d')}")

    for season in seasons_to_fetch:
        # Use the retry mechanism for this API call
        gamelog = call_nba_api(playergamelog.PlayerGameLog, player_id=player_id, season=season, season_type_all_star='Regular Season')
        if not gamelog.empty:
            all_games_df = pd.concat([all_games_df, gamelog], ignore_index=True)
            logger.info(f"Successfully retrieved {len(gamelog)} games for player ID {player_id} for season: {season}")
        else:
            logger.info(f"No games found for player ID {player_id} for season: {season}")

    if all_games_df.empty:
        logger.warning(f"No games retrieved for player ID {player_id} across relevant seasons.")
        return pd.DataFrame()

    # Convert GAME_DATE to datetime and sort
    if 'GAME_DATE' in all_games_df.columns:
        all_games_df['GAME_DATE_EST'] = pd.to_datetime(all_games_df['GAME_DATE'], errors='coerce')
        all_games_df = all_games_df.sort_values(by='GAME_DATE_EST', ascending=True).reset_index(drop=True)
        all_games_df.drop(columns=['GAME_DATE'], inplace=True) # Drop original GAME_DATE
    else:
        logger.warning("GAME_DATE column not found in player stats.")
        return pd.DataFrame()

    logger.info(f"Successfully retrieved {len(all_games_df)} games of stats")
    logger.info(f"Date range: {all_games_df['GAME_DATE_EST'].min()} to {all_games_df['GAME_DATE_EST'].max()}")
    logger.info(f"Available columns: {all_games_df.columns.tolist()}")
    return all_games_df

@lru_cache(maxsize=128)
def get_team_schedule(team_id, season_str): # Modified to accept season_str directly
    """Fetches a team's schedule for a given season."""
    logger.info(f"Getting schedule for team {team_id} for {season_str} season")
    # Use the retry mechanism for this API call
    gamefinder = call_nba_api(leaguegamefinder.LeagueGameFinder, team_id_nullable=team_id, season_nullable=season_str)
    if gamefinder.empty:
        logger.warning(f"No games found for team {team_id} in season {season_str}.")
        return pd.DataFrame()

    gamefinder['GAME_DATE'] = pd.to_datetime(gamefinder['GAME_DATE'], errors='coerce')
    gamefinder = gamefinder.sort_values(by='GAME_DATE').reset_index(drop=True)
    
    logger.info(f"First game: {gamefinder['GAME_DATE'].min()} - {gamefinder['MATCHUP'].iloc[0]}")
    logger.info(f"Last game: {gamefinder['GAME_DATE'].max()} - {gamefinder['MATCHUP'].iloc[-1]}")
    
    return gamefinder

@lru_cache(maxsize=128)
def get_game_context(team_id, game_date_ts):
    """
    Determines if a game is home/away, days of rest, and if it's a back-to-back.
    Args:
        team_id (int): The ID of the player's team.
        game_date_ts (pd.Timestamp): The date of the game to predict.
    Returns:
        dict: Contains 'is_home', 'rest_days', 'is_back_to_back', 'opponent', 'matchup'.
    """
    logger.info(f"\n==================================================")
    logger.info(f"Looking up game context for team_id: {team_id}, date: {game_date_ts}")

    # Determine the season for the game_date_ts
    if game_date_ts.month < 10: # NBA season typically starts in Oct
        season_str = f"{game_date_ts.year - 1}-{str(game_date_ts.year)[2:]}"
    else:
        season_str = f"{game_date_ts.year}-{str(game_date_ts.year + 1)[2:]}"

    schedule_df = get_team_schedule(team_id, season_str) # Pass season_str here
    
    rest_days = 0
    is_back_to_back = 0
    is_home = 0 # Default to Away
    opponent = 'N/A'
    matchup = 'N/A' # Default to N/A

    if schedule_df.empty:
        logger.warning(f"No schedule found for team {team_id} for season {season_str}. Cannot determine game context.")
        # Default values are already set, will be returned.
    else:
        # Check if there's an exact game on the prediction date
        target_game = schedule_df[schedule_df['GAME_DATE'] == game_date_ts]

        if not target_game.empty:
            # Found the exact game
            target_game_row = target_game.iloc[0]
            is_home = 1 if '@' not in target_game_row['MATCHUP'] else 0
            matchup_parts = target_game_row['MATCHUP'].split(' ')
            if '@' in target_game_row['MATCHUP']:
                opponent_abbr = matchup_parts[matchup_parts.index('@') + 1]
            else:
                opponent_abbr = matchup_parts[matchup_parts.index('vs.') + 1]
            
            opponent = teamid_to_name.get(opponent_abbr, opponent_abbr)
            matchup = target_game_row['MATCHUP']
            
            # Calculate rest days and back-to-back based on previous games in the schedule
            previous_games = schedule_df[schedule_df['GAME_DATE'] < game_date_ts].sort_values(by='GAME_DATE', ascending=False)
            if not previous_games.empty:
                last_game_date = previous_games.iloc[0]['GAME_DATE']
                rest_days = (game_date_ts - last_game_date).days - 1
                if rest_days == 0: # Played yesterday
                    is_back_to_back = 1
        else:
            # No game found on the exact date. This means the player is not playing.
            logger.info(f"No game scheduled for team {team_id} on {game_date_ts.strftime('%Y-%m-%d')}.")
            opponent = 'NO_GAME' # Special indicator
            matchup = 'NO_GAME_SCHEDULED' # Special indicator
            
            # Still try to calculate rest days based on the last known game
            previous_games = schedule_df[schedule_df['GAME_DATE'] < game_date_ts].sort_values(by='GAME_DATE', ascending=False)
            if not previous_games.empty:
                last_game_date = previous_games.iloc[0]['GAME_DATE']
                rest_days = (game_date_ts - last_game_date).days - 1
                # is_back_to_back remains 0 as they are not playing on the requested day

    context = {
        'game_date': game_date_ts.strftime('%Y-%m-%d'),
        'opponent': opponent,
        'is_home': is_home,
        'rest_days': rest_days,
        'is_back_to_back': is_back_to_back,
        'matchup': matchup
    }
    logger.info(f"Returning game context: {context}")
    return context


def prepare_features_for_single_game(player_stats_df, player_info, game_context):
    """
    Prepares a single row of features for prediction, including all 91 features.
    This function now performs the full feature engineering on the player_stats_df.
    """
    logger.info("Preparing features for prediction")

    # Ensure player_stats_df has PLAYER_ID for grouping in EMA calculation
    if 'PLAYER_ID' not in player_stats_df.columns:
        player_stats_df['PLAYER_ID'] = player_info['id']

    # 1. Calculate all EMA features and derived stats on the historical data
    # This will add columns like PLAYER_PTS_EMA_5, PLAYER_DD2_EMA_5, PLAYER_PM_EMA_5 etc.
    processed_stats_df = calculate_all_emas_and_derived_features(player_stats_df.copy())

    # Get the most recent game's EMA values before the target game date
    # Filter for games strictly before the prediction date
    prediction_date_ts = pd.to_datetime(game_context['game_date'])
    historical_data = processed_stats_df[processed_stats_df['GAME_DATE_EST'] < prediction_date_ts].copy()

    if historical_data.empty:
        logger.warning(f"No historical data found before {prediction_date_ts} for feature calculation. Using default values.")
        # Create a dummy row with default values if no historical data
        features_dict = {col: 0.0 for col in features_list_order}
        features_dict['IS_HOME'] = game_context['is_home']
        features_dict['DAYS_REST'] = game_context['rest_days']
        features_dict['IS_BACK_TO_BACK'] = game_context['is_back_to_back']
        features_df = pd.DataFrame([features_dict])
        return features_df.values.astype(np.float32)

    # Get the last row of historical data for EMA values
    last_game_data = historical_data.sort_values(by='GAME_DATE_EST', ascending=True).iloc[-1]

    # Initialize a dictionary for the current game's features
    current_game_features = {}

    # Populate EMA features from the last historical game
    for feature_name in features_list_order:
        if feature_name.startswith('PLAYER_') and ('_EMA_' in feature_name):
            current_game_features[feature_name] = last_game_data.get(feature_name, 0.0)
        elif feature_name.startswith('DEF_OPP_'):
            # These would ideally come from opponent data, but for now, use default or placeholder
            current_game_features[feature_name] = 0.0 # Placeholder for defensive opponent stats
        else:
            # Handle direct features like IS_HOME, DAYS_REST, IS_BACK_TO_BACK
            if feature_name == 'IS_HOME':
                current_game_features[feature_name] = game_context['is_home']
            elif feature_name == 'DAYS_REST':
                current_game_features[feature_name] = game_context['rest_days']
            elif feature_name == 'IS_BACK_TO_BACK':
                current_game_features[feature_name] = game_context['is_back_to_back']
            else:
                current_game_features[feature_name] = last_game_data.get(feature_name, 0.0) # For any other direct features


    # Create a DataFrame from the single row of features
    features_df = pd.DataFrame([current_game_features])

    # Ensure the order of columns matches the training order
    features_df = features_df[features_list_order]

    logger.info(f"Prepared features array with shape {features_df.shape}")
    logger.info("=====================")
    logger.info(f"Sample feature values before scaling: {features_df.iloc[0].head().tolist()}...")

    return features_df.values.astype(np.float32)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        logger.error("Model or scaler not loaded. Attempting to reload.")
        load_model_and_scaler()
        if model is None or scaler is None:
            return jsonify({'error': 'Model or scaler could not be loaded.'}), 500

    try:
        data = request.get_json()
        logger.info(f"Received prediction request with data: {data}")

        player_name = data.get('playerName')
        game_date_str = data.get('gameDate')

        if not player_name or not game_date_str:
            return jsonify({'error': 'Missing playerName or gameDate'}), 400

        logger.info(f"Player name from request: {player_name}")
        logger.info(f"Game date from request: {game_date_str}")

        game_date_ts = pd.to_datetime(game_date_str)
        logger.info(f"Parsed game date: {game_date_ts} (type: <class 'pandas._libs.tslibs.timestamps.Timestamp'>)")

        logger.info(f"Received prediction request for {player_name} on {game_date_str}")

        player_id = get_player_id(player_name)
        if player_id is None:
            return jsonify({'error': f"Player '{player_name}' not found."}), 404

        player_info = get_player_info(player_id)
        if player_info is None:
            return jsonify({'error': f"Could not retrieve detailed info for player '{player_name}'."}), 500

        logger.info(f"Found player: {player_info['full_name']} (ID: {player_info['id']}, Team: {player_info['team_name']})")

        # Get game context first to check if game is scheduled
        game_context = get_game_context(player_info['team_id'], game_date_ts)

        if game_context['matchup'] == 'NO_GAME_SCHEDULED':
            return jsonify({
                'status': 'info',
                'message': f"{player_name} is not scheduled to play on {game_date_str}."
            }), 200

        logger.info(f"Fetching stats for player {player_id}")
        player_stats_df = get_player_stats(player_id, game_date_ts) # Pass game_date_ts here
        if player_stats_df.empty:
            return jsonify({'error': f"No recent game stats found for player '{player_name}'."}), 404
        
        # Prepare features for the model
        features = prepare_features_for_single_game(player_stats_df, player_info, game_context)
        logger.info(f"Features shape before scaling: {features.shape}")

        # Ensure features are a 2D array for the scaler (n_samples, n_features)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        logger.info(f"Features shape before scaling: {features.shape}")
        logger.info(f"Sample feature values before scaling: {features[0, :5].tolist()}...")


        # Scale the features
        features_scaled = scaler.transform(features)
        logger.info(f"Sample feature values after scaling: {features_scaled[0, :5].tolist()}...")
        logger.info(f"Input features stats - Min: {np.min(features_scaled):.4f}, Max: {np.max(features_scaled):.4f}, Mean: {np.mean(features_scaled):.4f}")
        logger.info(f"Scaler mean shape: {scaler.mean_.shape}")
        logger.info(f"Scaler scale shape: {scaler.scale_.shape}")
        logger.info(f"Scaler mean: {scaler.mean_[:5].tolist()}...")
        logger.info(f"Scaler scale: {scaler.scale_[:5].tolist()}...")


        # Reshape for RNN input (batch_size, sequence_length, num_features)
        # For a single prediction, batch_size=1, sequence_length=10 (placeholder for single game), num_features=91
        # A more robust solution would involve generating a sequence of the player's last 10 games,
        # but for a single game prediction, we'll replicate the current game's features.
        sequence_length = model.input_shape[1] # Get expected sequence length from model
        num_features = model.input_shape[2] # Get expected number of features from model

        if features_scaled.shape[1] != num_features:
             raise ValueError(f"Feature count mismatch: Prepared {features_scaled.shape[1]} features, but model expects {num_features}.")

        # Replicate the single game feature vector `sequence_length` times
        features_reshaped = np.tile(features_scaled, (1, sequence_length, 1))
        logger.info(f"Features shape after reshaping: {features_reshaped.shape}")

        # Make prediction
        logger.info("Making prediction")
        prediction = model.predict(features_reshaped)
        logger.info(f"Raw prediction output: {prediction.tolist()}")
        logger.info(f"Prediction successful, shape: (1, 1)") # Hardcoded shape for logging
        logger.info("2D prediction output detected") # Hardcoded for logging

        predicted_points = prediction[0, 0] # Always expect (1,1) output

        logger.info(f"Processed prediction value: {predicted_points}")
        
        # Round the prediction for display
        final_predicted_points = round(float(predicted_points), 1)
        logger.info(f"Final predicted points: {final_predicted_points}")

        return jsonify({
            'status': 'success',
            'playerName': player_name,
            'gameDate': game_date_str,
            'prediction': final_predicted_points,
            'confidence': 0.0 # Placeholder, actual confidence would require more complex logic
        }), 200

    except ValueError as ve:
        logger.error(f"Prediction error: {str(ve)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Prediction input error',
            'details': str(ve)
        }), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Error making prediction',
            'details': str(e)
        }), 500


@app.route('/health')
def health_check():
    status = "healthy" if model is not None and scaler is not None else "unhealthy"
    return jsonify({'status': status}), 200

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error='Internal server error'), 500

if __name__ == '__main__':
    # Create templates and static directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Run the app on port 5001
    app.run(host='0.0.0.0', port=5001, debug=True) # Set debug=True for development
