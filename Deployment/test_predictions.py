import requests
import pandas as pd
from datetime import datetime, timedelta
import random
import time
import json
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

# Configuration
# Using localhost for local testing - match the port with Flask server
FLASK_APP_URL = "http://127.0.0.1:5001/predict" 
NUM_TESTS = 50
RECENT_SEASONS_TO_CONSIDER = 3 # Look for games in the last 3 seasons

def get_random_player_and_game_date():
    """
    Selects a random active player and a random game date from their recent history.
    Returns (player_name, game_date_str, actual_pts) or None if no suitable data found.
    """
    nba_players = players.get_players()
    active_players = [p for p in nba_players if p['is_active']]

    if not active_players:
        print("No active players found.")
        return None

    # Try more times to find a player with sufficient valid game data
    for _ in range(100): # Increased attempts
        player_info = random.choice(active_players)
        player_id = player_info['id']
        player_name = player_info['full_name']

        all_games_df = pd.DataFrame()
        
        # Determine seasons to fetch based on current date, as we need to find past games
        current_year = datetime.now().year
        seasons_to_fetch = []
        for i in range(RECENT_SEASONS_TO_CONSIDER):
            season_start_year = current_year - i
            season_end_year_abbr = str(season_start_year + 1)[2:]
            seasons_to_fetch.append(f"{season_start_year}-{season_end_year_abbr}")

        for season in seasons_to_fetch:
            try:
                gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, season_type_all_star='Regular Season').get_data_frames()[0]
                if not gamelog.empty:
                    all_games_df = pd.concat([all_games_df, gamelog], ignore_index=True)
            except Exception as e:
                pass # Suppress warnings during random selection for cleaner output

        if all_games_df.empty:
            continue # Try another player

        # Filter for games that have PTS and GAME_DATE
        all_games_df = all_games_df.dropna(subset=['PTS', 'GAME_DATE'])
        if all_games_df.empty:
            continue

        all_games_df['GAME_DATE'] = pd.to_datetime(all_games_df['GAME_DATE'])
        
        # Select a random game from the player's actual game history
        # Ensure there are enough games for meaningful EMA calculation (e.g., at least 10 games)
        if len(all_games_df) > 10:
            # Pick a game that is not among the very first few, to ensure some historical data exists
            # Also, pick a game that is not the very last one, to allow for future prediction.
            # Let's pick a game from the middle 80% of their available games to ensure context.
            start_idx = min(10, len(all_games_df) // 5) # At least 10 games in, or 20% into season
            end_idx = max(start_idx + 1, len(all_games_df) - 1) # Ensure at least one game can be picked
            
            if end_idx <= start_idx: # Not enough games after filtering
                continue

            random_game = all_games_df.iloc[random.randint(start_idx, end_idx)] 
            game_date_str = random_game['GAME_DATE'].strftime('%Y-%m-%d')
            actual_pts = random_game['PTS']
            
            # Check if game_date is too close to today (future) or too far in past
            # We want to pick dates where the model should have data.
            # Let's target games within the last 2 full seasons.
            today = datetime.now()
            two_years_ago = today - timedelta(days=2 * 365)
            
            if random_game['GAME_DATE'] < two_years_ago:
                continue # Skip games too old

            print(f"Selected: {player_name} on {game_date_str} (Actual PTS: {actual_pts})")
            return player_name, game_date_str, actual_pts
    
    print("Could not find enough players with recent game data for testing after many attempts.")
    return None

def run_prediction_test():
    """
    Runs prediction tests for random players and dates, calculates accuracy,
    and reports server responsiveness.
    """
    predictions_data = []
    total_response_time = 0
    successful_predictions_count = 0
    skipped_predictions_count = 0
    failed_predictions_count = 0

    print(f"Starting {NUM_TESTS} prediction tests...")
    for i in range(NUM_TESTS):
        print(f"\n--- Test {i+1}/{NUM_TESTS} ---")
        player_date_data = get_random_player_and_game_date()
        if player_date_data is None:
            print("Skipping test due to inability to find suitable player/game data for this iteration.")
            skipped_predictions_count += 1
            continue
        
        player_name, game_date_str, actual_pts = player_date_data

        payload = {
            "playerName": player_name,
            "gameDate": game_date_str
        }

        try:
            start_time = time.time()
            response = requests.post(FLASK_APP_URL, json=payload, timeout=30) # Added timeout
            end_time = time.time()
            
            response_time = end_time - start_time
            total_response_time += response_time

            result = response.json()

            if response.status_code == 200:
                if result.get('status') == 'success' and result.get('prediction') is not None:
                    predicted_pts = result.get('prediction')
                    abs_error = abs(actual_pts - predicted_pts)
                    predictions_data.append({
                        'player_name': player_name,
                        'game_date': game_date_str,
                        'actual_pts': actual_pts,
                        'predicted_pts': predicted_pts,
                        'abs_error': abs_error,
                        'response_time': response_time
                    })
                    successful_predictions_count += 1
                    print(f"  Prediction: {predicted_pts:.1f}, Actual: {actual_pts:.1f}, Error: {abs_error:.1f} (Time: {response_time:.2f}s)")
                elif result.get('status') == 'info' and result.get('message'):
                    print(f"  Info: {result['message']} (Time: {response_time:.2f}s)")
                    skipped_predictions_count += 1
                else:
                    print(f"  Prediction failed (unexpected success status): {result.get('error', 'Unknown error')}. Result: {result} (Time: {response_time:.2f}s)")
                    failed_predictions_count += 1
            else:
                print(f"  Prediction failed (HTTP {response.status_code}): {result.get('error', response.status_code)}. Result: {result} (Time: {response_time:.2f}s)")
                failed_predictions_count += 1

        except requests.exceptions.Timeout:
            print(f"  Request timed out for {player_name} on {game_date_str}.")
            failed_predictions_count += 1
        except requests.exceptions.ConnectionError as e:
            print(f"  Network/Connection error for {player_name} on {game_date_str}: {e}")
            failed_predictions_count += 1
        except json.JSONDecodeError:
            print(f"  JSON decoding error for {player_name} on {game_date_str}. Response: {response.text}")
            failed_predictions_count += 1
        except Exception as e:
            print(f"  An unexpected error occurred for {player_name} on {game_date_str}: {e}")
            failed_predictions_count += 1

    if not predictions_data:
        print("\nNo successful predictions were made to calculate metrics.")
        return

    df_results = pd.DataFrame(predictions_data)

    overall_mae = df_results['abs_error'].mean()
    avg_response_time = total_response_time / successful_predictions_count if successful_predictions_count > 0 else 0

    # Define "accuracy" as prediction within +/- X points
    ACCURACY_THRESHOLD = 5.0 # points
    within_threshold_count = (df_results['abs_error'] <= ACCURACY_THRESHOLD).sum()
    percent_accurate = (within_threshold_count / successful_predictions_count) * 100 if successful_predictions_count > 0 else 0

    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    print(f"Total tests attempted: {NUM_TESTS}")
    print(f"Successful predictions: {successful_predictions_count}")
    print(f"Skipped predictions (e.g., no game scheduled, or no data found for random player selection): {skipped_predictions_count}")
    print(f"Failed predictions (errors from backend or network): {failed_predictions_count}")
    print(f"Overall Mean Absolute Error (MAE) for successful predictions: {overall_mae:.2f} points")
    print(f"Average Response Time for successful predictions: {avg_response_time:.2f} seconds per prediction")
    print(f"Percentage of successful predictions within +/- {ACCURACY_THRESHOLD} points: {percent_accurate:.2f}%")
    print("\nComparison to Model's Reported MAE (4.7 points):")
    if overall_mae <= 4.7:
        print(f"The observed MAE ({overall_mae:.2f}) is comparable to or better than the model's reported MAE of 4.7. This indicates the server is responding well and the model is performing as expected on this sample.")
    else:
        print(f"The observed MAE ({overall_mae:.2f}) is higher than the model's reported MAE of 4.7. This might suggest a discrepancy between training and inference performance, or the selected random sample is less representative.")

if __name__ == "__main__":
    run_prediction_test()

